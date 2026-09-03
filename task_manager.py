"""任务管理器：SQLite 持久化 + asyncio.Queue 串行 worker。

- ComfyUI 不支持并发（决策4）：单 worker 逐个消费，max_concurrent 固定为 1。
- 任务生命周期：pending → running → completed / failed。
- 完成或失败时通过 on_complete 回调把任务行 dict 交给上层（事件重入队通知）。
- 重启恢复：running 且已有 run_id 的任务恢复轮询；pending 且未提交的重新提交。
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import parse_qs, urlparse

import httpx

from .mcp_client import ComfyuiMcpClient, ComfyuiMcpError

logger = logging.getLogger(__name__)

# R2 上传钩子：(png_bytes, origin_name) -> 公网URL；返回空串/抛异常均视为未上传
R2Uploader = Callable[[bytes, str], Awaitable[str]]

TASK_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    prompt TEXT,
    template TEXT,
    params_json TEXT,
    source_session TEXT,
    source_user TEXT,
    image_url TEXT,
    local_url TEXT,
    image_path TEXT,
    result_json TEXT,
    error TEXT,
    created_at TEXT,
    completed_at TEXT
);
"""

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

_ACTIVE_STATUSES = (STATUS_PENDING, STATUS_RUNNING)
_POLL_OK_STATUSES = ("queued", "pending", "running", "completed")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class TaskManager:
    def __init__(
        self,
        db_path: str | Path,
        images_dir: str | Path,
        client: ComfyuiMcpClient,
        comfyui_base_url: str = "http://127.0.0.1:8188",
        poll_interval: float = 3.0,
        task_timeout: float = 900.0,
        on_complete: Callable[[dict], Awaitable[None]] | None = None,
        r2_uploader: R2Uploader | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.images_dir = Path(images_dir)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.client = client
        self.comfyui_base_url = comfyui_base_url.rstrip("/")
        self.poll_interval = poll_interval
        self.task_timeout = task_timeout
        self.on_complete = on_complete
        self.r2_uploader = r2_uploader

        self._queue: asyncio.Queue[int] = asyncio.Queue()
        self._worker: asyncio.Task | None = None
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute(TASK_SCHEMA)
            self._migrate()
            self._db.commit()

    def _migrate(self) -> None:
        """旧库补列（如 result_json），已存在则跳过。"""
        cols = {r[1] for r in self._db.execute("PRAGMA table_info(tasks)")}
        if "result_json" not in cols:
            self._db.execute("ALTER TABLE tasks ADD COLUMN result_json TEXT")

    # ── 数据库操作 ──────────────────────────────────────────

    def _execute(self, sql: str, args: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._db.execute(sql, args)
            self._db.commit()
            return cur

    def _query_one(self, sql: str, args: tuple = ()) -> dict | None:
        with self._lock:
            row = self._db.execute(sql, args).fetchone()
        return dict(row) if row else None

    def _query_all(self, sql: str, args: tuple = ()) -> list[dict]:
        with self._lock:
            rows = self._db.execute(sql, args).fetchall()
        return [dict(r) for r in rows]

    def _update(self, task_id: int, **fields: Any) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        self._execute(
            f"UPDATE tasks SET {cols} WHERE id=?", (*fields.values(), task_id)
        )

    # ── 生命周期 ────────────────────────────────────────────

    async def start(self) -> None:
        """启动 worker 并恢复重启前的未完成任务。"""
        await self._recover()
        self._worker = asyncio.create_task(self._worker_loop(), name="comfyui-task-worker")

    async def stop(self) -> None:
        if self._worker:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None

    async def _recover(self) -> None:
        rows = self._query_all(
            "SELECT * FROM tasks WHERE status IN (?, ?)", _ACTIVE_STATUSES
        )
        for row in rows:
            if row["run_id"]:
                # 已提交 ComfyUI：恢复轮询（不重复提交，避免重复占 GPU）
                self._queue.put_nowait(row["id"])
            else:
                # 尚未提交：重置回 pending 重新提交
                self._update(row["id"], status=STATUS_PENDING)
                self._queue.put_nowait(row["id"])

    # ── 对外接口 ────────────────────────────────────────────

    async def submit(
        self,
        prompt: str,
        template: str,
        params: dict,
        source_session: str = "",
        source_user: str = "",
    ) -> int:
        """新建任务并入队，立即返回任务 id。"""
        cur = self._execute(
            "INSERT INTO tasks (status, prompt, template, params_json, source_session,"
            " source_user, created_at) VALUES (?,?,?,?,?,?,?)",
            (
                STATUS_PENDING,
                prompt,
                template,
                json.dumps(params, ensure_ascii=False),
                source_session,
                source_user,
                _now(),
            ),
        )
        task_id = int(cur.lastrowid)
        self._queue.put_nowait(task_id)
        return task_id

    def get_task(self, task_id: int) -> dict | None:
        return self._query_one("SELECT * FROM tasks WHERE id=?", (task_id,))

    def pending_count(self) -> int:
        """仍在排队/执行中的任务数（含当前正在跑的）。"""
        row = self._query_one(
            "SELECT COUNT(*) AS c FROM tasks WHERE status IN (?, ?)", _ACTIVE_STATUSES
        )
        return int(row["c"]) if row else 0

    def queue_overview(self) -> dict:
        """队列概览：执行中的任务 + 排队数。"""
        running = self._query_one(
            "SELECT * FROM tasks WHERE status=? ORDER BY id DESC LIMIT 1", (STATUS_RUNNING,)
        )
        row = self._query_one(
            "SELECT COUNT(*) AS c FROM tasks WHERE status=?", (STATUS_PENDING,)
        )
        return {"running": running, "pending": int(row["c"]) if row else 0}

    def list_tasks(
        self, limit: int = 50, offset: int = 0, status: str | None = None
    ) -> list[dict]:
        if status:
            return self._query_all(
                "SELECT * FROM tasks WHERE status=? ORDER BY id DESC LIMIT ? OFFSET ?",
                (status, limit, offset),
            )
        return self._query_all(
            "SELECT * FROM tasks ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset)
        )

    def count_tasks(self, status: str | None = None) -> int:
        if status:
            row = self._query_one(
                "SELECT COUNT(*) AS c FROM tasks WHERE status=?", (status,)
            )
        else:
            row = self._query_one("SELECT COUNT(*) AS c FROM tasks")
        return int(row["c"]) if row else 0

    def delete_task(self, task_id: int, remove_file: bool = True) -> bool:
        """删除任务记录；remove_file 时一并删本地图片副本（R2 对象保留）。"""
        t = self.get_task(task_id)
        if not t:
            return False
        if remove_file and t.get("image_path"):
            try:
                Path(t["image_path"]).unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"task#{task_id} 删除本地文件失败（忽略）: {e}")
        self._execute("DELETE FROM tasks WHERE id=?", (task_id,))
        return True

    # ── worker ──────────────────────────────────────────────

    async def _worker_loop(self) -> None:
        while True:
            task_id = await self._queue.get()
            try:
                await self._process(task_id)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # 兜底：worker 永不退出
                self._update(task_id, status=STATUS_FAILED, error=f"内部错误: {e}")
                await self._notify(task_id)
            finally:
                self._queue.task_done()

    async def _process(self, task_id: int) -> None:
        task = self.get_task(task_id)
        if not task or task["status"] == STATUS_COMPLETED:
            return

        self._update(task_id, status=STATUS_RUNNING)
        template = task["template"]
        params = json.loads(task["params_json"] or "{}")

        # 1) 提交（若重启恢复的任务已有 run_id 则跳过）
        run_id = task["run_id"]
        if not run_id:
            try:
                r = await self.client.run_template(template, params, wait=False)
            except ComfyuiMcpError as e:
                self._finish(task_id, STATUS_FAILED, error=str(e))
                await self._notify(task_id)
                return
            run_id = r.get("run_id") or ""
            if not run_id:
                self._finish(task_id, STATUS_FAILED, error=f"未返回 run_id: {r}")
                await self._notify(task_id)
                return
            self._update(task_id, run_id=run_id)

        # 2) 轮询结果（连续传输错误重试，超时失败）
        result: dict | None = None
        transport_errors = 0
        deadline = time.monotonic() + self.task_timeout
        while time.monotonic() < deadline:
            try:
                r = await self.client.get_template_result(template, run_id, wait=False)
                transport_errors = 0
            except ComfyuiMcpError as e:
                transport_errors += 1
                if transport_errors >= 3:
                    self._finish(task_id, STATUS_FAILED, error=f"轮询失败: {e}")
                    await self._notify(task_id)
                    return
                await asyncio.sleep(self.poll_interval)
                continue

            status = r.get("status", "")
            if status == "completed":
                result = r
                break
            if status not in _POLL_OK_STATUSES:
                # 服务端异常状态（含 unknown）
                self._finish(
                    task_id, STATUS_FAILED,
                    error=f"ComfyUI 状态异常: {status or json.dumps(r, ensure_ascii=False)[:200]}",
                )
                await self._notify(task_id)
                return
            await asyncio.sleep(self.poll_interval)
        else:
            self._finish(task_id, STATUS_FAILED, error=f"任务超时（>{self.task_timeout:.0f}s）")
            await self._notify(task_id)
            return

        # 3) 提取输出图片并下载本地副本
        outputs = result.get("outputs", {}) if isinstance(result, dict) else {}
        image_out = self._pick_image_output(outputs)
        local_url = (image_out or {}).get("url", "")
        if not local_url:
            self._finish(task_id, STATUS_FAILED, error=f"输出中没有图片: {outputs}")
            await self._notify(task_id)
            return

        url = local_url if local_url.startswith("http") else self.comfyui_base_url + local_url
        origin_name = self._origin_name(url)
        image_path = ""
        image_bytes = b""
        try:
            image_bytes = await self._fetch(url)
            dest = self.images_dir / f"task{task_id}_{origin_name}"
            dest.write_bytes(image_bytes)
            image_path = str(dest)
        except Exception as e:
            # 图片下载失败：任务仍算完成（URL 在），但不落本地副本
            logger.warning(f"task#{task_id} 图片下载/落盘失败（忽略）: {e}")

        # 图床URL：输出自带公网URL（旧图床链接节点还在时）优先；否则上传 R2
        image_url = self._pick_public_url(outputs)
        if not image_url and self.r2_uploader and image_bytes:
            try:
                image_url = await self.r2_uploader(image_bytes, origin_name) or ""
            except Exception as e:
                # R2 失败降级：任务仍成功，仅无图床URL
                logger.warning(f"task#{task_id} R2 上传失败（降级仅保留本地）: {e}")

        self._finish(
            task_id,
            STATUS_COMPLETED,
            local_url=local_url,
            image_path=image_path,
            image_url=image_url or None,
            result_json=json.dumps(result, ensure_ascii=False),
        )
        await self._notify(task_id)

    def _pick_image_output(self, outputs: dict) -> dict | None:
        """从 outputs 里挑图片输出：优先 SaveToGallery，其次任意 type=image。"""
        if not isinstance(outputs, dict):
            return None
        if isinstance(outputs.get("SaveToGallery"), dict):
            return outputs["SaveToGallery"]
        for v in outputs.values():
            if isinstance(v, dict) and (v.get("type") == "image" or v.get("url")):
                return v
        return None

    def _pick_public_url(self, outputs: dict) -> str:
        """输出中若自带公网 URL（图床链接节点），直接采用，跳过 R2 上传。"""
        for v in (outputs or {}).values():
            if not isinstance(v, dict):
                continue
            for field in ("url", "value"):
                s = str(v.get(field) or "")
                if s.startswith("https://"):
                    return s
                if s.startswith("http://") and not s.startswith(self.comfyui_base_url) and "127.0.0.1" not in s:
                    return s
        return ""

    @staticmethod
    def _origin_name(url: str) -> str:
        qs = parse_qs(urlparse(url).query)
        return Path(qs["filename"][0]).name if "filename" in qs else "image.png"

    @staticmethod
    async def _fetch(url: str) -> bytes:
        async with httpx.AsyncClient(timeout=120) as hc:
            resp = await hc.get(url)
            resp.raise_for_status()
            return resp.content

    def _finish(self, task_id: int, status: str, **fields: Any) -> None:
        fields["status"] = status
        fields["completed_at"] = _now()
        self._update(task_id, **fields)

    async def _notify(self, task_id: int) -> None:
        if self.on_complete:
            try:
                await self.on_complete(self.get_task(task_id))
            except Exception:
                pass  # 通知失败不影响任务状态

    def close(self) -> None:
        self._db.close()
