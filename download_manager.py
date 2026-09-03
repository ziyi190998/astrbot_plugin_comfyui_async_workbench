"""CivitAI 模型下载器：流式分块 + .part 断点续传 + 串行队列 + SQLite 任务记录。

参考 ComfyUI-Lora-Manager 的下载链路：
- URL 取版本 files[].downloadUrl，Authorization: Bearer <key>
- .part 临时文件 + Range 续传；服务器不支持续传(返回200而非206)时重下
- 完成后按大小校验 → rename → 写 metadata.json → 下载预览图
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import httpx

from .civitai_api import CivitaiApi, CivitaiError

if TYPE_CHECKING:
    from .model_manager import ModelManager

logger = logging.getLogger(__name__)

CHUNK = 4 * 1024 * 1024  # 4MB 分块
SPEED_INTERVAL = 1.0
DB_WRITE_INTERVAL = 3.0

# 版本 model.type（归一小写去空格）→ 默认文件夹（用户规则：绝大多数基模进 diffusion_models，
# lora 系进 loras/{base_model}；sdxl 等需进 checkpoints 的由用户手动选择）
_LORA_TYPES = {"lora", "locon", "dora", "lycoris"}
_SUPPORTED_FOLDERS = {"checkpoints", "diffusion_models", "loras"}
_SUB_SAFE = __import__("re").compile(r"[^A-Za-z0-9_\-\. ]+")


def sanitize_sub(sub: str) -> str:
    """子目录清洗：去非法字符、禁止 ..、限两层。"""
    parts = []
    for seg in (sub or "").replace("\\", "/").split("/"):
        seg = _SUB_SAFE.sub("", seg).strip().strip(".")
        if seg and seg not in ("..", "."):
            parts.append(seg)
    return "/".join(parts[:2])


def default_target(model_type: str, base_model: str) -> tuple[str, str]:
    """(文件夹, 子目录) 默认落盘规则：lora 系 → loras/{base_model}；其余 → diffusion_models/{base_model}。"""
    key = (model_type or "").lower().replace(" ", "").replace("_", "")
    base = sanitize_sub(base_model or "")
    if key in _LORA_TYPES:
        return "loras", base
    return "diffusion_models", base


def route_folder(model_type: str) -> str | None:
    """是否为支持的模型类型（用于拒绝 VAE/embedding 等）。"""
    key = (model_type or "").lower().replace(" ", "").replace("_", "")
    if key in _LORA_TYPES or key in ("checkpoint", "diffusionmodel", "unet"):
        return "ok"
    return None

DOWNLOAD_SCHEMA = """
CREATE TABLE IF NOT EXISTS downloads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version_id INTEGER,
    filename TEXT,
    folder TEXT,
    status TEXT NOT NULL DEFAULT 'queued',
    size INTEGER DEFAULT 0,
    received INTEGER DEFAULT 0,
    error TEXT,
    created_at TEXT,
    completed_at TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DownloadManager:
    def __init__(
        self,
        models_root: str | Path,
        db_path: str | Path,
        civitai_factory: Callable[[], CivitaiApi],
        model_manager: "ModelManager",
    ) -> None:
        self.root = Path(models_root)
        self.db_path = Path(db_path)
        self.civitai_factory = civitai_factory
        self.mm = model_manager
        self._tasks: dict[int, dict] = {}
        self._cancel: set[int] = set()
        self._sem = asyncio.Semaphore(1)  # 串行下载，避免挤占带宽
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute(DOWNLOAD_SCHEMA)
            self._db.commit()
            row = self._db.execute("SELECT MAX(id) AS m FROM downloads").fetchone()
        self._next_id = (row["m"] or 0) + 1
        # 历史任务载入内存（会话内可查询；上次未完成的任务不再自动续传，可手动重启）
        with self._lock:
            for r in self._db.execute("SELECT * FROM downloads ORDER BY id"):
                t = dict(r)
                t["speed"] = 0
                if t["status"] in ("queued", "downloading"):
                    t["status"] = "interrupted"  # 进程重启过的进行中任务
                self._tasks[t["id"]] = t

    # ── 对外接口 ────────────────────────────────────────────

    async def start(
        self, version_id: int, file_id: int | None = None,
        folder: str | None = None, sub: str | None = None,
    ) -> dict:
        """解析版本 → 选文件 → 建任务并后台执行。返回任务 dict。

        folder/sub 为用户手动指定的落盘位置（folder 必须是三个受支持文件夹之一）；
        不传时按默认规则：lora 系 → loras/{base_model}，其余 → diffusion_models/{base_model}。
        """
        api = self.civitai_factory()
        version = await api.get_version(version_id)
        files = version.get("files") or []
        if not files:
            raise ValueError("该版本没有可下载文件")
        entry = None
        if file_id:
            entry = next((f for f in files if f.get("id") == file_id), None)
        if entry is None:
            entry = next((f for f in files if f.get("primary")), files[0])
        url = entry.get("downloadUrl") or ""
        if not url:
            raise ValueError("文件缺少下载地址（可能为早期访问/需付费模型）")

        model_type = (version.get("model") or {}).get("type") or ""
        base_model = version.get("baseModel") or ""
        if not route_folder(model_type):
            raise ValueError(
                f"不支持的模型类型「{model_type}」，本插件仅管理 checkpoints / diffusion_models / loras"
            )
        if folder is not None:
            if folder not in _SUPPORTED_FOLDERS:
                raise ValueError(f"目标文件夹必须是 {sorted(_SUPPORTED_FOLDERS)} 之一")
            target_folder = folder
            target_sub = sanitize_sub(sub or "")
        else:
            target_folder, target_sub = default_target(model_type, base_model)

        filename = entry.get("name") or f"model_v{version_id}.safetensors"
        # 防路径穿越：文件名去目录部分
        filename = Path(filename).name

        target_dir = self.root / target_folder / target_sub
        target_dir.mkdir(parents=True, exist_ok=True)
        size = int((entry.get("sizeKB") or 0) * 1024)
        du = shutil.disk_usage(target_dir)
        need = size if size > 0 else 1024 * 1024 * 1024  # 大小未知按 1GB 预留
        if du.free < need * 1.05:
            raise ValueError(
                f"磁盘空间不足：{target_dir.drive} 剩余 {du.free / 2**30:.1f}GB，"
                f"需要约 {need * 1.05 / 2**30:.1f}GB"
            )
        if (target_dir / filename).exists():
            raise ValueError(f"目标目录已存在同名文件：{filename}")

        full_folder = "/".join(p for p in (target_folder, target_sub) if p)
        task_id = self._next_id
        self._next_id += 1
        task: dict[str, Any] = {
            "id": task_id, "version_id": version_id, "filename": filename,
            "folder": full_folder, "status": "queued", "size": size, "received": 0,
            "speed": 0, "error": "", "created_at": _now(), "completed_at": "",
            "url": url, "_version": version,
        }
        self._tasks[task_id] = task
        self._persist(task)
        asyncio.get_running_loop().create_task(self._run(task_id))
        return self.public(task)

    def list_tasks(self) -> list[dict]:
        return [self.public(t) for t in sorted(self._tasks.values(), key=lambda t: -t["id"])]

    def cancel(self, task_id: int) -> bool:
        t = self._tasks.get(task_id)
        if not t or t["status"] not in ("queued", "downloading"):
            return False
        self._cancel.add(task_id)
        return True

    def delete_task(self, task_id: int) -> bool:
        """删除下载记录（不删已下载的模型文件）；进行中的任务不可删。"""
        t = self._tasks.get(task_id)
        if not t or t["status"] in ("queued", "downloading"):
            return False
        del self._tasks[task_id]
        with self._lock:
            self._db.execute("DELETE FROM downloads WHERE id=?", (task_id,))
            self._db.commit()
        return True

    @staticmethod
    def public(t: dict) -> dict:
        return {k: v for k, v in t.items() if not k.startswith("_")}

    # ── 执行 ────────────────────────────────────────────────

    async def _run(self, task_id: int) -> None:
        async with self._sem:
            t = self._tasks.get(task_id)
            if not t or t["status"] != "queued":
                return
            t["status"] = "downloading"
            self._persist(t)
            api = self.civitai_factory()
            headers: dict[str, str] = {}
            if api.api_key:
                headers["Authorization"] = f"Bearer {api.api_key}"
            part = self.root / t["folder"] / (t["filename"] + ".part")
            final = self.root / t["folder"] / t["filename"]
            try:
                offset = part.stat().st_size if part.exists() else 0
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                last_t = time.monotonic()
                last_recv = offset
                last_db = last_t
                timeout = httpx.Timeout(30.0, read=120.0)
                async with httpx.AsyncClient(proxy=api.proxy, timeout=timeout, follow_redirects=True) as c:
                    async with c.stream("GET", t["url"], headers=headers) as resp:
                        if offset and resp.status_code == 200:
                            offset = 0  # 服务器不支持 Range，重下
                            t["received"] = 0
                        elif resp.status_code not in (200, 206):
                            raise CivitaiError(
                                f"下载失败（HTTP {resp.status_code}）"
                                f"{'，请检查 API Key 权限' if resp.status_code in (401, 403) else ''}"
                            )
                        clen = int(resp.headers.get("content-length") or 0)
                        if clen:
                            t["size"] = max(t["size"], clen + offset)
                        with open(part, "ab" if offset else "wb") as f:
                            async for chunk in resp.aiter_bytes(CHUNK):
                                if task_id in self._cancel:
                                    t["status"] = "cancelled"
                                    t["completed_at"] = _now()
                                    break
                                f.write(chunk)
                                t["received"] = offset + f.tell()
                                now = time.monotonic()
                                if now - last_t >= SPEED_INTERVAL:
                                    t["speed"] = int((t["received"] - last_recv) / (now - last_t))
                                    last_t, last_recv = now, t["received"]
                                if now - last_db >= DB_WRITE_INTERVAL:
                                    self._persist(t)
                                    last_db = now
                if t["status"] == "downloading":
                    if t["size"] and t["received"] < t["size"] * 0.98:
                        raise CivitaiError(
                            f"下载数据不完整（{t['received']}/{t['size']} 字节），.part 已保留可续传"
                        )
                    os.replace(part, final)
                    t["status"] = "completed"
                    t["speed"] = 0
                    t["completed_at"] = _now()
                    self._persist(t)
                    # 元数据 + 预览图
                    await self._post_download(t, final, api)
                    logger.info(f"[downloads] task#{task_id} 完成: {final}")
                else:
                    self._persist(t)
            except Exception as e:
                t["status"] = "failed"
                t["error"] = str(e)[:300]
                t["speed"] = 0
                t["completed_at"] = _now()
                self._persist(t)
                logger.error(f"[downloads] task#{task_id} 失败: {e}")
            finally:
                self._cancel.discard(task_id)

    async def _post_download(self, t: dict, final: Path, api: CivitaiApi) -> None:
        """下载完成后写 metadata.json 并尝试下载预览图。"""
        version = t.get("_version") or {}
        try:
            image, nsfw = self.mm._select_preview(version)
            updates = {
                "from_civitai": True, "civitai": version,
                "model_name": (version.get("model") or {}).get("name") or final.stem,
                "base_model": version.get("baseModel") or "",
                "preview_nsfw_level": nsfw,
                "last_checked_at": _now(),
                "size": final.stat().st_size,
            }
            rel = final.relative_to(self.root).as_posix()
            self.mm.save_metadata(rel, updates)
            if image and image.get("url"):
                await self.mm._download_preview(final, image["url"], api)
        except Exception as e:
            logger.warning(f"[downloads] task#{t['id']} 后处理失败（不影响下载）: {e}")

    # ── 持久化 ──────────────────────────────────────────────

    def _persist(self, t: dict) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO downloads (id, version_id, filename, folder, status,"
                " size, received, error, created_at, completed_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (t["id"], t["version_id"], t["filename"], t["folder"], t["status"],
                 t["size"], t["received"], t["error"], t["created_at"], t.get("completed_at") or ""),
            )
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()
