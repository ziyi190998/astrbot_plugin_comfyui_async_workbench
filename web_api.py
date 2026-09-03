"""ComfyUI MCP工作台 Web API：画廊/任务/元数据/删除（AstrBot WebUI 插件扩展路由）。

- 路由前缀：/api/plugin/plugins/extensions/<plugin_name>/<route>
- handler 统一用 astrbot.api.web 的 request / json_response / error_response
- 画廊预览本地文件优先（不依赖 r2.dev 公开状态）；R2 URL 作为字段展示
"""

from __future__ import annotations

import base64
import io
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from astrbot.api import logger
from astrbot.api.web import error_response, json_response, request

from .civitai_api import CivitaiError
from .png_metadata import parse_png_file

if TYPE_CHECKING:
    from .main import ComfyuiWorkbenchPlugin

PLUGIN_NAME = "astrbot_plugin_comfyui_async_workbench"
PAGE_SIZE_DEFAULT = 24
PAGE_SIZE_MAX = 100
THUMB_SIZE = 360  # 缩略图最长边像素
THUMB_JPEG_QUALITY = 72


class WebApiHandler:
    def __init__(self, plugin: "ComfyuiWorkbenchPlugin") -> None:
        self.plugin = plugin
        self.mgr = plugin._mgr

    def register_routes(self) -> None:
        routes = [
            ("tasks", self.handle_tasks, ["GET"], "任务列表（分页）"),
            ("task", self.handle_task, ["GET"], "任务详情"),
            ("image", self.handle_image, ["GET"], "任务图片文件"),
            ("delete", self.handle_delete, ["POST"], "删除任务（本地副本+记录）"),
            ("resend", self.handle_resend, ["POST"], "重发完成推送"),
            ("config", self.handle_get_config, ["GET"], "插件配置（分组）"),
            ("config/save", self.handle_save_config, ["POST"], "保存插件配置"),
            ("models/list", self.handle_models_list, ["GET"], "本地模型列表"),
            ("model/detail", self.handle_model_detail, ["GET"], "模型详情"),
            ("model/preview", self.handle_model_preview, ["GET"], "模型预览图"),
            ("model/delete", self.handle_model_delete, ["POST"], "删除模型及附属文件"),
            ("model/fetch", self.handle_model_fetch, ["POST"], "抓取单个模型 CivitAI 信息"),
            ("model/fetch-all", self.handle_model_fetch_all, ["POST"], "批量抓取文件夹（后台）"),
            ("model/fetch-progress", self.handle_fetch_progress, ["GET"], "批量抓取进度"),
            ("civitai/search", self.handle_civitai_search, ["GET"], "CivitAI 模型搜索"),
            ("civitai/version", self.handle_civitai_version, ["GET"], "CivitAI 版本详情"),
            ("civitai/model", self.handle_civitai_model, ["GET"], "CivitAI 模型版本列表"),
            ("civitai/image", self.handle_civitai_image, ["GET"], "CivitAI 图片代理"),
            ("download/start", self.handle_download_start, ["POST"], "开始下载模型"),
            ("download/list", self.handle_download_list, ["GET"], "下载任务列表"),
            ("download/cancel", self.handle_download_cancel, ["POST"], "取消下载"),
            ("download/delete", self.handle_download_delete, ["POST"], "删除下载记录"),
        ]
        for sub, handler, methods, desc in routes:
            self.plugin.context.register_web_api(
                f"/{PLUGIN_NAME}/{sub}", handler, methods, desc
            )
        logger.info(f"[{PLUGIN_NAME}] Web API 已注册: {[r[0] for r in routes]}")

    # ── 本地模型管理 ────────────────────────────────────────

    async def handle_models_list(self) -> Any:
        folder = request.query.get("folder", "")
        refresh = request.query.get("refresh", "") in ("1", "true")
        try:
            models = self.plugin._models.list_models(folder, refresh=refresh)
        except ValueError as e:
            return error_response(str(e))
        return json_response({"items": models, "total": len(models)})

    async def handle_model_detail(self) -> Any:
        rel = request.query.get("path", "")
        d = self.plugin._models.get_detail(rel)
        if not d:
            return error_response("模型不存在或路径非法", status_code=404)
        return json_response(d)

    async def handle_model_preview(self) -> Any:
        rel = request.query.get("path", "")
        mode = request.query.get("mode", "thumb")
        mm = self.plugin._models
        if mode == "video":
            uri = mm.video_data_uri(rel)
            if not uri:
                return error_response("视频预览不可用（非视频或超过 40MB）", status_code=404)
            return json_response({"data_uri": uri})
        uri = mm.preview_data_uri(rel)
        if not uri:
            return error_response("该模型没有可用预览", status_code=404)
        return json_response({"data_uri": uri})

    async def handle_model_delete(self) -> Any:
        body = await request.json() or {}
        rel = body.get("path") or ""
        try:
            deleted = self.plugin._models.delete(rel)
        except ValueError as e:
            return error_response(str(e))
        return json_response({"message": f"已删除 {len(deleted)} 个文件", "deleted": deleted})

    # ── CivitAI 抓取 / 搜索 ────────────────────────────────

    # 哈希匹配/版本查询固定走红站全量库；搜索按配置站点（蓝站=SFW 视图）
    def _civitai_red(self):
        from .civitai_api import RED_BASE, CivitaiApi
        return CivitaiApi(
            self.plugin.config.get("civitai_api_key", ""),
            RED_BASE,
            self.plugin.config.get("civitai_proxy", ""),
        )

    def _civitai_site(self):
        from .civitai_api import CivitaiApi
        return CivitaiApi(
            self.plugin.config.get("civitai_api_key", ""),
            self.plugin.config.get("civitai_base_url", "https://civitai.red/api/v1/"),
            self.plugin.config.get("civitai_proxy", ""),
        )

    async def handle_model_fetch(self) -> Any:
        body = await request.json() or {}
        rel = body.get("path") or ""
        force = bool(body.get("force"))
        try:
            result = await self.plugin._models.fetch_civitai(rel, self._civitai_red(), force=force)
        except ValueError as e:
            return error_response(str(e))
        except CivitaiError as e:
            return error_response(str(e))
        msg = {
            "updated": "已更新 CivitAI 元数据",
            "not_found": "CivitAI 未收录该模型（已标记，不再重复查询）",
            "skipped": result.get("reason") or "跳过",
        }[result.get("status", "skipped")]
        return json_response({"result": result.get("status"), "message": msg,
                              "preview": result.get("preview", False)})

    # 批量抓取进度（模块级，单实例运行）
    _fetch_progress: dict = {"running": False}

    async def handle_model_fetch_all(self) -> Any:
        body = await request.json() or {}
        folder = body.get("folder") or ""
        force = bool(body.get("force"))
        if self._fetch_progress.get("running"):
            return error_response("已有批量抓取在进行中")
        try:
            models = self.plugin._models.list_models(folder)
        except ValueError as e:
            return error_response(str(e))
        pending = [m for m in models if force or not m["from_civitai"]]
        if not pending:
            return json_response({"message": "没有需要抓取的模型（均已有元数据）", "started": False})

        import asyncio

        self._fetch_progress = {
            "running": True, "folder": folder, "total": len(pending), "done": 0,
            "updated": 0, "not_found": 0, "failed": 0, "current": "",
            "last_error": "",
        }
        asyncio.get_running_loop().create_task(self._fetch_all_task(folder, pending))
        return json_response({"message": f"开始批量抓取 {len(pending)} 个模型", "started": True})

    async def _fetch_all_task(self, folder: str, pending: list[dict]) -> None:
        import asyncio

        api = self._civitai_red()
        for m in pending:
            self._fetch_progress["current"] = m["name"]
            try:
                r = await self.plugin._models.fetch_civitai(m["rel_path"], api)
                self._fetch_progress[r.get("status", "failed")] = \
                    self._fetch_progress.get(r.get("status", "failed"), 0) + 1
            except CivitaiError as e:
                self._fetch_progress["failed"] += 1
                self._fetch_progress["last_error"] = str(e)
                if "429" in str(e):  # 限速：多等一会儿
                    await asyncio.sleep(8)
            except Exception as e:
                self._fetch_progress["failed"] += 1
                self._fetch_progress["last_error"] = str(e)
            self._fetch_progress["done"] += 1
            await asyncio.sleep(0.4)  # 温和限速
        self._fetch_progress["running"] = False
        self._fetch_progress["current"] = ""

    async def handle_fetch_progress(self) -> Any:
        return json_response(self._fetch_progress)

    async def handle_civitai_search(self) -> Any:
        query = request.query.get("query", "")
        mtype = request.query.get("type", "")
        cursor = request.query.get("cursor", "")
        page = self._int_arg("page", 1, 1, 100)
        try:
            data = await self._civitai_site().search_models(
                query=query, model_type=mtype, limit=20, page=page, cursor=cursor,
            )
        except CivitaiError as e:
            return error_response(str(e))
        # nextPage 是完整 URL，抽出 cursor 供前端翻页
        next_page = (data.get("metadata") or {}).get("nextPage") or ""
        next_cursor = ""
        if next_page:
            from urllib.parse import parse_qs, urlparse

            qs = parse_qs(urlparse(next_page).query)
            next_cursor = (qs.get("cursor") or [""])[0]
        items = []
        for it in data.get("items", []):
            latest = {}
            for v in (it.get("modelVersions") or [])[:1]:
                latest = v
            items.append({
                "model_id": it.get("id"),
                "name": it.get("name"),
                "type": it.get("type"),
                "nsfw": it.get("nsfw"),
                "download_count": (it.get("stats") or {}).get("downloadCount"),
                "thumbs_up": (it.get("stats") or {}).get("thumbsUpCount"),
                "creator": ((it.get("creator") or {}).get("username")) or "",
                "tags": (it.get("tags") or [])[:6],
                "version_id": latest.get("id"),
                "version_name": latest.get("name"),
                "base_model": latest.get("baseModel"),
                "preview_url": ((latest.get("images") or [{}])[0]).get("url") or "",
            })
        return json_response({"items": items, "next_cursor": next_cursor,
                              "metadata": data.get("metadata") or {}})

    async def handle_civitai_version(self) -> Any:
        vid = self._int_arg("version_id", 0, 1, 10**9)
        try:
            data = await self._civitai_red().get_version(vid)
        except CivitaiError as e:
            return error_response(str(e))
        files = [
            {"id": f.get("id"), "name": f.get("name"), "type": f.get("type"),
             "size_kb": f.get("sizeKB"), "primary": f.get("primary", False),
             "download_url": f.get("downloadUrl")}
            for f in (data.get("files") or [])
        ]
        return json_response({
            "version_id": data.get("id"), "model_id": data.get("modelId"),
            "version_name": data.get("name"), "base_model": data.get("baseModel"),
            "model_type": (data.get("model") or {}).get("type"),
            "model_name": (data.get("model") or {}).get("name"),
            "files": files,
            "download_url": ((data.get("files") or [{}])[0]).get("downloadUrl") or "",
        })

    async def handle_civitai_model(self) -> Any:
        """模型详情 + 版本列表（含 in_library 库内检测）。"""
        model_id = self._int_arg("model_id", 0, 1, 10**9)
        try:
            data = await self._civitai_red().get_model(model_id)
        except CivitaiError as e:
            return error_response(str(e))
        local_ids = self.plugin._models.local_version_ids()
        versions = []
        for v in (data.get("modelVersions") or []):
            files = v.get("files") or []
            primary = next((f for f in files if f.get("primary")), files[0] if files else {})
            size_kb = primary.get("sizeKB") or 0
            versions.append({
                "version_id": v.get("id"),
                "name": v.get("name"),
                "base_model": v.get("baseModel"),
                "created_at": v.get("createdAt"),
                "size_kb": size_kb,
                "filename": primary.get("name"),
                "files_count": len(files),
                "in_library": v.get("id") in local_ids,
            })
        return json_response({
            "model_id": data.get("id"),
            "name": data.get("name"),
            "type": data.get("type"),
            "creator": ((data.get("creator") or {}).get("username")) or "",
            "versions": versions,
        })

    async def handle_civitai_image(self) -> Any:
        """图片代理：后端代取 CivitAI 图片（走配置代理）→ data URI。

        用于浏览器直连 image.civitai.com 不通时的前端兜底。
        """
        url = request.query.get("path", "")
        if not url.startswith("https://") or "/civitai" not in url:
            return error_response("仅允许代理 CivitAI 图片 URL", status_code=400)
        try:
            data = await self._civitai_site().download_bytes(url, timeout=25)
        except CivitaiError as e:
            return error_response(f"图片代理失败: {e}")
        if len(data) > 8 * 1024 * 1024:
            return error_response("图片过大")
        import base64

        uri = f"data:image/jpeg;base64,{base64.b64encode(data).decode()}"
        return json_response({"data_uri": uri})

    # ── 下载 ────────────────────────────────────────────────

    async def handle_download_start(self) -> Any:
        body = await request.json() or {}
        version_id = body.get("version_id")
        file_id = body.get("file_id")
        folder = body.get("folder")
        sub = body.get("sub")
        if not isinstance(version_id, int) or version_id <= 0:
            return error_response("version_id 参数无效")
        if folder is not None and not isinstance(folder, str):
            return error_response("folder 参数无效")
        try:
            task = await self.plugin._downloads.start(
                version_id, file_id,
                folder=folder or None,
                sub=sub if isinstance(sub, str) else None,
            )
        except (ValueError, CivitaiError) as e:
            return error_response(str(e))
        return json_response({
            "message": f"已加入下载队列：{task['filename']} → models/{task['folder']}/",
            "task": task,
        })

    async def handle_download_list(self) -> Any:
        return json_response({"items": self.plugin._downloads.list_tasks()})

    async def handle_download_cancel(self) -> Any:
        body = await request.json() or {}
        task_id = body.get("task_id")
        if not isinstance(task_id, int) or task_id <= 0:
            return error_response("task_id 参数无效")
        ok = self.plugin._downloads.cancel(task_id)
        return json_response({
            "message": "已请求取消，正在停止…" if ok else "任务不存在或已结束"
        })

    async def handle_download_delete(self) -> Any:
        body = await request.json() or {}
        task_id = body.get("task_id")
        if not isinstance(task_id, int) or task_id <= 0:
            return error_response("task_id 参数无效")
        ok = self.plugin._downloads.delete_task(task_id)
        if not ok:
            return error_response("任务不存在或仍在进行中（请先取消）")
        return json_response({"message": f"下载记录 #{task_id} 已删除（模型文件保留）"})

    # ── 配置 ────────────────────────────────────────────────

    # (组id, 组标题, [(key, 标签, kind, options)])  kind: text/int/bool/select
    _CONFIG_GROUPS = [
        ("comfyui", "ComfyUI 配置", [
            ("mcp_url", "MCP 服务地址", "text", []),
            ("comfyui_base_url", "ComfyUI 基础地址", "text", []),
            ("comfyui_models_root", "模型库根目录", "text", []),
            ("poll_interval", "轮询间隔（秒）", "int", []),
            ("task_timeout", "任务超时（秒）", "int", []),
        ]),
        ("civitai", "CivitAI 配置", [
            ("civitai_api_key", "API Key", "text", []),
            ("civitai_base_url", "站点", "select",
             ["https://civitai.red/api/v1/", "https://civitai.com/api/v1/"]),
            ("civitai_proxy", "HTTP 代理（留空直连）", "text", []),
        ]),
        ("r2", "R2 图床配置", [
            ("r2_enabled", "启用 R2 上传", "bool", []),
            ("r2_account_id", "Account ID", "text", []),
            ("r2_access_key_id", "Access Key ID", "text", []),
            ("r2_secret_access_key", "Secret Access Key", "text", []),
            ("r2_bucket", "Bucket", "text", []),
            ("r2_public_base_url", "公网基础 URL", "text", []),
            ("r2_key_prefix", "对象 Key 前缀", "text", []),
            ("r2_timeout_seconds", "上传超时（秒）", "int", []),
        ]),
    ]

    async def handle_get_config(self) -> Any:
        cfg = self.plugin.config
        groups = []
        for gid, title, fields in self._CONFIG_GROUPS:
            fl = []
            for key, label, kind, options in fields:
                raw = cfg.get(key)
                if kind == "int":
                    val: Any = int(raw or 0)
                elif kind == "bool":
                    val = bool(raw)
                else:
                    val = "" if raw is None else str(raw)
                fl.append({"key": key, "label": label, "kind": kind,
                           "options": options, "value": val})
            groups.append({"id": gid, "title": title, "fields": fl})
        return json_response({"groups": groups})

    async def handle_save_config(self) -> Any:
        body = await request.json() or {}
        updates = body.get("updates")
        if not isinstance(updates, dict) or not updates:
            return error_response("updates 参数无效")
        allowed = {k: kind for _, _, fs in self._CONFIG_GROUPS
                   for k, _, kind, _ in fs}
        cfg = self.plugin.config
        applied = []
        for k, v in updates.items():
            if k not in allowed:
                continue
            kind = allowed[k]
            if kind == "int":
                try:
                    v = int(v)
                except (TypeError, ValueError):
                    return error_response(f"{k} 需要数字")
            if kind == "bool":
                v = v in (True, 1, "1", "true", "True", "on")
            cfg[k] = v
            applied.append(k)
        cfg.save_config()
        # 模型库根目录变更 → 立即重建 ModelManager（无需重载插件）
        if "comfyui_models_root" in applied:
            self.plugin.rebuild_model_manager()
        return json_response({
            "message": f"已保存 {len(applied)} 项配置（MCP 地址等运行时参数需重载插件后生效）"
        })

    # ── 工具 ────────────────────────────────────────────────

    @staticmethod
    def _int_arg(name: str, default: int, lo: int, hi: int) -> int:
        raw = request.query.get(name, "")
        try:
            v = int(raw)
        except (TypeError, ValueError):
            return default
        return max(lo, min(hi, v))

    def _task_public(self, t: dict) -> dict:
        """任务行 → 前端友好结构。"""
        image_path = t.get("image_path") or ""
        return {
            "id": t["id"],
            "status": t["status"],
            "template": t["template"],
            "prompt": t["prompt"],
            "image_url": t.get("image_url") or "",
            "local_url": t.get("local_url") or "",
            "has_file": bool(image_path and Path(image_path).exists()),
            "error": t.get("error") or "",
            "created_at": t.get("created_at") or "",
            "completed_at": t.get("completed_at") or "",
        }

    # ── 路由 ────────────────────────────────────────────────

    async def handle_tasks(self) -> Any:
        page = self._int_arg("page", 1, 1, 100000)
        size = self._int_arg("size", PAGE_SIZE_DEFAULT, 1, PAGE_SIZE_MAX)
        status = request.query.get("status", "")
        rows = self.mgr.list_tasks(limit=size, offset=(page - 1) * size, status=status or None)
        total = self.mgr.count_tasks(status=status or None)
        # 注意：桥接会对含 "data" 键的响应做 data 解包（PluginPagePage.vue:
        # response.data?.data ?? response.data），并列字段会被吞掉——
        # 列表字段必须叫别的名字（items）
        return json_response({
            "items": [self._task_public(t) for t in rows],
            "page": page,
            "size": size,
            "total": total,
        })

    async def handle_task(self) -> Any:
        try:
            task_id = self._int_arg("task_id", 0, 1, 10**9)
        except Exception:
            return error_response("task_id 参数无效")
        t = self.mgr.get_task(task_id)
        if not t:
            return error_response(f"任务 #{task_id} 不存在", status_code=404)
        # 元数据仅提取模型；提示词直接用任务记录的输入值（不解析工作流）
        meta = {"model": None}
        image_path = t.get("image_path") or ""
        if image_path and Path(image_path).exists():
            parsed = parse_png_file(image_path)
            meta = {"model": parsed.get("model")}
        data = self._task_public(t)
        data["image_path"] = image_path
        data["metadata"] = meta
        return json_response(data)

    async def handle_image(self) -> Any:
        """任务图片。mode=thumb 返回缩略图 data URI（走桥接）；mode=full 返回原图 data URI。

        页面运行在无 allow-same-origin 的沙箱 iframe 中，<img> 直连后端接口带不上
        鉴权，因此图片统一经桥接拉取，以 data URI 形式注入 <img>。
        """
        task_id = self._int_arg("task_id", 0, 1, 10**9)
        mode = request.query.get("mode", "thumb")
        t = self.mgr.get_task(task_id)
        if not t:
            return error_response(f"任务 #{task_id} 不存在", status_code=404)
        path = t.get("image_path") or ""
        if not path or not Path(path).exists():
            return error_response("该任务没有本地图片文件", status_code=404)

        try:
            if mode == "full":
                data = Path(path).read_bytes()
                uri = f"data:image/png;base64,{base64.b64encode(data).decode()}"
            else:
                uri = self._make_thumb(task_id, path)
            return json_response({"data_uri": uri})
        except Exception as e:
            logger.error(f"[{PLUGIN_NAME}] 读取图片失败 task#{task_id}: {e}")
            return error_response(f"读取图片失败: {e}")

    _thumb_cache: dict[int, tuple[str, str]] = {}

    def _make_thumb(self, task_id: int, path: str) -> str:
        """生成（或取缓存的）JPEG 缩略图 data URI，缓存键含文件 mtime。"""
        mtime = str(Path(path).stat().st_mtime)
        cached = self._thumb_cache.get(task_id)
        if cached and cached[0] == mtime:
            return cached[1]
        from PIL import Image  # 延迟导入，避免无 Pillow 环境下插件加载失败

        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((THUMB_SIZE, THUMB_SIZE))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=THUMB_JPEG_QUALITY)
        uri = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
        self._thumb_cache[task_id] = (mtime, uri)
        return uri

    async def handle_delete(self) -> Any:
        body = await request.json() or {}
        task_id = body.get("task_id")
        if not isinstance(task_id, int) or task_id <= 0:
            return error_response("task_id 参数无效")
        t = self.mgr.get_task(task_id)
        if not t:
            return error_response(f"任务 #{task_id} 不存在", status_code=404)
        if t["status"] in ("pending", "running"):
            return error_response("任务执行中，不能删除")
        self.mgr.delete_task(task_id)
        return json_response({"message": f"任务 #{task_id} 已删除"})

    async def handle_resend(self) -> Any:
        body = await request.json() or {}
        task_id = body.get("task_id")
        if not isinstance(task_id, int) or task_id <= 0:
            return error_response("task_id 参数无效")
        t = self.mgr.get_task(task_id)
        if not t:
            return error_response(f"任务 #{task_id} 不存在", status_code=404)
        if t["status"] != "completed":
            return error_response("仅已完成任务可重发")
        try:
            await self.plugin._on_task_complete(t)
            return json_response({"message": f"任务 #{task_id} 已重新推送"})
        except Exception as e:
            return error_response(f"推送失败: {e}")
