"""本地模型管理：扫描 checkpoints / diffusion_models / loras，兼容 ComfyUI-Lora-Manager 约定。

约定（与 ComfyUI-Lora-Manager v1.2.0 一致，双方互操作）：
- 元数据：<模型名>.metadata.json（同目录同名），civitai 键存完整版本 JSON
- 预览图：<模型名>.<扩展名>（webp/jpeg/png/mp4 等，同名任意扩展）
- 哈希：全文件 SHA256，按需计算后写回 metadata.json
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .civitai_api import CivitaiApi

logger = logging.getLogger(__name__)

FOLDERS = ("checkpoints", "diffusion_models", "loras")
MODEL_EXTS = {".safetensors", ".ckpt", ".pt", ".pt2", ".pth", ".bin", ".sft", ".gguf"}
# 预览扩展名发现优先级（参考 Lora-Manager file_utils.find_preview_file）
PREVIEW_EXTS = (
    ".webp", ".preview.webp", ".preview.png", ".preview.jpeg", ".preview.jpg",
    ".png", ".jpeg", ".jpg", ".gif", ".mp4", ".webm", ".avif", ".jxl",
)
VIDEO_EXTS = {".mp4", ".webm", ".gif"}
SCAN_CACHE_TTL = 30.0


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


class ModelManager:
    def __init__(self, models_root: str | Path) -> None:
        self.root = Path(models_root)
        # folder -> (扫描时间, 模型列表)
        self._scan_cache: dict[str, tuple[float, list[dict]]] = {}
        # 预览路径 -> (mtime, data_uri)
        self._thumb_cache: dict[str, tuple[str, str]] = {}

    # ── 路径安全 ────────────────────────────────────────────

    def _resolve(self, rel_path: str) -> Path | None:
        """相对路径 → 根目录内绝对路径；越界返回 None。"""
        try:
            p = (self.root / rel_path).resolve()
            p.relative_to(self.root.resolve())
        except (ValueError, OSError):
            return None
        return p

    # ── 扫描 ────────────────────────────────────────────────

    def list_models(self, folder: str, refresh: bool = False) -> list[dict]:
        if folder not in FOLDERS:
            raise ValueError(f"未知模型文件夹: {folder}（可选 {FOLDERS}）")
        cached = self._scan_cache.get(folder)
        if not refresh and cached and time.time() - cached[0] < SCAN_CACHE_TTL:
            return cached[1]

        models: list[dict] = []
        base = self.root / folder
        if base.is_dir():
            for dirpath, _dirnames, filenames in os.walk(base):
                dir_p = Path(dirpath)
                for fn in filenames:
                    p = dir_p / fn
                    if p.suffix.lower() not in MODEL_EXTS:
                        continue
                    models.append(self._model_entry(p, folder))
        models.sort(key=lambda m: m["name"].lower())
        self._scan_cache[folder] = (time.time(), models)
        return models

    def _model_entry(self, path: Path, folder: str) -> dict:
        rel = path.relative_to(self.root).as_posix()
        meta = self._read_metadata(path)
        civ = meta.get("civitai") or {}
        model_info = civ.get("model") or {}
        preview = self._find_preview(path)
        try:
            stat = path.stat()
            size, mtime = stat.st_size, stat.st_mtime
        except OSError:
            size, mtime = 0, 0.0
        return {
            "rel_path": rel,
            "name": path.stem,
            "file_name": path.name,
            "folder": folder,
            "sub_dir": path.parent.relative_to(self.root / folder).as_posix(),
            "size": size,
            "mtime": mtime,
            "has_metadata": bool(meta),
            "from_civitai": bool(meta.get("from_civitai")),
            "civitai_deleted": bool(meta.get("civitai_deleted")),
            "sha256_known": bool(meta.get("sha256")),
            "version_id": civ.get("id"),
            "model_name": meta.get("model_name") or model_info.get("name") or "",
            "model_type": model_info.get("type") or civ.get("baseModel") or "",
            "base_model": meta.get("base_model") or civ.get("baseModel") or "",
            "version_name": civ.get("name") or "",
            "preview_rel": (preview.relative_to(self.root).as_posix()) if preview else "",
            "preview_kind": self._preview_kind(preview),
            "preview_nsfw_level": meta.get("preview_nsfw_level"),
        }

    def local_version_ids(self) -> set[int]:
        """本地库中已存在的 CivitAI 版本 id 集合（三个文件夹全扫，含缓存）。"""
        ids: set[int] = set()
        for folder in FOLDERS:
            for m in self.list_models(folder):
                vid = m.get("version_id")
                if isinstance(vid, int):
                    ids.add(vid)
        return ids

    def _read_metadata(self, model_path: Path) -> dict:
        mp = model_path.with_suffix(".metadata.json")
        try:
            if mp.is_file():
                data = json.loads(mp.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
        except (json.JSONDecodeError, OSError):
            pass
        return {}

    def _find_preview(self, model_path: Path) -> Path | None:
        stem = model_path.stem
        try:
            names = {p.name.lower(): p for p in model_path.parent.iterdir() if p.is_file()}
        except OSError:
            return None
        for ext in PREVIEW_EXTS:
            p = names.get((stem + ext).lower())
            if p:
                return p
        return None

    @staticmethod
    def _preview_kind(preview: Path | None) -> str:
        if not preview:
            return "none"
        return "video" if preview.suffix.lower() in VIDEO_EXTS else "image"

    # ── 详情 / 删除 ──────────────────────────────────────────

    def get_detail(self, rel_path: str) -> dict | None:
        p = self._resolve(rel_path)
        if not p or not p.is_file() or p.suffix.lower() not in MODEL_EXTS:
            return None
        entry = self._model_entry(p, p.parent.relative_to(self.root).parts[0])
        meta = self._read_metadata(p)
        civ = meta.get("civitai") or {}
        entry["civitai_summary"] = {
            "version_id": civ.get("id"),
            "model_id": civ.get("modelId"),
            "version_name": civ.get("name"),
            "base_model": civ.get("baseModel"),
            "download_count": (civ.get("stats") or {}).get("downloadCount"),
            "thumbs_up": (civ.get("stats") or {}).get("thumbsUpCount"),
            "trained_words": civ.get("trainedWords") or [],
            "nsfw_level": civ.get("nsfwLevel"),
            "description": (civ.get("description") or "")[:1200],
            "model_page": (f"https://civitai.red/models/{civ.get('modelId')}"
                           f"?modelVersionId={civ.get('id')}") if civ.get("modelId") else "",
        }
        entry["sha256"] = meta.get("sha256") or ""
        entry["notes"] = meta.get("notes") or ""
        return entry

    def delete(self, rel_path: str) -> list[str]:
        """删除模型文件 + 同名元数据 + 同名预览；返回被删除的相对路径列表。

        模型本体删除失败（如被 ComfyUI 占用）时抛错并放弃整个删除，
        避免出现"记录删了、文件还在"的半删状态。
        """
        p = self._resolve(rel_path)
        if not p or not p.is_file() or p.suffix.lower() not in MODEL_EXTS:
            raise ValueError("非法或不存在的模型路径")
        try:
            p.unlink()
        except OSError as e:
            raise ValueError(
                f"模型文件删除失败（可能被 ComfyUI 或资源管理器占用）：{e}"
            ) from e
        deleted: list[str] = [p.relative_to(self.root).as_posix()]
        for t in (p.with_suffix(".metadata.json"), self._find_preview(p)):
            if t and t.is_file():
                try:
                    t.unlink()
                    deleted.append(t.relative_to(self.root).as_posix())
                except OSError as e:
                    logger.warning(f"附属文件删除失败（忽略） {t}: {e}")
        self._scan_cache.pop(p.parent.relative_to(self.root).parts[0], None)
        return deleted

    # ── 预览缩略图（桥接 data URI） ──────────────────────────

    def preview_data_uri(self, rel_path: str, max_width: int = 400) -> str:
        """预览缩略图 data URI（mtime 缓存）。视频取首帧（ffmpeg），失败返回空串。"""
        p = self._resolve(rel_path)
        if not p or not p.is_file() or self._preview_kind(p) == "none":
            return ""
        key = str(p)
        try:
            mtime = str(p.stat().st_mtime)
        except OSError:
            return ""
        cached = self._thumb_cache.get(key)
        if cached and cached[0] == mtime:
            return cached[1]
        if self._preview_kind(p) == "video":
            uri = self._video_first_frame(p, max_width)
        else:
            uri = self._image_thumb(p, max_width)
        if uri:
            self._thumb_cache[key] = (mtime, uri)
        return uri

    @staticmethod
    def _image_thumb(p: Path, max_width: int) -> str:
        try:
            import base64
            import io

            from PIL import Image

            with Image.open(p) as im:
                im = im.convert("RGB")
                im.thumbnail((max_width, max_width))
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=78)
            return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
        except Exception as e:
            logger.warning(f"预览图读取失败 {p}: {e}")
            return ""

    @staticmethod
    def _video_first_frame(p: Path, max_width: int) -> str:
        """ffmpeg 抽视频首帧（imageio-ffmpeg 自带静态二进制）。"""
        import base64
        import subprocess
        import tempfile

        try:
            import imageio_ffmpeg

            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as e:
            logger.warning(f"视频抽帧不可用（未安装 imageio-ffmpeg）: {e}")
            return ""
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "frame.jpg"
            cmd = [
                exe, "-hide_banner", "-loglevel", "error",
                "-ss", "0", "-i", str(p),
                "-frames:v", "1", "-vf", f"scale={max_width}:-2",
                "-q:v", "4", "-y", str(out),
            ]
            try:
                subprocess.run(cmd, capture_output=True, timeout=30, check=True)
                data = out.read_bytes()
            except Exception as e:
                logger.warning(f"视频首帧抽取失败 {p.name}: {e}")
                return ""
            if not data:
                return ""
            return f"data:image/jpeg;base64,{base64.b64encode(data).decode()}"

    def video_data_uri(self, rel_path: str, max_mb: int = 40) -> str:
        """视频预览整文件 base64（灯箱播放用）；超限返回空串。"""
        p = self._resolve(rel_path)
        if not p or not p.is_file() or self._preview_kind(p) != "video":
            return ""
        try:
            if p.stat().st_size > max_mb * 1024 * 1024:
                return ""
            import base64

            data = p.read_bytes()
        except OSError:
            return ""
        mime = "video/mp4" if p.suffix.lower() == ".mp4" else "video/webm"
        return f"data:{mime};base64,{base64.b64encode(data).decode()}"

    # ── 哈希（M2 抓取时按需调用） ────────────────────────────

    @staticmethod
    def calculate_sha256(file_path: Path, chunk_mb: int = 4) -> str:
        """全文件 SHA256（与 CivitAI by-hash 及 Lora-Manager 一致）。"""
        h = hashlib.sha256()
        chunk = chunk_mb * 1024 * 1024
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(chunk), b""):
                h.update(block)
        return h.hexdigest()

    def save_metadata(self, model_rel: str, updates: dict[str, Any]) -> dict:
        """合并写入 metadata.json（原子写），返回写入后的完整数据。"""
        p = self._resolve(model_rel)
        if not p or not p.is_file():
            raise ValueError("模型路径不存在")
        mp = p.with_suffix(".metadata.json")
        data = self._read_metadata(p)
        data.update(updates)
        data.setdefault("file_name", p.stem)
        data.setdefault("file_path", str(p))
        _atomic_write_json(mp, data)
        return data

    # ── CivitAI 抓取 ────────────────────────────────────────

    async def fetch_civitai(
        self, rel_path: str, civitai: "CivitaiApi", force: bool = False
    ) -> dict:
        """单个模型：按需算 SHA256 → by-hash → 合并元数据 → 下载预览图。

        返回 {status: updated|not_found|skipped, preview: bool, sha256: str}
        """
        p = self._resolve(rel_path)
        if not p or not p.is_file() or p.suffix.lower() not in MODEL_EXTS:
            raise ValueError("模型路径不存在或非法")
        meta = self._read_metadata(p)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")

        if not force and meta.get("from_civitai"):
            return {"status": "skipped", "reason": "已有 CivitAI 元数据", "preview": bool(self._find_preview(p))}
        if not force and meta.get("civitai_deleted"):
            return {"status": "skipped", "reason": "CivitAI 未收录（此前已确认）", "preview": False}

        # 1) SHA256（大文件放线程池，结果持久化避免重复计算）
        sha = meta.get("sha256") or ""
        if not sha:
            sha = await asyncio.to_thread(self.calculate_sha256, p)
            self.save_metadata(rel_path, {"sha256": sha})

        # 2) by-hash（哈希匹配一律走红站全量库，由调用方保证 base_url）
        version = await civitai.get_version_by_hash(sha)
        if version is None:
            self.save_metadata(rel_path, {
                "from_civitai": False, "civitai_deleted": True, "last_checked_at": now,
            })
            self._scan_cache.pop(p.parent.relative_to(self.root).parts[0], None)
            return {"status": "not_found", "preview": False, "sha256": sha}

        model_info = version.get("model") or {}
        image, nsfw_level = self._select_preview(version)
        updates = {
            "from_civitai": True,
            "civitai_deleted": False,
            "civitai": version,
            "model_name": model_info.get("name") or p.stem,
            "base_model": version.get("baseModel") or meta.get("base_model") or "",
            "preview_nsfw_level": nsfw_level,
            "last_checked_at": now,
        }
        self.save_metadata(rel_path, updates)

        # 3) 预览图（force 时删除旧预览后重新下载，避免残留不同扩展名旧图）
        got_preview = False
        if image and image.get("url"):
            existing = self._find_preview(p)
            if force and existing:
                try:
                    existing.unlink()
                except OSError:
                    pass
                existing = None
            if force or not existing:
                got_preview = await self._download_preview(p, image["url"], civitai)
        self._scan_cache.pop(p.parent.relative_to(self.root).parts[0], None)
        return {"status": "updated", "preview": got_preview, "sha256": sha}

    @staticmethod
    def _select_preview(version: dict) -> tuple[dict | None, int | None]:
        """选预览：优先 图片类型 且 nsfwLevel<4；其次任意类型 nsfwLevel<4；否则最低 nsfwLevel。

        （图片优先：卡片缩略图无需抽帧，视频仅在没有合适图片时选用）
        """
        images = version.get("images") or []
        for img in images:
            if img.get("type") == "image" and img.get("nsfwLevel", 0) < 4:
                return img, img.get("nsfwLevel")
        for img in images:
            if img.get("nsfwLevel", 0) < 4:
                return img, img.get("nsfwLevel")
        if images:
            lowest = min(images, key=lambda i: i.get("nsfwLevel", 0))
            return lowest, lowest.get("nsfwLevel")
        return None, None

    async def _download_preview(self, model_path: Path, url: str, civitai: "CivitaiApi") -> bool:
        """下载预览到 <模型名>.<扩展名>（URL 重写为 450px 优化版，参考 Lora-Manager）。"""
        video = any(x in url.lower() for x in (".mp4", ".webm")) or "/video/" in url
        dl_url = self._rewrite_preview_url(url, video)
        try:
            data = await civitai.download_bytes(dl_url, timeout=60)
        except Exception as e:
            logger.warning(f"预览下载失败 {model_path.name}: {e}")
            return False
        m = re.search(r"\.(jpe?g|png|webp|gif|avif|mp4|webm)(?:[?#]|$)", dl_url, re.I)
        ext = ("." + m.group(1).lower()) if m else (".mp4" if video else ".webp")
        dest = model_path.with_suffix(ext)
        try:
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, dest)
        except OSError as e:
            logger.warning(f"预览写盘失败 {dest}: {e}")
            return False
        return True

    @staticmethod
    def _rewrite_preview_url(url: str, video: bool) -> str:
        if "/original=true" in url:
            repl = "transcode=true,width=450,optimized=true" if video else "width=450,optimized=true"
            return url.replace("/original=true", "/" + repl)
        return url
