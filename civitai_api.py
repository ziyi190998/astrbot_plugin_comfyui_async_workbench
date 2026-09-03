"""CivitAI REST 客户端。

站点语义：红站 civitai.red = 官方 NSFW 站（内容全量，哈希匹配/版本查询默认走此站）；
蓝站 civitai.com = 官方 SFW 站（搜索时可选，结果为 SFW 过滤视图）。
认证：Authorization: Bearer <civitai_api_key>。
"""

from __future__ import annotations

import json
from typing import Any

import httpx

RED_BASE = "https://civitai.red/api/v1"
BLUE_BASE = "https://civitai.com/api/v1"


class CivitaiError(RuntimeError):
    """CivitAI API 调用失败（网络/认证/限速/服务端错误）。"""


class CivitaiApi:
    def __init__(self, api_key: str, base_url: str = RED_BASE, proxy: str = "") -> None:
        self.api_key = (api_key or "").strip()
        self.base_url = (base_url or RED_BASE).rstrip("/")
        self.proxy = (proxy or "").strip() or None

    def _client(self, timeout: float = 30.0) -> httpx.AsyncClient:
        headers = {"User-Agent": "astrbot-comfyui-workbench/0.1"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            proxy=self.proxy,
            timeout=timeout,
            follow_redirects=True,
        )

    async def _get(self, path: str, params: dict | None = None, timeout: float = 30.0) -> httpx.Response:
        async with self._client(timeout) as c:
            resp = await c.get(path, params=params)
        return resp

    @staticmethod
    def _raise_for_status(resp: httpx.Response, what: str) -> None:
        if resp.status_code == 401:
            raise CivitaiError("CivitAI API Key 无效（401），请检查插件配置")
        if resp.status_code == 429:
            retry = resp.headers.get("Retry-After", "")
            raise CivitaiError(f"CivitAI 限速（429），请稍后重试{f'，建议等待 {retry}s' if retry else ''}")
        if resp.status_code >= 500:
            raise CivitaiError(f"CivitAI 服务端错误（{resp.status_code}）: {what}")
        if resp.status_code != 200:
            raise CivitaiError(f"CivitAI 请求失败（{resp.status_code}）: {what}")

    # ── 业务接口 ────────────────────────────────────────────

    async def get_version_by_hash(self, sha256: str) -> dict | None:
        """按全文件 SHA256 查询版本；未收录返回 None，其余错误抛 CivitaiError。"""
        resp = await self._get(f"/model-versions/by-hash/{sha256}")
        if resp.status_code == 404:
            return None
        self._raise_for_status(resp, "by-hash")
        try:
            return resp.json()
        except json.JSONDecodeError as e:
            raise CivitaiError(f"by-hash 返回非 JSON: {e}") from e

    async def get_version(self, version_id: int) -> dict:
        resp = await self._get(f"/model-versions/{version_id}")
        if resp.status_code == 404:
            raise CivitaiError(f"版本 {version_id} 不存在")
        self._raise_for_status(resp, "version-detail")
        return resp.json()

    async def get_model(self, model_id: int) -> dict:
        """模型详情（含 modelVersions 版本列表）。"""
        resp = await self._get(f"/models/{model_id}")
        if resp.status_code == 404:
            raise CivitaiError(f"模型 {model_id} 不存在")
        self._raise_for_status(resp, "model-detail")
        return resp.json()

    async def search_models(
        self, query: str = "", model_type: str = "", limit: int = 20, page: int = 1,
        cursor: str = "", nsfw: str = "",
    ) -> dict:
        """搜索模型（GET /models）。站点决定内容范围：红站全量、蓝站 SFW。

        注意 CivitAI 规则：带 query 时必须用 cursor 游标分页（page 会 400）；
        无 query（按类型浏览）可用 page。
        """
        params: dict[str, Any] = {"limit": max(1, min(limit, 100))}
        if query:
            params["query"] = query
        if model_type:
            params["types"] = model_type
        if nsfw:
            params["nsfw"] = nsfw
        if cursor:
            params["cursor"] = cursor
        elif not query:
            params["page"] = max(1, page)
        resp = await self._get("/models", params=params)
        self._raise_for_status(resp, "search")
        return resp.json()

    async def download_bytes(self, url: str, timeout: float = 60.0) -> bytes:
        """下载小文件（预览图）。大模型下载走 M3 的专用下载器。"""
        async with self._client(timeout) as c:
            resp = await c.get(url)
        if resp.status_code != 200:
            raise CivitaiError(f"下载失败（{resp.status_code}）: {url[:80]}")
        return resp.content
