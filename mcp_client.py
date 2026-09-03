"""ComfyUI-APP-MCP 客户端封装（streamable_http，stateless）。

服务端为 FastMCP stateless_http=True，每次调用独立处理；本封装采用
"每次调用独立建连 → initialize → tools/call" 的模式，天然规避长连接
断线问题，轮询场景下开销可接受（本机回环）。

错误约定：服务端工具内部异常时返回 {"error": "..."} JSON（而非 MCP isError），
本封装统一转成 ComfyuiMcpError 抛出。
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

DEFAULT_MCP_URL = "http://127.0.0.1:8188/app-mcp"


class ComfyuiMcpError(RuntimeError):
    """ComfyUI MCP 调用失败（传输失败 / 服务端报错 / 返回异常）。"""


class ComfyuiMcpClient:
    def __init__(
        self,
        url: str = DEFAULT_MCP_URL,
        timeout: float = 30.0,
        sse_read_timeout: float = 600.0,
    ) -> None:
        self.url = url
        self.timeout = timeout
        self.sse_read_timeout = sse_read_timeout

    async def call_tool(self, tool: str, args: dict[str, Any] | None = None) -> Any:
        """调用 MCP 工具并解析 JSON 结果；出错抛 ComfyuiMcpError。"""
        args = args or {}
        try:
            async with streamablehttp_client(
                self.url,
                timeout=timedelta(seconds=self.timeout),
                sse_read_timeout=timedelta(seconds=self.sse_read_timeout),
            ) as (read_stream, write_stream, _):
                async with ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(seconds=self.sse_read_timeout),
                ) as session:
                    await session.initialize()
                    result = await session.call_tool(tool, args)
        except ComfyuiMcpError:
            raise
        except Exception as e:  # 传输层/协议层异常
            raise ComfyuiMcpError(f"MCP 调用 {tool} 传输失败: {e}") from e

        if result.isError:
            raise ComfyuiMcpError(f"MCP 调用 {tool} 失败: {result.content}")

        try:
            text = "".join(
                getattr(block, "text", "") for block in result.content
            )
            data = json.loads(text)
        except Exception as e:
            raise ComfyuiMcpError(f"MCP 调用 {tool} 返回非 JSON 内容: {e}") from e

        if isinstance(data, dict) and "error" in data:
            raise ComfyuiMcpError(f"MCP 调用 {tool} 报错: {data['error']}")
        return data

    # ── 业务封装 ────────────────────────────────────────────

    async def list_templates(self) -> list[dict]:
        """列出全部模板。"""
        data = await self.call_tool("list_templates")
        return data.get("templates", []) if isinstance(data, dict) else []

    async def get_template(self, name: str) -> dict:
        """获取模板详情（inputs/outputs/docs）。"""
        return await self.call_tool("get_template", {"name": name})

    async def run_template(self, name: str, params: dict, wait: bool = False) -> dict:
        """执行模板。wait=False 立即返回 {run_id, status}。"""
        return await self.call_tool(
            "run_template",
            {"name": name, "params": json.dumps(params, ensure_ascii=False), "wait": wait},
        )

    async def get_template_result(
        self, name: str, run_id: str, wait: bool = False, timeout: float | None = None
    ) -> dict:
        """按 run_id 获取执行结果（注意：服务端要求同时传模板名 name）。"""
        args: dict[str, Any] = {"name": name, "run_id": run_id, "wait": wait}
        if timeout is not None:
            args["timeout"] = timeout
        return await self.call_tool("get_template_result", args)

    async def upload_image(self, source: str) -> dict:
        """上传新图片到 ComfyUI。source 支持本地路径 / HTTP URL / data:base64 URL。"""
        return await self.call_tool("upload_image", {"source": source})

    async def list_models(self, folder: str = "", keywords: str = "") -> dict:
        """列出模型目录或某目录下的模型。"""
        return await self.call_tool(
            "list_models", {"folder": folder, "keywords": keywords}
        )
