"""AstrBot 插件 · ComfyUI MCP工作台 主逻辑。

- LLM 工具 comfyui_draw：提交异步画图任务（串行队列），立即返回任务号
- 完成推送：插件用 context.send_message 官方主动消息通道，把生成的图片
  连同简短任务信息直接推回提交时的会话（unified_msg_origin 存于任务表 source_session）
- LLM 工具 comfyui_send_image：按任务号补发图片（用户索要重发时用）
"""

from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime
from pathlib import Path

import astrbot.api.message_components as Comp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.star import Context, Star, StarTools, register

from . import r2_upload
from .civitai_api import RED_BASE, CivitaiApi, CivitaiError
from .download_manager import DownloadManager
from .mcp_client import ComfyuiMcpClient, ComfyuiMcpError
from .model_manager import ModelManager
from .task_manager import TaskManager
from .web_api import WebApiHandler

PLUGIN_NAME = "astrbot_plugin_comfyui_async_workbench"
DEFAULT_TEMPLATE = "1、Anima文生图"
TEMPLATE_CACHE_TTL = 300.0

# 模板输入名中代表"输入图片"的键（模板10 实测为 image）
IMAGE_INPUT_KEYS = ("image", "图片", "输入图片", "输入图像")

ASPECT_RATIOS = (
    "1:1 (Square)", "2:3 (Portrait Photo)", "3:2 (Photo)", "3:4 (Portrait Standard)",
    "4:3 (Standard)", "9:16 (Portrait Widescreen)", "16:9 (Widescreen)", "21:9 (Ultrawide)",
)

# 常见参数名别名组：LLM 传中文名或常见英文名都能归一到模板实际键（最终以模板 inputs 为准）
_PARAM_GROUPS: list[tuple[str, list[str]]] = [
    ("提示词", ["提示词", "prompt", "positive_prompt", "正面提示词"]),
    ("负面提示词", ["负面提示词", "negative_prompt", "negative"]),
    ("宽高比例", ["宽高比例", "宽高比", "aspect_ratio", "ratio"]),
    ("输入图片", ["image", "图片", "输入图片", "输入图像"]),
    ("画师", ["画师", "artist"]),
    ("采样步数", ["采样步数", "steps"]),
    ("lora预设", ["lora预设", "lora_preset", "lora"]),
    ("cfg", ["cfg", "cfg_scale"]),
    ("加速强度", ["加速强度", "strength"]),
]


def _format_inputs(inputs: dict) -> str:
    """把模板 inputs 格式化成给 LLM 看的参数速查串。"""
    return "、".join(
        f"{k}({(v or {}).get('type', '?')}，默认 {(v or {}).get('default', '')!r})"
        for k, v in inputs.items()
    )


@register(
    PLUGIN_NAME,
    "千代鳶",
    "对接本地 ComfyUI-APP-MCP：LLM 异步画图任务、R2 图床上传、图片管理工作台、本地模型管理与 CivitAI 下载",
    "0.2.0",
)
class ComfyuiWorkbenchPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._client: ComfyuiMcpClient | None = None
        self._mgr: TaskManager | None = None
        self._data_dir: Path | None = None
        self._tpl_cache_ts = 0.0
        self._tpl_names: list[str] = []
        self._tpl_schemas: dict[str, dict] = {}

    # ── 生命周期 ────────────────────────────────────────────

    async def initialize(self):
        self._data_dir = StarTools.get_data_dir(PLUGIN_NAME)
        self._client = ComfyuiMcpClient(
            url=self.config.get("mcp_url", "http://127.0.0.1:8188/app-mcp"),
            sse_read_timeout=600.0,
        )
        self._mgr = TaskManager(
            db_path=self._data_dir / "tasks.db",
            images_dir=self._data_dir / "images",
            client=self._client,
            comfyui_base_url=self.config.get("comfyui_base_url", "http://127.0.0.1:8188"),
            poll_interval=int(self.config.get("poll_interval", 3)),
            task_timeout=float(self.config.get("task_timeout", 900)),
            on_complete=self._on_task_complete,
            r2_uploader=self._r2_uploader,
        )
        await self._mgr.start()
        # 本地模型管理（三文件夹，兼容 Lora-Manager 元数据约定）
        self._models = ModelManager(
            self.config.get("comfyui_models_root", r"F:\ComfyUI-aki-v3\ComfyUI\models")
        )
        # CivitAI 模型下载器（串行队列）
        self._downloads = DownloadManager(
            models_root=self.config.get("comfyui_models_root", r"F:\ComfyUI-aki-v3\ComfyUI\models"),
            db_path=self._data_dir / "downloads.db",
            civitai_factory=self._civitai_red_api,
            model_manager=self._models,
        )
        # Web API / 前端面板
        self.web_api = WebApiHandler(self)
        self.web_api.register_routes()
        logger.info(f"{PLUGIN_NAME}: 已初始化, data_dir={self._data_dir}")

    def _civitai_red_api(self) -> CivitaiApi:
        """下载/哈希匹配用：红站全量库 + Key + 代理。"""
        return CivitaiApi(
            self.config.get("civitai_api_key", ""),
            RED_BASE,
            self.config.get("civitai_proxy", ""),
        )

    def rebuild_model_manager(self) -> None:
        """模型库根目录配置变更后热重建（扫描缓存自动清空）。"""
        self._models = ModelManager(
            self.config.get("comfyui_models_root", r"F:\ComfyUI-aki-v3\ComfyUI\models")
        )
        self._downloads.mm = self._models
        logger.info(
            f"{PLUGIN_NAME}: ModelManager 已重建, models_root={self.config.get('comfyui_models_root')}"
        )

    async def _r2_uploader(self, png_bytes: bytes, origin_name: str) -> str:
        """R2 上传钩子：凭证读插件配置；r2_enabled=False 返回空串（跳过上传）。"""
        if not self.config.get("r2_enabled", True):
            return ""
        prefix = str(self.config.get("r2_key_prefix", "comfyui/")).strip().strip("/")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe = re.sub(r"[^\w.\-]+", "_", origin_name) or "image.png"
        key = f"{prefix}/{stamp}_{safe}" if prefix else f"{stamp}_{safe}"
        # upload_png_bytes 为阻塞 urllib，放线程池避免卡事件循环
        return await asyncio.to_thread(
            r2_upload.upload_png_bytes,
            png_bytes,
            account_id=self.config.get("r2_account_id", ""),
            access_key_id=self.config.get("r2_access_key_id", ""),
            secret_access_key=self.config.get("r2_secret_access_key", ""),
            bucket=self.config.get("r2_bucket", ""),
            public_base_url=self.config.get("r2_public_base_url", ""),
            key_prefix=prefix + "/",
            timeout_seconds=int(self.config.get("r2_timeout_seconds", 60)),
            object_key=key,
        )

    async def terminate(self):
        if self._mgr:
            await self._mgr.stop()
            self._mgr.close()
            self._mgr = None
        logger.info(f"{PLUGIN_NAME}: 已卸载")

    # ── 模板信息（动态缓存） ────────────────────────────────

    async def _refresh_templates(self, force: bool = False) -> None:
        if not force and time.time() - self._tpl_cache_ts < TEMPLATE_CACHE_TTL and self._tpl_names:
            return
        templates = await self._client.list_templates()
        self._tpl_names = [t["name"] for t in templates if t.get("name")]
        self._tpl_cache_ts = time.time()

    async def _get_template_inputs(self, name: str) -> dict:
        """返回模板 inputs（dict），失败返回空 dict。"""
        if name in self._tpl_schemas:
            return self._tpl_schemas[name].get("inputs", {})
        try:
            detail = await self._client.get_template(name)
        except ComfyuiMcpError as e:
            logger.warning(f"{PLUGIN_NAME}: get_template({name}) 失败: {e}")
            return {}
        self._tpl_schemas[name] = detail
        return detail.get("inputs", {})

    # ── LLM 工具：画图 ──────────────────────────────────────

    @filter.llm_tool(name="comfyui_draw")
    async def draw_image(
        self,
        event: AstrMessageEvent,
        prompt: str,
        template: str = DEFAULT_TEMPLATE,
        aspect_ratio: str = "",
        extra_params: dict = {},
    ):
        '''画图工具（异步）。向 ComfyUI 提交画图任务，立即返回任务号，不阻塞对话；可连续提交多个任务按顺序执行，生成完成后图片会自动发送到当前会话。

        Args:
            prompt(string): 画面描述（提示词）
            template(string): ComfyUI 模板名，默认 "1、Anima文生图"；模板清单与各模板参数可通过 MCP 工具 list_templates / get_template 查询
            aspect_ratio(string): 宽高比例，仅文生图模板有效："1:1 (Square)"、"3:4 (Portrait Standard)"、"4:3 (Standard)"、"9:16 (Portrait Widescreen)"、"16:9 (Widescreen)"、"21:9 (Ultrawide)" 等，留空用模板默认
            extra_params(object): 模板的其他参数（可选），键名按模板实际参数填写，中英文别名均可自动识别，如 {"画师":"@sy4","lora预设":1,"采样步数":16}
        '''
        # 1) 校验模板名（动态）
        await self._refresh_templates()
        if template not in self._tpl_names:
            await self._refresh_templates(force=True)
        if template not in self._tpl_names:
            return (
                f"模板不存在：{template}。当前可用模板：{'、'.join(self._tpl_names)}。"
                "请向用户确认后用正确模板名重试。"
            )

        # 2) 按模板 inputs 组装参数：extra_params 键名归一化，未匹配则返回参数速查表让 LLM 自纠
        inputs = await self._get_template_inputs(template)
        if not inputs:
            return f"无法获取模板「{template}」的参数定义（ComfyUI MCP 异常），请稍后重试。"
        if not prompt.strip():
            return "画面描述（prompt）不能为空。"

        params: dict = {}
        unknown: list[str] = []
        for k, v in (extra_params or {}).items():
            key = self._resolve_param_key(k, inputs)
            if key is None:
                unknown.append(k)
            else:
                params[key] = v
        if unknown:
            return (
                f"参数名无法匹配模板输入：{'、'.join(unknown)}。\n"
                f"模板「{template}」可用参数：{_format_inputs(inputs)}。\n"
                "建议先用 MCP 工具 get_template 查询该模板的参数表后再投递。"
            )

        # 显式参数（prompt/aspect_ratio）优先于 extra_params
        if "提示词" in inputs:
            params["提示词"] = prompt
        elif "prompt" in inputs:
            params["prompt"] = prompt
        if aspect_ratio and "宽高比例" in inputs:
            params["宽高比例"] = aspect_ratio

        # 3) 图像编辑类模板：从当前消息链取用户发的图片并上传 ComfyUI
        image_key = next((k for k in IMAGE_INPUT_KEYS if k in inputs), None)
        if image_key:
            try:
                uploaded = await self._upload_message_image(event)
            except ValueError as e:  # 用户没发图
                return str(e)
            except ComfyuiMcpError as e:
                return f"图片上传 ComfyUI 失败：{e}"
            params[image_key] = uploaded

        # 4) 入队（source_session 存 unified_msg_origin，完成后主动推送用）
        task_id = await self._mgr.submit(
            prompt=prompt,
            template=template,
            params=params,
            source_session=event.unified_msg_origin,
            source_user=event.get_sender_name() or "",
        )

        waiting = self._mgr.pending_count()
        suffix = "，正在生成…" if waiting <= 1 else f"，前面还有 {waiting - 1} 个任务排队中…"
        param_summary = "，".join(f"{k}={str(v)[:40]}" for k, v in params.items())
        return f"已提交画图任务 #{task_id}（模板：{template}；参数：{param_summary}）{suffix}"

    @staticmethod
    def _resolve_param_key(key: str, inputs: dict) -> str | None:
        """把 LLM 传的参数名归一到模板实际键名；未匹配返回 None。"""
        if key in inputs:
            return key
        low = key.lower()
        for k in inputs:
            if k.lower() == low:
                return k
        for _, members in _PARAM_GROUPS:
            member_lows = [m.lower() for m in members]
            if low in member_lows:
                for k in inputs:
                    if k.lower() in member_lows:
                        return k
        return None

    async def _upload_message_image(self, event: AstrMessageEvent) -> str:
        """取消息链中最后一张图片，经 ComfyUI upload_image 上传，返回文件名。

        图片来源可能是 http(s) URL、data URL 或本地路径（QQ 附件常见本地临时路径），
        统一交给 MCP upload_image 处理（服务端与本机同机，三种 source 均支持）。
        """
        images = [m for m in event.get_messages() if isinstance(m, Comp.Image)]
        if not images:
            raise ValueError(
                "该模板需要一张输入图片：请让用户先发送一张图片（和编辑要求在同一条消息里），然后再调用画图。"
            )
        source = self._image_source(images[-1])
        if not source:
            raise ValueError(
                "消息中的图片无法读取（既不是可用 URL 也不是本地文件），请让用户重新发送图片。"
            )
        try:
            result = await self._client.upload_image(source)
        except ComfyuiMcpError as e:
            raise ValueError(f"图片上传失败，请让用户重发原图后再试。（{str(e)[:80]}）")
        name = result.get("name") or result.get("filename") or ""
        if not name:
            raise ValueError("图片上传后未返回文件名，请让用户重发图片后重试。")
        return name

    @staticmethod
    def _image_source(img: Comp.Image) -> str | None:
        """从图片组件提取可上传的 source：http(s)/data URL 或存在的本地路径。"""
        for cand in (getattr(img, "url", None), getattr(img, "file", None)):
            if not cand:
                continue
            cand = str(cand).strip()
            if cand.startswith("file://"):
                cand = cand[7:]
            if cand.startswith(("http://", "https://", "data:")):
                return cand
            if Path(cand).exists():
                return str(Path(cand))
        return None

    # ── LLM 工具：直发图片文件 ──────────────────────────────

    @filter.llm_tool(name="comfyui_send_image")
    async def send_image(self, event: AstrMessageEvent, task_id: int):
        '''把画图任务生成的图片文件作为图片消息直接发送。任务需处于已完成状态。

        Args:
            task_id(int): 画图任务编号（任务通知里的 #号）
        '''
        task = self._mgr.get_task(task_id)
        if not task:
            return f"任务 #{task_id} 不存在。"
        if task["status"] != "completed":
            return f"任务 #{task_id} 状态为 {task['status']}，还没有可发送的图片。"
        path = task.get("image_path") or ""
        if not path or not Path(path).exists():
            url = task.get("image_url") or task.get("local_url") or ""
            if not url:
                return f"任务 #{task_id} 的图片文件已不存在，请改发图床/本地URL。"
            await event.send(MessageChain().url_image(url))
        else:
            await event.send(MessageChain().file_image(path))
        return f"任务 #{task_id} 的图片已发送。"

    @filter.llm_tool(name="comfyui_task_status")
    async def task_status(self, event: AstrMessageEvent, task_id: int = 0):
        '''查询画图任务队列状态。传 task_id 查单个任务详情；不传查队列概览（执行中/排队数/最近任务）。

        Args:
            task_id(int): 任务编号，可选；不传或传 0 查队列概览
        '''
        if task_id and task_id > 0:
            t = self._mgr.get_task(task_id)
            if not t:
                return f"任务 #{task_id} 不存在。"
            lines = [
                f"任务 #{task_id}：{t['status']}（模板：{t['template']}）",
                f"提示词：{(t['prompt'] or '')[:80]}",
            ]
            if t.get("error"):
                lines.append(f"错误：{t['error'][:150]}")
            if t.get("image_url"):
                lines.append(f"图床URL：{t['image_url']}")
            if t.get("completed_at"):
                lines.append(f"完成时间：{t['completed_at']}")
            return "\n".join(lines)

        ov = self._mgr.queue_overview()
        running = ov["running"]
        if running:
            head = f"执行中：#{running['id']}（{running['template']}），排队等待 {ov['pending']} 个"
        else:
            head = f"当前无执行中任务，排队等待 {ov['pending']} 个"
        lines = [head, "最近任务："]
        status_cn = {"completed": "已完成", "failed": "失败", "running": "执行中", "pending": "排队中"}
        for t in self._mgr.list_tasks(limit=5):
            err = f"（{(t['error'] or '')[:40]}）" if t["status"] == "failed" and t.get("error") else ""
            lines.append(f"#{t['id']} {status_cn.get(t['status'], t['status'])} {t['template']}{err}")
        return "\n".join(lines)

    # ── 完成推送：官方主动消息通道（context.send_message） ──

    async def _on_task_complete(self, task: dict) -> None:
        """任务完成后把图片（或失败原因）直接推回提交时的会话。"""
        task_id = task["id"]
        session = task.get("source_session") or ""
        if not session:
            logger.warning(f"{PLUGIN_NAME}: task#{task_id} 无会话信息(umo)，跳过推送")
            return

        try:
            if task["status"] == "completed":
                caption = (
                    f"画图任务 #{task_id} 完成\n"
                    f"模板：{task['template']}\n"
                    f"提示词：{task['prompt']}"
                )
                chain = MessageChain().message(caption)
                path = task.get("image_path") or ""
                if path and Path(path).exists():
                    chain.file_image(path)
                elif task.get("local_url"):
                    chain.url_image(task["local_url"])
                else:
                    chain.message("\n（图片文件缺失，可稍后用任务号补发）")
            else:
                chain = MessageChain().message(
                    f"画图任务 #{task_id} 失败\n"
                    f"模板：{task['template']}\n"
                    f"提示词：{task['prompt']}\n"
                    f"错误：{task.get('error') or '未知'}"
                )
            ok = await self.context.send_message(session, chain)
            if not ok:
                logger.warning(f"{PLUGIN_NAME}: task#{task_id} 主动推送返回 False (session={session})")
        except Exception as e:
            logger.error(f"{PLUGIN_NAME}: task#{task_id} 完成推送失败: {e}")
