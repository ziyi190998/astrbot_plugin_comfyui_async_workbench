# ComfyUI MCP工作台 (astrbot_plugin_comfyui_async_workbench)

AstrBot 插件：对接本地 ComfyUI-APP-MCP，提供 **LLM 异步画图任务、R2 图床上传、生成档案面板、本地模型管理与 CivitAI 下载** 一站式工作台。

> 版本 v0.2.0 · 适配 AstrBot >= v4.25.3 · ComfyUI 需安装 [ComfyUI-APP-MCP](https://github.com/Lotus0614/ComfyUI-APP-MCP)（`http://127.0.0.1:8188/app-mcp`）

## 功能总览

### 1. LLM 画图工具（对话内使用）

| 工具 | 用途 |
|---|---|
| `comfyui_draw` | 画图任务投递口：接收参数 → 串行队列异步执行 → 立即返回任务号，可连投多个任务 |
| `comfyui_task_status` | 查询队列状态（执行中/排队数/最近任务）或单个任务详情 |
| `comfyui_send_image` | 按任务号补发图片文件（用户索要重发时用） |

- **完成后主动推送**：通过官方主动消息通道 `context.send_message` 把图片直接推回发起会话（私聊回私聊、群聊回群），不依赖 LLM 唤醒、不注入提示词
- **参数归一化兜底**：`prompt→提示词`、`artist→画师`、`steps→采样步数` 等中英文别名自动映射；传错参数名返回该模板的完整参数表供 LLM 自纠
- **图生图**：图像编辑模板（如 `10、Flux.2klein图像编辑`）自动取消息链中的用户图片上传（本地路径/URL/base64 均可）
- **R2 图床上传**：生成完成后自动上传 Cloudflare R2，公网 URL 存入任务记录；失败降级仅保留本地文件
- 建议工作流：LLM 先用 MCP 工具 `list_templates` / `get_template` 查参数表，再投递 `comfyui_draw`

### 2. 生成档案（WebUI 面板）

- 任务列表（任务号/模板/提示词/生成时间/缩略图/删除）、状态筛选（全部/完成/失败/排队）、执行中置顶
- 灯箱详情：模型（从 PNG 元数据提取）、输入提示词、图床URL、本地文件、完成时间
- 复制图床URL / 重新推送 / 删除（本地副本+记录，R2 画廊对象保留）
- 分页：每页 10/20/30/40 条 + 图标换页栏

### 3. 模型管理（WebUI 面板）

- 扫描 `checkpoints / diffusion_models / loras` 三文件夹（递归子目录），**完全兼容 ComfyUI-Lora-Manager 的 `.metadata.json` 与预览文件约定**，双方互操作
- 目录树侧栏（可折叠、子目录计数筛选）、卡片网格（预览图懒加载、CivitAI 徽标）
- **CivitAI 抓取**：按需计算全文件 SHA256 → `by-hash` 匹配 → 元数据合并写入 + 预览图下载（nsfwLevel<4 选图、450px 优化）；单个/批量（带进度）；未收录模型标记后不再重复查询；已有元数据重新抓取需确认（force 覆盖含预览）
- **视频预览**：卡片显示首帧（ffmpeg 抽帧，依赖 `imageio-ffmpeg`），详情灯箱内直接播放
- 删除：模型+元数据+预览三件套，文件被占用时明确报错

### 4. CivitAI 下载（WebUI 面板）

- **三步向导**：① 粘贴模型 URL → ② 选版本（**已在库中的版本灰显锁定**，有任一版本在库时金色提示）→ ③ 落盘路径（默认规则：基模 → `diffusion_models/{基础模型}`、LoRA 系 → `loras/{基础模型}`；sdxl 等需进 checkpoints 的关闭默认后手选）
- **关键词搜索**：红站全量/蓝站 SFW（按配置站点），卡片「选用」载入向导
- **下载队列**：串行下载、`.part` 断点续传、磁盘空间预检、信号灯状态（灰=完成/绿=下载中/蓝=队列/红=失败）、进度条+速度、取消、删除记录（保留模型文件）、分页

## 安装

1. 复制本插件目录到 `AstrBot/data/plugins/astrbot_plugin_comfyui_async_workbench/`
2. 重载插件（首次会自动安装 `requirements.txt` 依赖：`imageio-ffmpeg` 等）
3. WebUI → 插件配置：填写 CivitAI API Key（下载/抓取必需）；R2 图床凭证按需填写；模型库根目录改为你的 ComfyUI models 路径（见下）

## 配置项（WebUI 插件配置页 / `_conf_schema.json`）

| 分组 | 键 | 默认 | 说明 |
|---|---|---|---|
| ComfyUI | `mcp_url` | `http://127.0.0.1:8188/app-mcp` | ComfyUI-APP-MCP 端点 |
| | `comfyui_base_url` | `http://127.0.0.1:8188` | 图片下载基址 |
| | `comfyui_models_root` | `F:\ComfyUI-aki-v3\ComfyUI\models` | 模型库根目录（保存后热生效） |
| | `poll_interval` / `task_timeout` | 3 / 900 | 轮询间隔与任务超时（秒） |
| CivitAI | `civitai_api_key` | 空 | Bearer 认证（在 civitai.com 个人设置生成后填写） |
| | `civitai_base_url` | `https://civitai.red/api/v1/` | **红站=官方 NSFW 站（内容全量，API/哈希匹配默认走此）**；蓝站 `civitai.com` 仅 SFW（搜索可选） |
| | `civitai_proxy` | 空 | HTTP 代理（预览图 `image.civitai.com` 与图片代理需要） |
| R2 图床 | `r2_*` 七项 | （内置） | 凭证/桶/公网URL/前缀/超时；`r2_enabled=false` 跳过上传 |

## 数据落点

- 任务库/图片副本：`data/plugin_data/astrbot_plugin_comfyui_async_workbench/`（tasks.db、images/）
- 下载任务库：同目录 `downloads.db`
- 模型元数据/预览：与模型同目录（`<名>.metadata.json`、`<名>.<ext>`），与 Lora-Manager 共享

## 常见问题

- **预览图抓取失败**：`image.civitai.com` 需代理，在 CivitAI 配置填 `civitai_proxy`（如 `http://127.0.0.1:7890`）
- **下载的模型去哪了**：按向导第三步的落盘路径；默认 `models/diffusion_models/{基础模型}/` 或 `models/loras/{基础模型}/`
- **删除模型报"被占用"**：ComfyUI 正在使用该模型，换模型或关闭 ComfyUI 后再删
- **任务完成后没收到推送**：查 `backend.log` 中 `主动推送` 相关日志；平台适配器掉线时会记录 `返回 False`
- **重载插件后下载中断**：队列中"已中断"任务是正常现象，重新发起同版本下载会从 `.part` 断点续传

## 架构

```
main.py            入口：LLM 工具 / 主动推送 / 模块装配
mcp_client.py      ComfyUI-APP-MCP 客户端（streamable_http）
task_manager.py    画图任务：SQLite + asyncio.Queue 串行 worker + R2 钩子
r2_upload.py       Cloudflare R2 上传（纯标准库 SigV4）
png_metadata.py    PNG 元数据（tEXt/节点图递归 → 模型）
model_manager.py   本地模型扫描/元数据/预览/删除（Lora-Manager 兼容）
civitai_api.py     CivitAI REST（by-hash/搜索/版本/图片）
download_manager.py 模型下载（续传/进度/队列持久化）
web_api.py         WebUI 路由（20 条）
pages/workbench/   前端单页（哥特暗色）
```
