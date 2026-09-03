"""PNG 元数据解析：读取 ComfyUI 生成图嵌入的 extra_pnginfo（tEXt/iTXt chunk）。

ComfyUI 约定把 API 格式节点图写进 tEXt chunk 的 "prompt" 键（嵌套 JSON）：
    {"<node_id>": {"class_type": "...", "inputs": {...}}, ...}
从 KSampler 的 positive/negative 引用解出正/负提示词所属节点，再取其 text。

必保三项：模型、正面提示词、负面提示词；尽力提取：采样器、seed、steps、cfg。
非 ComfyUI 图（截图等）无相关 chunk，返回空字段、绝不抛错。
"""

from __future__ import annotations

import json
import re
import struct
import zlib
import ast
from pathlib import Path

PNG_SIG = b"\x89PNG\r\n\x1a\n"

_CHECKPOINT_NODES = (
    "CheckpointLoaderSimple", "CheckpointLoader", "UNETLoader",
    "unCLIPCheckpointLoader", "VAELoader",
)
_SAMPLER_NODES = ("KSampler", "KSamplerAdvanced")
_TEXT_ENCODE_NODES = (
    "CLIPTextEncode", "CLIPTextEncodeSDXL", "smZCLIPTextEncode",
    "T5TextEncode", "CLIPT5TextEncode",
)


def read_png_text_chunks(data: bytes) -> dict[str, str]:
    """读取 PNG 的 tEXt/iTXt 文本 chunk（纯标准库，不加载整图）。"""
    texts: dict[str, str] = {}
    if not data.startswith(PNG_SIG):
        return texts
    pos = 8
    size = len(data)
    while pos + 8 <= size:
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        ctype = data[pos + 4:pos + 8]
        chunk = data[pos + 8:pos + 8 + length]
        if ctype == b"tEXt":
            key, _, val = chunk.partition(b"\x00")
            texts[key.decode("latin-1", "replace")] = val.decode("latin-1", "replace")
        elif ctype == b"iTXt":
            key, _, rest = chunk.partition(b"\x00")
            if not rest:
                continue
            comp_flag = rest[0]
            rest = rest[1:]
            _, _, rest = rest.partition(b"\x00")  # language tag
            _, sep, val = rest.partition(b"\x00")  # translated keyword
            if not sep:
                continue
            try:
                if comp_flag == 1:
                    val = zlib.decompress(val)
                texts[key.decode("utf-8", "replace")] = val.decode("utf-8", "replace")
            except Exception:
                continue
        if ctype == b"IEND":
            break
        pos += 12 + length
    return texts


def _ref_node_id(ref) -> str | None:
    """KSampler 的 positive/negative 引用形如 ["6", 0]，取节点 id。"""
    if isinstance(ref, (list, tuple)) and ref and isinstance(ref[0], (str, int)):
        return str(ref[0])
    return None


_REF_STR_RE = re.compile(r"^\[\s*['\"]?\w+['\"]?\s*,\s*\d+\s*\]$")


def _parse_ref_str(val: str):
    """把字符串形式的引用 "['387', 0]" 解析回 (node_id, slot)。"""
    try:
        parsed = ast.literal_eval(val)
    except Exception:
        return None
    if isinstance(parsed, (list, tuple)) and parsed and isinstance(parsed[0], (str, int)):
        return str(parsed[0]), int(parsed[1]) if len(parsed) > 1 else 0
    return None


# 已知直接承载常量字符串的字段名（按优先级）
_CONST_TEXT_KEYS = ("text", "value", "string", "string_1", "content", "prompt")


def _resolve_node_text(nodes: dict, node_id: str | None, depth: int = 0) -> str:
    """通用递归解析一个节点"承载的文本"。

    覆盖实际遇到的链路：
      CLIPTextEncode.text(字符串或引用串) / JoinStringMulti(string_1..N 引用拼接) /
      PrimitiveStringMultiline.value / ReferenceLatent.conditioning(透传 conditioning)
    规则：常量字符串字段优先返回；否则收集全部引用型输入递归解析后拼接。
    """
    if node_id is None or depth > 6:
        return ""
    node = nodes.get(node_id)
    if not isinstance(node, dict):
        return ""
    # ConditioningZeroOut：Flux 系工作流把负面条件零化，语义上等于空提示词
    if node.get("class_type") == "ConditioningZeroOut":
        return ""
    inputs = node.get("inputs") or {}

    for key in _CONST_TEXT_KEYS:
        v = inputs.get(key)
        if isinstance(v, str) and not _REF_STR_RE.match(v):
            return v

    parts: list[str] = []
    delimiter = inputs.get("delimiter") if isinstance(inputs.get("delimiter"), str) else ", "
    for v in inputs.values():
        if isinstance(v, str) and _REF_STR_RE.match(v):
            ref = _parse_ref_str(v)
            if ref:
                parts.append(_resolve_node_text(nodes, ref[0], depth + 1))
        elif isinstance(v, (list, tuple)) and v and isinstance(v[0], (str, int)) and len(v) == 2:
            parts.append(_resolve_node_text(nodes, str(v[0]), depth + 1))
    return delimiter.join(p for p in parts if p)


def parse_prompt_graph(prompt: dict) -> dict:
    """从 API 格式节点图提取关键字段。"""
    out = {
        "model": None, "models": [],
        "positive_prompt": None, "negative_prompt": None,
        "sampler": None, "seed": None, "steps": None, "cfg": None,
    }
    if not isinstance(prompt, dict):
        return out
    nodes = {str(k): v for k, v in prompt.items() if isinstance(v, dict)}

    for node in nodes.values():
        ct = node.get("class_type", "")
        inputs = node.get("inputs") or {}
        if ct in _CHECKPOINT_NODES:
            name = inputs.get("ckpt_name") or inputs.get("unet_name") or inputs.get("vae_name")
            if name:
                out["models"].append(f"{ct}:{name}")
        elif ct in _SAMPLER_NODES:
            out["sampler"] = inputs.get("sampler_name", out["sampler"])
            out["seed"] = inputs.get("seed", inputs.get("noise_seed", out["seed"]))
            out["steps"] = inputs.get("steps", out["steps"])
            out["cfg"] = inputs.get("cfg", out["cfg"])
            pos = _resolve_node_text(nodes, _ref_node_id(inputs.get("positive")))
            neg = _resolve_node_text(nodes, _ref_node_id(inputs.get("negative")))
            if isinstance(inputs.get("positive"), str):
                pos = str(inputs.get("positive"))  # 直接内联字符串的情况
            if isinstance(inputs.get("negative"), str):
                neg = str(inputs.get("negative"))
            out["positive_prompt"] = pos or out["positive_prompt"]
            out["negative_prompt"] = neg or out["negative_prompt"]

    if out["models"]:
        out["model"] = out["models"][0].split(":", 1)[-1]
    return out


def parse_png_metadata(data: bytes) -> dict:
    """解析 PNG 字节中的 ComfyUI 元数据；无元数据返回空字段不抛错。"""
    result = {
        "has_metadata": False, "raw_keys": [],
        "model": None, "models": [],
        "positive_prompt": None, "negative_prompt": None,
        "sampler": None, "seed": None, "steps": None, "cfg": None,
    }
    try:
        texts = read_png_text_chunks(data)
    except Exception:
        return result
    result["raw_keys"] = list(texts.keys())
    prompt_raw = texts.get("prompt")
    if not prompt_raw:
        return result
    try:
        prompt = json.loads(prompt_raw)
    except Exception:
        return result
    parsed = parse_prompt_graph(prompt)
    result.update(parsed)
    result["has_metadata"] = bool(
        parsed["model"] or parsed["positive_prompt"] or parsed["negative_prompt"]
    )
    return result


def parse_png_file(path: str | Path) -> dict:
    """解析 PNG 文件（读字节后调 parse_png_metadata；文件不存在/读失败返回空结果）。"""
    try:
        return parse_png_metadata(Path(path).read_bytes())
    except Exception:
        return {
            "has_metadata": False, "raw_keys": [],
            "model": None, "models": [],
            "positive_prompt": None, "negative_prompt": None,
            "sampler": None, "seed": None, "steps": None, "cfg": None,
        }
