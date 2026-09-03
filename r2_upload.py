"""Cloudflare R2 上传（S3 API + AWS SigV4，纯标准库）。

移植自 ComfyUI-ImgBed-Link 的 r2_upload.py，去掉 config.json 加载逻辑：
凭证由调用方（AstrBot 插件配置）显式传入。

注意：内部使用阻塞的 urllib，请在异步上下文中用 asyncio.to_thread 包裹调用。
"""

from __future__ import annotations

import hashlib
import hmac
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

_REGION = "auto"
_SERVICE = "s3"


def make_object_key(key_prefix: str = "comfyui/", now: datetime | None = None) -> str:
    """生成 ``comfyui/20260812-153045.png`` 形式的对象 key。"""
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    prefix = str(key_prefix or "").strip().strip("/")
    return f"{prefix}/{stamp}.png" if prefix else f"{stamp}.png"


def upload_png_bytes(
    png_bytes: bytes,
    *,
    account_id: str,
    access_key_id: str,
    secret_access_key: str,
    bucket: str,
    public_base_url: str,
    key_prefix: str = "comfyui/",
    timeout_seconds: int = 60,
    object_key: str | None = None,
) -> str:
    """PUT PNG 字节到 R2 桶并返回公网 URL；配置缺失/HTTP 错误抛 ValueError。"""
    account_id = str(account_id or "").strip()
    access_key_id = str(access_key_id or "").strip()
    secret_access_key = str(secret_access_key or "").strip()
    bucket = str(bucket or "").strip()
    public_base_url = str(public_base_url or "").strip().rstrip("/")
    if not account_id:
        raise ValueError("R2 account_id 为空：请在插件配置中填写")
    if not access_key_id or not secret_access_key:
        raise ValueError("R2 access_key_id / secret_access_key 为空：请在插件配置中填写")
    if not bucket:
        raise ValueError("R2 bucket 为空：请在插件配置中填写")
    if not public_base_url:
        raise ValueError("R2 public_base_url 为空：请在插件配置中填写")

    key = str(object_key or "").strip().lstrip("/") or make_object_key(key_prefix)
    host = f"{account_id}.r2.cloudflarestorage.com"
    path = "/" + "/".join(
        urllib.parse.quote(part, safe="") for part in [bucket, *key.split("/")] if part
    )
    url = f"https://{host}{path}"

    payload_hash = hashlib.sha256(png_bytes).hexdigest()
    now = datetime.now(timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")

    canonical_headers = (
        f"host:{host}\n"
        f"x-amz-content-sha256:{payload_hash}\n"
        f"x-amz-date:{amz_date}\n"
    )
    signed_headers = "host;x-amz-content-sha256;x-amz-date"
    canonical_request = "\n".join(
        ["PUT", path, "", canonical_headers, signed_headers, payload_hash]
    )
    scope = f"{date_stamp}/{_REGION}/{_SERVICE}/aws4_request"
    string_to_sign = "\n".join(
        [
            "AWS4-HMAC-SHA256",
            amz_date,
            scope,
            hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
        ]
    )
    signing_key = _derive_signing_key(secret_access_key, date_stamp)
    signature = hmac.new(
        signing_key, string_to_sign.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    authorization = (
        "AWS4-HMAC-SHA256 "
        f"Credential={access_key_id}/{scope}, "
        f"SignedHeaders={signed_headers}, "
        f"Signature={signature}"
    )

    request = urllib.request.Request(
        url,
        data=png_bytes,
        method="PUT",
        headers={
            "Authorization": authorization,
            "x-amz-date": amz_date,
            "x-amz-content-sha256": payload_hash,
            "Content-Type": "image/png",
            "User-Agent": "astrbot_plugin_comfyui_async_workbench/0.1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=max(5, int(timeout_seconds or 60))) as resp:
            status = getattr(resp, "status", 200)
            if status >= 400:
                detail = resp.read().decode("utf-8", errors="replace")
                raise ValueError(f"R2 HTTP {status}: {_clip(detail)}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ValueError(f"R2 HTTP {exc.code}: {_clip(detail)}") from exc

    return f"{public_base_url}/{key}"


def _derive_signing_key(secret_access_key: str, date_stamp: str) -> bytes:
    k_date = _hmac_sha256(("AWS4" + secret_access_key).encode("utf-8"), date_stamp)
    k_region = _hmac_sha256(k_date, _REGION)
    k_service = _hmac_sha256(k_region, _SERVICE)
    return _hmac_sha256(k_service, "aws4_request")


def _hmac_sha256(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _clip(text: str, limit: int = 240) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "…"
