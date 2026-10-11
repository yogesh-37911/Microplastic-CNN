"""Private Supabase Storage helpers with local-disk fallback."""

from __future__ import annotations

from urllib.parse import quote

import httpx

from app.config import get_settings


def _storage_config() -> tuple[str, str, str]:
    settings = get_settings()
    if not settings.supabase_url or not settings.supabase_secret_key:
        raise RuntimeError("Supabase Storage is not configured.")
    return (
        settings.supabase_url.rstrip("/"),
        settings.supabase_secret_key,
        settings.supabase_storage_bucket,
    )


def is_supabase_path(path: str) -> bool:
    return path.startswith("supabase://")


def _object_url(path: str) -> tuple[str, dict[str, str]]:
    base, key, bucket = _storage_config()
    object_name = path.removeprefix(f"supabase://{bucket}/")
    return (
        f"{base}/storage/v1/object/{quote(bucket, safe='')}/{quote(object_name, safe='/')} ".strip(),
        {"Authorization": f"Bearer {key}", "apikey": key},
    )


async def upload_image(object_name: str, content: bytes, content_type: str) -> str:
    settings = get_settings()
    path = f"supabase://{settings.supabase_storage_bucket}/{object_name}"
    url, headers = _object_url(path)
    headers["Content-Type"] = content_type
    headers["x-upsert"] = "false"
    async with httpx.AsyncClient(timeout=45) as client:
        response = await client.post(url, headers=headers, content=content)
    response.raise_for_status()
    return path


async def download_image(path: str) -> bytes:
    url, headers = _object_url(path)
    async with httpx.AsyncClient(timeout=45) as client:
        response = await client.get(url, headers=headers)
    response.raise_for_status()
    return response.content


async def delete_image(path: str) -> None:
    url, headers = _object_url(path)
    async with httpx.AsyncClient(timeout=45) as client:
        response = await client.delete(url, headers=headers)
    if response.status_code != 404:
        response.raise_for_status()
