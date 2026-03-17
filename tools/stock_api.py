"""
Stock media API clients for Pexels and Pixabay.

Provides video search and download with retry logic, streaming downloads,
and a unified multi-source search helper.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import settings

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Shared HTTP helpers
# ─────────────────────────────────────────────────────────────────────────────

_CHUNK_SIZE = 1024 * 64  # 64 KB chunks for streaming


def _make_session(max_retries: int = 3) -> requests.Session:
    """Create a requests Session with exponential-backoff retry logic."""
    session = requests.Session()
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=1.0,          # 1s, 2s, 4s between retries
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _download_file(url: str, output_path: str, session: requests.Session) -> str:
    """
    Stream-download *url* to *output_path*.

    Logs progress every ~10 MB. Returns the output path on success.
    Raises requests.HTTPError on non-2xx status.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with session.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        log_interval = 10 * 1024 * 1024  # log every 10 MB
        last_logged = 0

        with open(output_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=_CHUNK_SIZE):
                if chunk:
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if downloaded - last_logged >= log_interval:
                        pct = f"{downloaded / total * 100:.1f}%" if total else f"{downloaded // 1024}KB"
                        logger.debug("Downloading %s — %s", Path(output_path).name, pct)
                        last_logged = downloaded

    logger.info("Downloaded %s (%s bytes)", Path(output_path).name, downloaded)
    return output_path


def _ext_from_url(url: str, fallback: str = ".mp4") -> str:
    """Extract file extension from a URL path, or return fallback."""
    path = urlparse(url).path
    _, ext = os.path.splitext(path)
    return ext if ext else fallback


# ─────────────────────────────────────────────────────────────────────────────
# Pexels Client
# ─────────────────────────────────────────────────────────────────────────────

class PexelsClient:
    """HTTP client for the Pexels Videos API."""

    BASE_URL = "https://api.pexels.com/videos"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._session = _make_session()
        self._session.headers.update({"Authorization": api_key})

    # ── Public API ────────────────────────────────────────────────────────────

    def search_videos(
        self,
        query: str,
        per_page: int = 10,
        min_duration: int = 5,
        max_duration: int = 30,
    ) -> list[dict]:
        """
        Search Pexels for stock videos.

        Returns a normalised list of dicts with keys:
            id, url, download_url, duration, width, height,
            thumbnail_url, photographer, source
        """
        params: dict = {
            "query": query,
            "per_page": min(per_page, 80),   # Pexels max is 80
            "min_duration": min_duration,
            "max_duration": max_duration,
        }

        url = f"{self.BASE_URL}/search"
        logger.info("Pexels search: %r (per_page=%d)", query, per_page)

        try:
            resp = self._session.get(url, params=params, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.error("Pexels search failed: %s", exc)
            return []

        data = resp.json()
        videos = data.get("videos", [])
        return [self._normalise(v) for v in videos]

    def download_video(self, video_dict: dict, output_dir: str) -> str:
        """
        Download a video to *output_dir*.

        *video_dict* must have a ``download_url`` key (as returned by
        ``search_videos``). Returns the absolute local file path.
        """
        download_url: str = video_dict["download_url"]
        ext = _ext_from_url(download_url)
        filename = f"pexels_{video_dict['id']}{ext}"
        output_path = str(Path(output_dir) / filename)

        if Path(output_path).exists():
            logger.info("Pexels: already downloaded %s", filename)
            return output_path

        return _download_file(download_url, output_path, self._session)

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _best_download_url(video_files: list[dict]) -> str:
        """Pick the highest-quality HD file available."""
        if not video_files:
            return ""
        # Prefer hd quality, then fall back to anything
        hd_files = [f for f in video_files if f.get("quality") == "hd"]
        candidates = hd_files or video_files
        # Sort by width descending, pick largest
        candidates_sorted = sorted(candidates, key=lambda f: f.get("width", 0), reverse=True)
        return candidates_sorted[0].get("link", "")

    @staticmethod
    def _normalise(raw: dict) -> dict:
        """Flatten a Pexels video object into the standard schema."""
        video_files: list[dict] = raw.get("video_files", [])
        download_url = PexelsClient._best_download_url(video_files)

        # Thumbnail from video_pictures
        pictures = raw.get("video_pictures", [])
        thumbnail_url = pictures[0].get("picture", "") if pictures else ""

        # Dimensions from the best file
        best_file = next(
            (f for f in video_files if f.get("link") == download_url), {}
        )

        return {
            "id": f"pexels_{raw.get('id', uuid.uuid4().hex[:8])}",
            "url": raw.get("url", ""),
            "download_url": download_url,
            "duration": float(raw.get("duration", 0)),
            "width": int(best_file.get("width") or raw.get("width", 0)),
            "height": int(best_file.get("height") or raw.get("height", 0)),
            "thumbnail_url": thumbnail_url,
            "photographer": raw.get("user", {}).get("name", "unknown"),
            "source": "pexels",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Pixabay Client
# ─────────────────────────────────────────────────────────────────────────────

class PixabayClient:
    """HTTP client for the Pixabay Videos API."""

    BASE_URL = "https://pixabay.com/api/videos/"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._session = _make_session()

    # ── Public API ────────────────────────────────────────────────────────────

    def search_videos(
        self,
        query: str,
        per_page: int = 10,
    ) -> list[dict]:
        """
        Search Pixabay for stock videos.

        Returns a normalised list of dicts using the same schema as
        ``PexelsClient.search_videos``.
        """
        params: dict = {
            "key": self._api_key,
            "q": query,
            "per_page": min(per_page, 200),   # Pixabay max is 200
            "video_type": "film",
        }

        logger.info("Pixabay search: %r (per_page=%d)", query, per_page)

        try:
            resp = self._session.get(self.BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.error("Pixabay search failed: %s", exc)
            return []

        data = resp.json()
        hits = data.get("hits", [])
        return [self._normalise(v) for v in hits]

    def download_video(self, video_dict: dict, output_dir: str) -> str:
        """
        Download a video to *output_dir*.

        Returns the absolute local file path.
        """
        download_url: str = video_dict["download_url"]
        ext = _ext_from_url(download_url)
        # Pixabay IDs already contain "pixabay_" prefix from normalise
        filename = f"{video_dict['id']}{ext}"
        output_path = str(Path(output_dir) / filename)

        if Path(output_path).exists():
            logger.info("Pixabay: already downloaded %s", filename)
            return output_path

        return _download_file(download_url, output_path, self._session)

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _best_download_url(videos: dict) -> tuple[str, int, int]:
        """
        Pick the best resolution stream from Pixabay's ``videos`` dict.

        Returns (url, width, height).
        """
        # Pixabay quality keys in descending preference
        for quality in ("large", "medium", "small", "tiny"):
            entry = videos.get(quality)
            if entry and entry.get("url"):
                return entry["url"], int(entry.get("width", 0)), int(entry.get("height", 0))
        return "", 0, 0

    @staticmethod
    def _normalise(raw: dict) -> dict:
        """Flatten a Pixabay video hit into the standard schema."""
        videos_dict = raw.get("videos", {})
        download_url, width, height = PixabayClient._best_download_url(videos_dict)

        # Pixabay doesn't provide a direct thumbnail in video results,
        # fall back to the picture_id pattern or an empty string.
        picture_id = raw.get("picture_id", "")
        thumbnail_url = (
            f"https://i.vimeocdn.com/video/{picture_id}_295x166.jpg" if picture_id else ""
        )

        duration = float(raw.get("duration", 0))

        return {
            "id": f"pixabay_{raw.get('id', uuid.uuid4().hex[:8])}",
            "url": raw.get("pageURL", ""),
            "download_url": download_url,
            "duration": duration,
            "width": width,
            "height": height,
            "thumbnail_url": thumbnail_url,
            "photographer": raw.get("user", "unknown"),
            "source": "pixabay",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Unified multi-source search
# ─────────────────────────────────────────────────────────────────────────────

def search_all_sources(
    query: str,
    output_dir: str,
    max_results: int = 5,
) -> list[dict]:
    """
    Search both Pexels and Pixabay, merge and deduplicate results, then
    download the top *max_results* videos into *output_dir*.

    API keys are read from ``config.settings``. Sources with missing keys are
    silently skipped (a warning is logged).

    Returns a list of AssetInfo-compatible dicts with keys:
        local_path, source_id, duration, width, height,
        thumbnail_url, photographer, source, download_url
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    all_results: list[dict] = []

    # ── Pexels ────────────────────────────────────────────────────────────────
    if settings.pexels_api_key:
        try:
            pexels = PexelsClient(settings.pexels_api_key)
            pexels_results = pexels.search_videos(query, per_page=max_results * 2)
            all_results.extend(pexels_results)
            logger.info("Pexels returned %d results for %r", len(pexels_results), query)
        except Exception as exc:
            logger.warning("Pexels search error: %s", exc)
    else:
        logger.warning("PEXELS_API_KEY not set — skipping Pexels search")

    # ── Pixabay ───────────────────────────────────────────────────────────────
    if settings.pixabay_api_key:
        try:
            pixabay = PixabayClient(settings.pixabay_api_key)
            pixabay_results = pixabay.search_videos(query, per_page=max_results * 2)
            all_results.extend(pixabay_results)
            logger.info("Pixabay returned %d results for %r", len(pixabay_results), query)
        except Exception as exc:
            logger.warning("Pixabay search error: %s", exc)
    else:
        logger.warning("PIXABAY_API_KEY not set — skipping Pixabay search")

    if not all_results:
        logger.warning("No stock video results found for query: %r", query)
        return []

    # ── Deduplicate by download_url ───────────────────────────────────────────
    seen_urls: set[str] = set()
    unique_results: list[dict] = []
    for item in all_results:
        url = item.get("download_url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(item)

    # Sort by duration (prefer clips >= 5s), then by resolution
    unique_results.sort(
        key=lambda v: (v.get("duration", 0) >= 5, v.get("width", 0)),
        reverse=True,
    )

    top_results = unique_results[:max_results]
    logger.info("Downloading %d/%d unique results for %r", len(top_results), len(unique_results), query)

    # ── Download ──────────────────────────────────────────────────────────────
    downloaded: list[dict] = []
    for item in top_results:
        if not item.get("download_url"):
            logger.warning("Skipping result with no download_url: %s", item.get("id"))
            continue

        source_name = item.get("source", "stock")
        try:
            if source_name == "pexels" and settings.pexels_api_key:
                client: PexelsClient | PixabayClient = PexelsClient(settings.pexels_api_key)
            elif source_name == "pixabay" and settings.pixabay_api_key:
                client = PixabayClient(settings.pixabay_api_key)
            else:
                logger.warning("Cannot download from source %r — no client available", source_name)
                continue

            local_path = client.download_video(item, output_dir)
            file_size = Path(local_path).stat().st_size if Path(local_path).exists() else 0

            downloaded.append({
                "local_path": local_path,
                "source_id": item["id"],
                "duration": item.get("duration"),
                "width": item.get("width"),
                "height": item.get("height"),
                "thumbnail_url": item.get("thumbnail_url", ""),
                "photographer": item.get("photographer", "unknown"),
                "source": source_name,
                "download_url": item["download_url"],
                "file_size_bytes": file_size,
            })

        except Exception as exc:
            logger.error("Failed to download %s: %s", item.get("id"), exc)

    logger.info("search_all_sources: downloaded %d files for %r", len(downloaded), query)
    return downloaded
