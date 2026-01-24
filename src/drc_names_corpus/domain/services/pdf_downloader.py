from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import httpx
from tqdm import tqdm

from drc_names_corpus.core import get_dataset_path

logger = logging.getLogger(__name__)


class PdfDownloader:
    """Download PDF files into a destination directory."""

    def __init__(self) -> None:
        self.target_dir = get_dataset_path("bronze", "pdf")
        headers = {"User-Agent": "drc-names-corpus/0.1 (+https://edu-nc.gouv.cd/)"}
        self._client = httpx.Client(
            headers=headers, timeout=600.0, follow_redirects=True
        )
        self._closed = False

    def close(self) -> None:
        if not self._closed:
            self._client.close()
            self._closed = True

    def __enter__(self) -> "PdfDownloader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def fetch_html(self, url: str) -> str:
        response = self._client.get(url)
        response.raise_for_status()
        return response.text

    def _slugify(self, value: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip()).strip("-").lower()
        return slug or "document"

    def _default_filename(self, url: str) -> str:
        parsed = urlparse(url)
        name = Path(parsed.path).stem or parsed.path or "document"
        return self._slugify(name)

    def download_pdf(self, url: str, filename: str | None = None) -> Path | None:
        safe_name = self._slugify(filename) if filename else self._default_filename(url)
        dest_path = self.target_dir / f"{safe_name}.pdf"
        if dest_path.exists() and dest_path.stat().st_size > 0:
            return dest_path
        try:
            with self._client.stream("GET", url) as response:
                response.raise_for_status()
                with dest_path.open("wb") as handle:
                    for chunk in response.iter_bytes():
                        handle.write(chunk)
        except Exception as exc:
            logger.error("Failed to download %s: %s", url, exc)
            return None
        return dest_path

    def download_pdfs(self, urls: Iterable[str]) -> list[Path]:
        downloaded: list[Path] = []
        for url in tqdm(urls, desc="Downloading PDFs", unit="file"):
            path = self.download_pdf(url)
            if path is not None:
                downloaded.append(path)
        logger.info("Downloaded %s PDFs", len(downloaded))
        return downloaded
