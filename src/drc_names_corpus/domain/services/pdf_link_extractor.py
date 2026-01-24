from __future__ import annotations

import logging
from urllib.parse import urljoin

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class PdfLinkExtractor:
    """Extract PDF links from an HTML page."""

    @staticmethod
    def extract_links(html: str, base_url: str) -> list[str]:
        soup = BeautifulSoup(html, "html.parser")
        links = []
        for anchor in soup.find_all("a", href=True):
            href = str(anchor.get("href", "")).strip()
            if href.lower().endswith(".pdf"):
                links.append(urljoin(base_url, href))
        logger.info("Extracted %s PDF links from HTML", len(links))
        return links
