from __future__ import annotations

import logging

from drc_names_corpus.domain import PdfDownloader, PdfLinkExtractor

logger = logging.getLogger(__name__)

DATA_URL = "https://edu-nc.gouv.cd/palmares_exetat2/"


def collect() -> None:
    logger.info("Collecting PDFs from %s", DATA_URL)
    with PdfDownloader() as downloader:
        html = downloader.fetch_html(DATA_URL)
        links = PdfLinkExtractor.extract_links(html, DATA_URL)
        logger.info("Discovered %s PDF links", len(links))
        downloaded = downloader.download_pdfs(links)
        logger.info("Downloaded %s PDFs", len(downloaded))


if __name__ == "__main__":
    collect()
