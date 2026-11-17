"""Paths and constants shared by the data pipeline.

Everything lives in the project folder: code in pipeline/, data in data/.
"""
from datetime import date
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
REFERENCE_DIR = DATA_DIR / "reference"    # source documents as downloaded (Nikkei PDFs, pages)
MEMBERSHIP_DIR = DATA_DIR / "membership"  # parsed index membership
YAHOO_DIR = DATA_DIR / "yahoo"            # corporate actions + check prices
JPX_RAW_DIR = PROJECT_DIR / "JPX Data"    # JPX daily-report files, downloaded by hand
JPX_MONTHLY_DIR = PROJECT_DIR / "JPX Monthly"  # JPX monthly quotation tables (詳細版), downloaded by hand

# First day of stock data (D9: no OCR of the 1996-1999 scans, so the stock data starts in 2000).
DATA_START = date(2000, 1, 1)

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

# Stocks kept in the merged price file: every Nikkei 225 member at any time since this date
# (the 394 companies identified on 2026-09-30). The Nikkei 300 was dropped: its change history is incomplete.
KEPT_SINCE = date(1996, 1, 1)

INDICES = {
    "nk225": {
        "name": "Nikkei 225",
        "history_url": "https://indexes.nikkei.co.jp/nkave/archives/file/"
                       "history_of_nikkei_stock_average_component_changes_en.pdf",
        "current_url": "https://indexes.nikkei.co.jp/en/nkave/index/component?idx=nk225",
        "date_format": "%b/%d/%Y",   # e.g. Apr/1/2026
        "size": 225,
    },
}
