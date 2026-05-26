"""
Keytable loader for RAG system.

Converts date-stamped key table JSON API responses into RAG-ready documents.
Each row in the series tree becomes one document combining a static definition
with a dynamically generated data narrative for the specific reporting period.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils import setup_logging, validate_file_exists


_DEFINITIONS_PATH = Path(__file__).parent / "data" / "keytable-definitions.json"

# Threshold below which a q/q % change is described as "broadly flat"
_FLAT_THRESHOLD = 0.1


class KeytableLoader:
    """
    Load and convert key table JSON API responses into RAG documents.

    The JSON format expected:
      {
        "params": {"asOfDate": "YYYY-MM-DD", "taxonomy": "..."},
        "tableData": {
          "headers": [
            {"dataIndex": "N_YYYY_MM_DD", "title": "Q4'25"},   # value column
            {"dataIndex": "10_qq",        "title": "q/q %"},
            {"dataIndex": "11_yy",        "title": "y/y %"},
            {"dataIndex": "12_vs_5y_avg", "title": "vs 5y avg"}
          ],
          "series": [...]
        }
      }
    """

    def __init__(
        self,
        definitions_path: str = None,
        log_level: str = "INFO",
    ):
        self.logger = setup_logging(log_level)
        defs_file = Path(definitions_path) if definitions_path else _DEFINITIONS_PATH

        if not validate_file_exists(str(defs_file)):
            raise FileNotFoundError(f"Definitions file not found: {defs_file}")

        with open(defs_file, "r", encoding="utf-8") as f:
            self._defs = json.load(f)

        self.logger.info(f"Loaded {len(self._defs)} row definitions from {defs_file.name}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load a keytable JSON file and return RAG-ready documents.

        Each row in the series tree becomes one document containing:
          - The row's static definition (and formula if applicable)
          - A dynamically generated narrative with the period's values

        Metadata per document:
          date, period, quarter, year, taxonomy, row_title, source

        Args:
            file_path: Path to the keytable JSON file

        Returns:
            List of {"text": str, "metadata": dict} documents
        """
        if not validate_file_exists(file_path):
            raise FileNotFoundError(f"Keytable file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        params = data["params"]
        table = data["tableData"]

        as_of_date = params["asOfDate"]
        taxonomy = params["taxonomy"]
        period, quarter, year = self._derive_period(as_of_date)

        # Identify analytics columns by their stable key names, then find the
        # value column as the last header that is neither "rowTitle" nor an
        # analytics column.  This works on both the slim keytable-data.json
        # format (4 headers) and the full raw API response (13+ headers that
        # include a rowTitle sentinel and 8+ historical quarter columns).
        headers = table["headers"]
        qq_key = next((h["dataIndex"] for h in headers if h["dataIndex"] == "10_qq"), None)
        yy_key = next((h["dataIndex"] for h in headers if h["dataIndex"] == "11_yy"), None)
        vs5y_key = next((h["dataIndex"] for h in headers if h["dataIndex"] == "12_vs_5y_avg"), None)

        _skip = {"rowTitle", qq_key, yy_key, vs5y_key} - {None}
        value_headers = [h for h in headers if h["dataIndex"] not in _skip]
        if not value_headers:
            raise ValueError(
                f"No value column found in headers: {[h['dataIndex'] for h in headers]}"
            )
        value_key = value_headers[-1]["dataIndex"]  # most recent period column

        self.logger.info(
            f"Loading keytable: taxonomy={taxonomy}, date={as_of_date}, period={period}"
        )

        rows = self._flatten_series(table["series"])
        documents = []
        for row in rows:
            doc = self._build_document(
                row, value_key, qq_key, yy_key, vs5y_key,
                period, quarter, year, as_of_date, taxonomy,
            )
            if doc:
                documents.append(doc)

        self.logger.info(f"Built {len(documents)} documents from keytable")
        return documents

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _derive_period(self, as_of_date: str) -> Tuple[str, str, str]:
        """Return (period_label, quarter_label, year_str) from an ISO date string."""
        date = datetime.strptime(as_of_date, "%Y-%m-%d")
        quarter = (date.month - 1) // 3 + 1
        return f"Q{quarter} {date.year}", f"Q{quarter}", str(date.year)

    def _flatten_series(self, series: List[Dict]) -> List[Dict]:
        """Recursively flatten the nested series tree into a flat list of rows."""
        rows = []
        for row in series:
            rows.append(row)
            if "children" in row:
                rows.extend(self._flatten_series(row["children"]))
        return rows

    def _fmt_number(self, value: float) -> str:
        """Format a number with thousands separator and no trailing zeros."""
        formatted = f"{value:,.10f}".rstrip("0").rstrip(".")
        return formatted

    def _pct_val(self, col: Optional[str], row: Dict) -> Optional[float]:
        """Extract a percentage value from a row's column dict. Returns None if null."""
        if col is None:
            return None
        cell = row.get(col)
        if not isinstance(cell, dict):
            return None
        raw = cell.get("value")
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    def _pct_phrase(self, val: Optional[float], label: str) -> Optional[str]:
        """
        Build a human-readable percentage change phrase.

        Returns None if val is None (skip this metric in the narrative).
        """
        if val is None:
            return None

        abs_val = abs(val)

        if label == "q/q" and abs_val <= _FLAT_THRESHOLD:
            return f"broadly flat q/q ({val:g}%)"
        elif val > 0:
            return f"up {self._fmt_number(abs_val)}% {label}"
        else:
            return f"down {self._fmt_number(abs_val)}% {label}"

    def _build_narrative(
        self,
        row_title: str,
        value: float,
        qq: Optional[float],
        yy: Optional[float],
        vs5y: Optional[float],
        period: str,
        unit: str,
    ) -> str:
        """Compose the dynamic data sentence for a row."""
        base = f"{row_title} in {period} was {self._fmt_number(value)} {unit}"

        pct_parts = [
            p for p in [
                self._pct_phrase(qq, "q/q"),
                self._pct_phrase(yy, "y/y"),
            ]
            if p is not None
        ]

        if vs5y is not None:
            direction = "above" if vs5y >= 0 else "below"
            vs5y_phrase = f"{self._fmt_number(abs(vs5y))}% {direction} its five-year average"
            if pct_parts:
                pct_parts.append(f"and {vs5y_phrase}")
            else:
                pct_parts.append(vs5y_phrase)

        if pct_parts:
            return base + ", " + ", ".join(pct_parts) + "."
        return base + "."

    def _build_document(
        self,
        row: Dict,
        value_key: str,
        qq_key: Optional[str],
        yy_key: Optional[str],
        vs5y_key: Optional[str],
        period: str,
        quarter: str,
        year: str,
        as_of_date: str,
        taxonomy: str,
    ) -> Optional[Dict[str, Any]]:
        """Build one RAG document from a series row."""
        row_title = row.get("rowTitle", "").strip()
        if not row_title:
            return None

        raw_value = row.get(value_key)
        if raw_value is None:
            return None
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return None

        row_def = self._defs.get(row_title, {})
        unit = row_def.get("unit", "tonnes")
        definition = row_def.get("definition", "")
        formula = row_def.get("formula", "")

        if not definition:
            self.logger.warning(f"No definition for row: '{row_title}'")

        qq = self._pct_val(qq_key, row)
        yy = self._pct_val(yy_key, row)
        vs5y = self._pct_val(vs5y_key, row)

        narrative = self._build_narrative(row_title, value, qq, yy, vs5y, period, unit)

        # Assemble full text: definition → formula (if any) → data narrative
        text_parts = [p for p in [definition, formula, narrative] if p]
        text = "\n\n".join(text_parts)

        return {
            "text": text,
            "metadata": {
                "date": as_of_date,
                "period": period,
                "quarter": quarter,
                "year": year,
                "taxonomy": taxonomy,
                "row_title": row_title,
                "source": f"{taxonomy}:{as_of_date}",
            },
        }


if __name__ == "__main__":
    import sys

    file_path = sys.argv[1] if len(sys.argv) > 1 else "keytable-data.json"
    loader = KeytableLoader()
    docs = loader.load(file_path)

    print(f"\nLoaded {len(docs)} documents\n")
    for doc in docs:
        meta = doc["metadata"]
        print(f"[{meta['row_title']}] ({meta['period']})")
        print(doc["text"])
        print()
