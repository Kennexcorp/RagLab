"""
Data loading module for RAG system.
Supports loading dashboard data from various sources (JSON, CSV, API).
"""

import json
from datetime import datetime
from typing import Any

import pandas as pd

from raglab.utils import setup_logging, validate_file_exists


class DataLoader:
    """Load and process dashboard data from various sources."""

    def __init__(self, log_level: str = "INFO"):
        """
        Initialize DataLoader.

        Args:
            log_level: Logging level
        """
        self.logger = setup_logging(log_level)

    def load_json(self, file_path: str) -> list[dict[str, Any]]:
        """
        Load data from JSON file.

        Args:
            file_path: Path to JSON file

        Returns:
            List of data records
        """
        if not validate_file_exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        self.logger.info(f"Loading JSON data from {file_path}")

        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        # Ensure data is a list
        if isinstance(data, dict):
            data = [data]

        self.logger.info(f"Loaded {len(data)} records from JSON")
        return data

    def load_csv(self, file_path: str) -> list[dict[str, Any]]:
        """
        Load data from CSV file.

        Args:
            file_path: Path to CSV file

        Returns:
            List of data records
        """
        if not validate_file_exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        self.logger.info(f"Loading CSV data from {file_path}")

        df = pd.read_csv(file_path)
        data = df.to_dict("records")

        self.logger.info(f"Loaded {len(data)} records from CSV")
        return data

    def load_from_dict(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Load data from dictionary (e.g., API response).

        Args:
            data: Dictionary containing data

        Returns:
            List of data records
        """
        self.logger.info("Loading data from dictionary")

        if isinstance(data, dict):
            data = [data]

        self.logger.info(f"Loaded {len(data)} records from dictionary")
        return data

    def extract_text_from_record(
        self, record: dict[str, Any], text_fields: list[str] | None = None
    ) -> str:
        """
        Extract text content from a data record.

        Args:
            record: Data record
            text_fields: Specific fields to extract (if None, extracts all)

        Returns:
            Extracted text content
        """
        if text_fields:
            # Extract only specified fields
            text_parts = []
            for field in text_fields:
                if field in record:
                    value = record[field]
                    if value is not None:
                        text_parts.append(f"{field}: {value}")
        else:
            # Extract all fields
            text_parts = []
            for key, value in record.items():
                if value is not None and not isinstance(value, (dict, list)):
                    text_parts.append(f"{key}: {value}")

        return "\n".join(text_parts)

    def extract_metadata(
        self, record: dict[str, Any], metadata_fields: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Extract metadata from a data record.

        Args:
            record: Data record
            metadata_fields: Specific metadata fields to extract

        Returns:
            Metadata dictionary
        """
        metadata = {}

        # Common metadata fields
        common_fields = ["id", "timestamp", "date", "category", "source", "type"]

        fields_to_extract = metadata_fields if metadata_fields else common_fields

        for field in fields_to_extract:
            if field in record:
                metadata[field] = record[field]

        # Add extraction timestamp
        metadata["extracted_at"] = datetime.now().isoformat()

        return metadata

    def process_records(
        self,
        records: list[dict[str, Any]],
        text_fields: list[str] | None = None,
        metadata_fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Process records into format suitable for RAG system.

        Args:
            records: List of data records
            text_fields: Fields to use for text extraction
            metadata_fields: Fields to use for metadata

        Returns:
            List of processed documents with text and metadata
        """
        self.logger.info(f"Processing {len(records)} records")

        documents = []

        for i, record in enumerate(records):
            try:
                text = self.extract_text_from_record(record, text_fields)
                metadata = self.extract_metadata(record, metadata_fields)
                metadata["record_index"] = i

                documents.append({"text": text, "metadata": metadata})
            except Exception as e:
                self.logger.warning(f"Error processing record {i}: {str(e)}")
                continue

        self.logger.info(f"Successfully processed {len(documents)} documents")
        return documents

    def load_and_process(
        self,
        source: str,
        source_type: str = "json",
        text_fields: list[str] | None = None,
        metadata_fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Load and process data in one step.

        Args:
            source: File path or data source
            source_type: Type of source (json, csv, dict)
            text_fields: Fields to use for text extraction
            metadata_fields: Fields to use for metadata

        Returns:
            List of processed documents
        """
        # Load data based on source type
        if source_type == "json":
            records = self.load_json(source)
        elif source_type == "csv":
            records = self.load_csv(source)
        elif source_type == "dict":
            records = self.load_from_dict(source)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")

        # Process records
        documents = self.process_records(records, text_fields, metadata_fields)

        return documents


if __name__ == "__main__":
    # Example usage
    loader = DataLoader()

    # Example with sample data
    sample_data = [
        {
            "id": 1,
            "title": "Q4 Revenue Report",
            "content": "Revenue increased by 25% in Q4 2025",
            "category": "finance",
            "date": "2025-12-31",
        },
        {
            "id": 2,
            "title": "User Engagement Metrics",
            "content": "Daily active users reached 1M milestone",
            "category": "analytics",
            "date": "2026-01-15",
        },
    ]

    documents = loader.process_records(sample_data)
    print(f"Processed {len(documents)} documents")
    for doc in documents:
        print(f"\nText: {doc['text'][:100]}...")
        print(f"Metadata: {doc['metadata']}")
