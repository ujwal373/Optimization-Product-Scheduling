"""
Data loading utilities for MAPSO

Loads datasets from various formats (JSON, CSV, Excel).
"""

import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np

from mapso.core.models import (
    SKU,
    Machine,
    Job,
    ChangeoverMatrix,
    Shift,
    ShiftCalendar,
    ShiftType,
)


class DataLoader:
    """Loads scheduling datasets from files"""

    @staticmethod
    def load_skus(file_path: str) -> List[SKU]:
        """Load SKUs from JSON file"""
        with open(file_path, "r") as f:
            skus_data = json.load(f)

        skus = []
        for data in skus_data:
            sku = SKU(**data)
            skus.append(sku)

        return skus

    @staticmethod
    def load_jobs(file_path: str) -> List[Job]:
        """Load jobs from JSON file"""
        with open(file_path, "r") as f:
            jobs_data = json.load(f)

        jobs = []
        for data in jobs_data:
            # Convert datetime strings
            data["due_date"] = datetime.fromisoformat(data["due_date"])
            data["release_date"] = datetime.fromisoformat(data["release_date"])

            job = Job(**data)
            jobs.append(job)

        return jobs

    @staticmethod
    def load_changeover_matrix(file_path: str) -> ChangeoverMatrix:
        """Load changeover matrix from JSON file"""
        with open(file_path, "r") as f:
            data = json.load(f)

        return ChangeoverMatrix(
            skus=data["skus"],
            setup_times=np.array(data["setup_times"]),
            setup_costs=np.array(data["setup_costs"]),
        )

    @staticmethod
    def load_historical_demand(file_path: str) -> pd.DataFrame:
        """Load historical demand from CSV file"""
        df = pd.read_csv(file_path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    @staticmethod
    def load_dataset(directory_path: str) -> Dict[str, Any]:
        """
        Load complete dataset from directory

        Args:
            directory_path: Directory containing dataset files

        Returns:
            Dictionary with dataset components
        """
        dataset = {}

        # Load SKUs
        skus_path = os.path.join(directory_path, "skus.json")
        if os.path.exists(skus_path):
            dataset["skus"] = DataLoader.load_skus(skus_path)

        # Load jobs
        jobs_path = os.path.join(directory_path, "jobs.json")
        if os.path.exists(jobs_path):
            dataset["jobs"] = DataLoader.load_jobs(jobs_path)

        # Load changeover matrix
        changeover_path = os.path.join(directory_path, "changeover_matrix.json")
        if os.path.exists(changeover_path):
            dataset["changeover_matrix"] = DataLoader.load_changeover_matrix(changeover_path)

        # Load historical demand
        demand_path = os.path.join(directory_path, "historical_demand.csv")
        if os.path.exists(demand_path):
            dataset["historical_demand"] = DataLoader.load_historical_demand(demand_path)

        # Load metadata
        metadata_path = os.path.join(directory_path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                dataset["metadata"] = json.load(f)

        return dataset
