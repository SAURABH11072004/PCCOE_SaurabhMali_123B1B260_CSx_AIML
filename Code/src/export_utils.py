"""Export Utility Module.
Handles structured serialization and downloading of architectural entities,
issues, and revision comparison reports to CSV and JSON formats.
"""

import json
from typing import List, Dict, Any
import pandas as pd


def export_dataframe_to_csv(df: pd.DataFrame) -> str:
    """Serialize DataFrame to CSV string."""
    return df.to_csv(index=False)


def export_dataframe_to_json(df: pd.DataFrame) -> str:
    """Serialize DataFrame to formatted JSON string."""
    return df.to_json(orient="records", indent=2)


def export_dict_to_json(data: Any) -> str:
    """Serialize Python dictionary or list to JSON string."""
    return json.dumps(data, indent=2, default=str)
