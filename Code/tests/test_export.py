"""Tests for CSV and JSON export utilities."""

import json
import pandas as pd
from src.export_utils import export_dataframe_to_csv, export_dataframe_to_json, export_dict_to_json


def test_export_dataframe_to_csv():
    df = pd.DataFrame([{"col1": "val1", "col2": 42}])
    csv_str = export_dataframe_to_csv(df)
    assert "col1,col2" in csv_str
    assert "val1,42" in csv_str


def test_export_dataframe_to_json():
    df = pd.DataFrame([{"name": "ThrottleControl_SWC", "type": "Component"}])
    json_str = export_dataframe_to_json(df)
    parsed = json.loads(json_str)
    assert isinstance(parsed, list)
    assert parsed[0]["name"] == "ThrottleControl_SWC"


def test_export_dict_to_json():
    data = {"metric": "Recall@k", "value": 1.0}
    json_str = export_dict_to_json(data)
    parsed = json.loads(json_str)
    assert parsed["metric"] == "Recall@k"
    assert parsed["value"] == 1.0
