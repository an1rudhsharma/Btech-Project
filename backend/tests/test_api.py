# -*- coding: utf-8 -*-
"""Comprehensive test suite for all API endpoints with edge cases."""

import pytest
import os
import io
import json
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app

client = TestClient(app)


class TestUploadEndpoint:
    """Tests for POST /api/upload"""

    def _make_csv(self, df, filename="test.csv"):
        buffer = io.BytesIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        return ("file", (filename, buffer, "text/csv"))

    def _make_excel(self, df, filename="test.xlsx"):
        buffer = io.BytesIO()
        df.to_excel(buffer, index=False)
        buffer.seek(0)
        return ("file", (filename, buffer, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"))

    def _make_zip(self, files_dict, zipname="data.zip"):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            for name, df in files_dict.items():
                csv_buf = io.BytesIO()
                df.to_csv(csv_buf, index=False)
                zf.writestr(name, csv_buf.getvalue())
        buffer.seek(0)
        return ("file", (zipname, buffer, "application/zip"))

    # --- VALID UPLOADS ---

    def test_valid_csv_right_names_right_data(self):
        df = pd.DataFrame({"price": [100, 200, 150, 50, 300], "churn": [0, 1, 0, 1, 0], "usage": [30, 80, 50, 10, 90]})
        r = client.post("/api/upload", files=[self._make_csv(df)])
        assert r.status_code == 200
        d = r.json()
        assert d["status"] == "success"
        assert d["rows"] == 5
        assert "price" in d["detected_mapping"]
        assert len(d["data_issues"]) == 0

    def test_valid_excel_upload(self):
        df = pd.DataFrame({"price": [100, 200], "demand": [500, 300]})
        r = client.post("/api/upload", files=[self._make_excel(df)])
        assert r.status_code == 200
        assert r.json()["status"] == "success"

    def test_csv_latin1_encoding(self):
        csv_bytes = b"price,description\n100,Caf\xe9\n200,r\xe9sum\xe9\n150,na\xefve\n"
        r = client.post("/api/upload", files=[("file", ("latin.csv", io.BytesIO(csv_bytes), "text/csv"))])
        assert r.status_code == 200
        assert r.json()["rows"] == 3

    def test_csv_cp1252_encoding(self):
        csv_bytes = b"price,name\n99,Item\x99\n149,Smart\x96dash\n"
        r = client.post("/api/upload", files=[("file", ("cp1252.csv", io.BytesIO(csv_bytes), "text/csv"))])
        assert r.status_code == 200

    def test_zip_multiple_files(self):
        files = {
            "churn.csv": pd.DataFrame({"churn": [0, 1, 0], "price": [100, 200, 150]}),
            "marketing.csv": pd.DataFrame({"impressions": [1000, 2000], "clicks": [50, 100]}),
        }
        r = client.post("/api/upload", files=[self._make_zip(files)])
        assert r.status_code == 200
        d = r.json()
        assert d["type"] == "zip"
        assert d["total_files"] == 2

    def test_zip_with_subdirectories(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            df = pd.DataFrame({"price": [100, 200], "churn": [0, 1]})
            csv_buf = io.BytesIO()
            df.to_csv(csv_buf, index=False)
            zf.writestr("subdir/data.csv", csv_buf.getvalue())
        buffer.seek(0)
        r = client.post("/api/upload", files=[("file", ("nested.zip", buffer, "application/zip"))])
        assert r.status_code == 200
        assert r.json()["total_files"] == 1

    # --- WRONG COLUMN NAMES, RIGHT DATA ---

    def test_wrong_name_right_data_infers_price(self):
        df = pd.DataFrame({"cost_per_unit": [100, 200, 150, 50, 300], "left_company": [0, 1, 0, 1, 0]})
        r = client.post("/api/upload", files=[self._make_csv(df)])
        assert r.status_code == 200
        d = r.json()
        assert "price" in d["detected_mapping"]

    # --- RIGHT COLUMN NAMES, WRONG DATA ---

    def test_right_name_all_text_in_price(self):
        df = pd.DataFrame({"price": ["abc", "def", "ghi", "jkl", "mno"], "churn": ["yes", "no", "yes", "no", "maybe"]})
        r = client.post("/api/upload", files=[self._make_csv(df)])
        d = r.json()
        if r.status_code == 200:
            assert len(d.get("data_issues", [])) > 0 or len(d.get("rejected_columns", [])) > 0
        else:
            assert r.status_code == 400

    def test_right_name_partial_bad_data(self):
        df = pd.DataFrame({"price": [100, 200, "N/A", 50, "bad", 250, 175, 80, 90, 110], "churn": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]})
        r = client.post("/api/upload", files=[self._make_csv(df)])
        assert r.status_code == 200

    # --- WRONG NAME + WRONG DATA ---

    def test_garbage_data(self):
        df = pd.DataFrame({"col_a": ["asdf", "qwer", "zxcv"], "col_b": ["!@#$", "^&*(", "[]{}"]})
        r = client.post("/api/upload", files=[self._make_csv(df)])
        d = r.json()
        if r.status_code == 200:
            assert d.get("trainable_models", []) == [] or len(d.get("data_issues", [])) > 0

    # --- EDGE CASES ---

    def test_empty_csv(self):
        r = client.post("/api/upload", files=[("file", ("empty.csv", io.BytesIO(b"price,churn\n"), "text/csv"))])
        assert r.status_code == 400

    def test_single_row(self):
        df = pd.DataFrame({"price": [100], "churn": [0]})
        r = client.post("/api/upload", files=[self._make_csv(df)])
        assert r.status_code == 200

    def test_100_columns(self):
        df = pd.DataFrame({f"col_{i}": range(10) for i in range(100)})
        r = client.post("/api/upload", files=[self._make_csv(df)])
        assert r.status_code == 200

    def test_columns_with_spaces(self):
        df = pd.DataFrame({"Product Price": [100, 200], "Customer Churn": [0, 1], "marketing spend": [5000, 3000]})
        r = client.post("/api/upload", files=[self._make_csv(df)])
        assert r.status_code == 200

    def test_all_nulls(self):
        df = pd.DataFrame({"price": [None, None, None], "churn": [None, None, None]})
        r = client.post("/api/upload", files=[self._make_csv(df)])
        assert r.status_code in [200, 400]

    def test_mixed_types(self):
        df = pd.DataFrame({"price": [100, "200.5", "N/A", 50, "abc"], "churn": [0, 1, 0, 1, 0]})
        r = client.post("/api/upload", files=[self._make_csv(df)])
        assert r.status_code == 200

    def test_duplicate_column_names(self):
        r = client.post("/api/upload", files=[("file", ("dupes.csv", io.BytesIO(b"price,price,churn\n100,200,0\n150,250,1\n"), "text/csv"))])
        assert r.status_code == 200

    def test_very_long_strings(self):
        df = pd.DataFrame({"text": ["x" * 10000, "y" * 5000, "Short"], "churn": [0, 1, 0]})
        r = client.post("/api/upload", files=[self._make_csv(df)])
        assert r.status_code == 200

    def test_negative_values(self):
        df = pd.DataFrame({"price": [-100, -50, 0, 50, 100], "churn": [0, 1, 0, 1, 0]})
        r = client.post("/api/upload", files=[self._make_csv(df)])
        assert r.status_code == 200

    def test_unsupported_file_type(self):
        r = client.post("/api/upload", files=[("file", ("data.txt", io.BytesIO(b"hello"), "text/plain"))])
        assert r.status_code == 400

    def test_corrupted_zip(self):
        r = client.post("/api/upload", files=[("file", ("bad.zip", io.BytesIO(b"PK\x03\x04not real zip"), "application/zip"))])
        assert r.status_code == 400

    def test_zip_no_data_files(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("readme.txt", "just text")
        buf.seek(0)
        r = client.post("/api/upload", files=[("file", ("nodata.zip", buf, "application/zip"))])
        assert r.status_code == 400

    def test_no_file_in_request(self):
        r = client.post("/api/upload")
        assert r.status_code == 422


class TestTrainEndpoint:

    def test_train_all(self):
        r = client.post("/api/train/all")
        assert r.status_code == 200
        d = r.json()
        assert "churn" in d
        assert "pricing" in d

    def test_train_invalid_model(self):
        r = client.post("/api/train", json={"dataset_path": "/fake.csv", "model_name": "bogus"})
        assert r.status_code == 400

    def test_train_nonexistent_path(self):
        r = client.post("/api/train", json={"dataset_path": "/tmp/no_such_file_999.csv", "model_name": "churn"})
        assert r.status_code == 400

    def test_train_churn_non_binary_target(self):
        df = pd.DataFrame({"price": [100, 200, 150, 50, 300], "churn": [0, 1, 2, 3, 4], "usage": [30, 80, 50, 10, 90]})
        path = "/tmp/_test_nonbinary.csv"
        df.to_csv(path, index=False)
        r = client.post("/api/train", json={"dataset_path": path, "model_name": "churn"})
        assert r.status_code == 400
        os.remove(path)

    def test_train_churn_all_null_target(self):
        df = pd.DataFrame({"price": [100, 200, 150], "churn": [None, None, None], "usage": [30, 80, 50]})
        path = "/tmp/_test_nulltarget.csv"
        df.to_csv(path, index=False)
        r = client.post("/api/train", json={"dataset_path": path, "model_name": "churn"})
        assert r.status_code == 400
        os.remove(path)

    def test_train_sentiment_no_text_col(self):
        df = pd.DataFrame({"price": [100, 200], "churn": [0, 1]})
        path = "/tmp/_test_notext.csv"
        df.to_csv(path, index=False)
        r = client.post("/api/train", json={"dataset_path": path, "model_name": "sentiment"})
        assert r.status_code == 400
        os.remove(path)


class TestChatEndpoint:

    def test_business_query(self):
        r = client.post("/api/chat", json={"message": "What happens if I raise price to 200?"})
        assert r.status_code == 200
        assert "insight" in r.json() or "error" in r.json()

    def test_unrelated_query(self):
        r = client.post("/api/chat", json={"message": "What is the weather in Delhi?"})
        assert r.status_code == 200
        d = r.json()
        # LLM should either classify as unrelated or return some response
        assert "insight" in d or "error" in d or "raw_response" in d

    def test_empty_message(self):
        r = client.post("/api/chat", json={"message": ""})
        assert r.status_code in [200, 400, 422]

    def test_long_message(self):
        r = client.post("/api/chat", json={"message": "price 200 " * 500})
        assert r.status_code == 200

    def test_special_chars(self):
        r = client.post("/api/chat", json={"message": "What about 25% increase?"})
        assert r.status_code == 200

    def test_injection_attempt(self):
        r = client.post("/api/chat", json={"message": "'; DROP TABLE users; --"})
        assert r.status_code == 200

    def test_missing_message_field(self):
        r = client.post("/api/chat", json={})
        assert r.status_code == 422


class TestSimulateEndpoint:

    def test_default_params(self):
        r = client.post("/api/simulate", json={"price": 100, "marketing_spend": 5000, "num_features": 5, "usage": 50, "impressions": 10000, "clicks": 500, "text": "good product"})
        assert r.status_code == 200
        assert "churn" in r.json() or "pricing" in r.json()

    def test_extreme_high_values(self):
        r = client.post("/api/simulate", json={"price": 999999, "marketing_spend": 10000000, "num_features": 100, "usage": 100, "impressions": 100000000, "clicks": 10000000, "text": "amazing"})
        assert r.status_code == 200

    def test_all_zeros(self):
        r = client.post("/api/simulate", json={"price": 0, "marketing_spend": 0, "num_features": 0, "usage": 0, "impressions": 0, "clicks": 0, "text": ""})
        assert r.status_code == 200

    def test_negative_values(self):
        r = client.post("/api/simulate", json={"price": -50, "marketing_spend": -1000, "num_features": -1, "usage": -10, "impressions": -100, "clicks": -5, "text": "bad"})
        assert r.status_code == 200

    def test_compare_scenarios(self):
        r = client.post("/api/simulate/compare", json={
            "baseline": {"price": 100, "marketing_spend": 5000, "num_features": 5, "usage": 50, "impressions": 10000, "clicks": 500},
            "scenario": {"price": 200, "marketing_spend": 10000, "num_features": 5, "usage": 50, "impressions": 10000, "clicks": 500},
            "text": "good",
        })
        assert r.status_code == 200


class TestCounterfactualEndpoint:

    def test_churn_counterfactual(self):
        r = client.post("/api/counterfactual", json={"model_name": "churn", "scenario": {"price": 200, "marketing_spend": 2000, "num_features": 2, "usage": 20, "tenure": 3, "satisfaction": 2}, "total_cfs": 3})
        assert r.status_code == 200

    def test_invalid_model(self):
        r = client.post("/api/counterfactual", json={"model_name": "nonexistent", "scenario": {"price": 100}, "total_cfs": 3})
        assert r.status_code in [200, 400]


class TestStatusEndpoint:

    def test_status(self):
        r = client.get("/api/status")
        assert r.status_code == 200
        d = r.json()
        assert "models" in d
        for name in ["churn", "pricing", "marketing", "sentiment"]:
            assert name in d["models"]
            assert "trained" in d["models"][name]
            assert "metrics" in d["models"][name]


class TestDatasetsEndpoint:

    def test_list(self):
        r = client.get("/api/upload/datasets")
        assert r.status_code == 200
        assert "datasets" in r.json()
        assert isinstance(r.json()["datasets"], list)


class TestIntegration:

    def test_upload_train_simulate(self):
        df = pd.DataFrame({
            "price": np.random.uniform(50, 500, 100),
            "churn": np.random.randint(0, 2, 100),
            "usage": np.random.uniform(10, 90, 100),
            "marketing_spend": np.random.uniform(1000, 20000, 100),
            "num_features": np.random.randint(1, 15, 100),
            "tenure": np.random.randint(1, 60, 100),
            "satisfaction": np.random.uniform(1, 5, 100),
        })
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        r = client.post("/api/upload", files=[("file", ("integ.csv", buf, "text/csv"))])
        assert r.status_code == 200
        path = r.json()["path"]

        r2 = client.post("/api/train", json={"dataset_path": path, "model_name": "churn"})
        assert r2.status_code == 200

        r3 = client.post("/api/simulate", json={"price": 200, "marketing_spend": 5000, "num_features": 5, "usage": 50, "impressions": 10000, "clicks": 500, "text": "ok"})
        assert r3.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
