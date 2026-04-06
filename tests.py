from __future__ import annotations

import unittest

import pandas as pd

import eda as EDA
import feature_engineering
import data_cleaning as cleaning
from app import app, load_builtin_dataset


TEST_DATA_PATH = "test_data/sleep_mobile_stress_dataset_15000.csv"


class LocalIntegrationSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.df = pd.read_csv(TEST_DATA_PATH)

    def test_shiny_app_imports(self) -> None:
        self.assertIsNotNone(app)

    def test_builtin_loaders(self) -> None:
        sleep_df = load_builtin_dataset("sleep_health")
        iris_df = load_builtin_dataset("iris")
        self.assertGreater(len(sleep_df), 0)
        self.assertGreater(len(iris_df), 0)

    def test_cleaning_module(self) -> None:
        transformed = cleaning.remove_duplicates(self.df)
        self.assertIsInstance(transformed, pd.DataFrame)
        self.assertGreater(len(transformed), 0)

    def test_feature_engineering_module(self) -> None:
        transformed, meta = feature_engineering.apply_feature_engineering_to_df(
            self.df,
            "interaction",
            "daily_screen_time_hours",
            col2="stress_level",
        )
        self.assertIn("daily_screen_time_hours_x_stress_level", transformed.columns)
        self.assertEqual(meta["feature_type"], "interaction")

    def test_eda_summaries(self) -> None:
        head_payload = EDA.show_head(self.df, n=5)
        describe_payload = EDA.describe_dataframe(self.df)
        types_payload = EDA.column_types(self.df)
        self.assertEqual(head_payload["status"], "success")
        self.assertEqual(describe_payload["status"], "success")
        self.assertEqual(types_payload["status"], "success")

    def test_eda_correlation_matrix(self) -> None:
        payload = EDA.correlation_matrix(self.df)
        self.assertEqual(payload["status"], "success")
        self.assertIn("values", payload["data"])
        self.assertGreater(len(payload["data"]["columns"]), 1)

    def test_knn_imputation(self) -> None:
        # Introduce some NaN values to test k-NN imputation
        df = self.df.copy()
        df.loc[0:4, "age"] = None
        result, warning = cleaning.knn_impute(df, ["age"], k=3)
        self.assertIsInstance(result, pd.DataFrame)
        # All NaN values in 'age' should be filled
        self.assertEqual(result["age"].isnull().sum(), 0)

    def test_custom_expression(self) -> None:
        transformed, meta = feature_engineering.apply_feature_engineering_to_df(
            self.df,
            "custom_expr",
            expr="age * 2 + stress_level",
            new_column="test_custom",
        )
        self.assertIn("test_custom", transformed.columns)
        self.assertEqual(meta["feature_type"], "custom_expr")

    def test_eda_plots(self) -> None:
        one_d = EDA.plot_numeric_1d(self.df, "age", bins=12)
        two_d = EDA.plot_two_columns(self.df, "age", "stress_level", kind="scatter")
        regression = EDA.regression_analysis(
            self.df,
            "daily_screen_time_hours",
            "stress_level",
        )
        multiline = EDA.plot_multiline(
            self.df,
            "sleep_duration_hours",
            group_by="gender",
            nbins=10,
        )
        for payload in [one_d, two_d, regression, multiline]:
            self.assertEqual(payload["status"], "success")


if __name__ == "__main__":
    unittest.main()
