import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).parents[1] / "main_figures" / "aggregate_fig2_metrics.py"


def load_aggregate_module():
    spec = importlib.util.spec_from_file_location("aggregate_fig2_metrics", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_metric_csvs(holdout_dir: Path, tag: str) -> None:
    holdout_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "tag": [tag, tag],
            "W1 Distance": [1.0, 3.0],
            "TMV": [0.2, 0.4],
        }
    ).to_csv(holdout_dir / "w1tmv.csv", index=False)
    pd.DataFrame(
        {
            "tag": [tag, tag],
            "Spearman": [0.5, 0.7],
            "Kendall": [0.4, 0.6],
        }
    ).to_csv(holdout_dir / "pseudotime.csv", index=False)
    pd.DataFrame(
        {
            "tag": [tag, tag],
            "Run": [0, 1],
            "TC": [0.1, 0.3],
            "VC": [0.8, 1.0],
        }
    ).to_csv(holdout_dir / "tcvc.csv", index=False)


class AggregateFig2MetricsPublicTest(unittest.TestCase):
    def test_module_exists(self):
        self.assertTrue(MODULE_PATH.exists(), "public aggregate_fig2_metrics.py should exist")

    def test_parses_scvi_nobatch_and_alignment_from_tag(self):
        agg = load_aggregate_module()
        parsed = agg.parse_tag_fields(
            "scvi_nobatch_deepruot_dim10_otmoderuot_gpa_consensus_refhvg"
        )
        self.assertEqual(parsed["model"], "scvi_nobatch")
        self.assertEqual(parsed["dim"], 10)
        self.assertEqual(parsed["otmode"], "ruot")
        self.assertEqual(parsed["alignment"], "gpa")

    def test_aggregates_one_dataset_from_flat_holdout_dirs(self):
        agg = load_aggregate_module()
        gpa_tag = "hvg_deepruot_dim10_otmoderuot_gpa_consensus_refhvg"
        identity_tag = "scvi_deepruot_dim10_otmoderuot_identity_refhvg"

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            gpa_root = tmp / "gpa"
            ident_root = tmp / "identity"
            _write_metric_csvs(gpa_root / "EMT_holdt0", gpa_tag)
            _write_metric_csvs(gpa_root / "EMT_holdt1", gpa_tag)
            _write_metric_csvs(ident_root / "EMT_holdt0", identity_tag)

            out_dir = tmp / "out"
            agg.aggregate_metrics(
                results_roots=[gpa_root, ident_root],
                dataset="EMT",
                output_dir=out_dir,
            )

            w1 = pd.read_csv(out_dir / "all_w1.csv")
            self.assertEqual(set(w1["dataset"]), {"EMT"})
            self.assertEqual(set(w1["time"]), {0, 1})
            self.assertEqual(set(w1["alignment"]), {"gpa", "identity"})
            self.assertEqual(set(w1["model"]), {"hvg", "scvi"})
            gpa_row = w1.loc[(w1["alignment"] == "gpa") & (w1["time"] == 0)].iloc[0]
            self.assertEqual(gpa_row["W1_Mean"], 2.0)

            pseudo = pd.read_csv(out_dir / "all_pseudotime.csv")
            tcvc = pd.read_csv(out_dir / "all_tcvc.csv")
            self.assertEqual(len(pseudo), 3)
            self.assertEqual(len(tcvc), 3)
            self.assertIn("Spearman_Mean", pseudo.columns)
            self.assertEqual(tcvc.loc[tcvc["alignment"] == "identity", "TC"].iloc[0], 0.2)

    def test_nested_dataset_holdt_layout(self):
        agg = load_aggregate_module()
        tag = "scvi_nobatch_deepruot_dim10_otmoderuot_gpa_consensus_refhvg"
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            holdout = tmp / "run" / "EMT" / "holdt0"
            _write_metric_csvs(holdout, tag)
            rows = list(agg.iter_holdout_dirs([tmp / "run"], dataset="EMT"))
            self.assertEqual([(ds, t) for ds, t, _ in rows], [("EMT", 0)])

    def test_public_api_has_no_override_hooks(self):
        agg = load_aggregate_module()
        self.assertFalse(hasattr(agg, "merge_with_gpa_override"))
        self.assertFalse(hasattr(agg, "merge_with_extra_rows"))


if __name__ == "__main__":
    unittest.main()
