import json

from openbench.util.report import ReportGenerator


def test_report_includes_uncertainty_summary(tmp_path):
    uncertainty = tmp_path / "uncertainty"
    uncertainty.mkdir()
    (uncertainty / "summary.json").write_text(
        json.dumps(
            {
                "bootstrap": [
                    {
                        "variable": "Flow",
                        "reference": "GRDC",
                        "simulation": "ModelA",
                        "metric": "RMSE",
                        "scope": "station_network",
                        "valid_pair_count": 240,
                        "segment_count": 2,
                        "status": "available",
                        "estimate": 1.2,
                        "lower": 0.8,
                        "upper": 1.5,
                    }
                ],
                "verdicts": [
                    {
                        "variable": "Flow",
                        "metric": "RMSE",
                        "simulation_a": "ModelA",
                        "simulation_b": "ModelB",
                        "status": "indistinguishable",
                        "winner": None,
                    }
                ],
                "products": {
                    "model_spread": ["uncertainty/model_spread/flow.nc"],
                    "reference_sensitivity": [],
                },
            }
        ),
        encoding="utf-8",
    )
    generator = ReportGenerator(
        {
            "evaluation_items": [],
            "metrics": {},
            "scores": {},
            "comparisons": {},
        },
        str(tmp_path),
    )

    data = generator._collect_report_data()
    html_path = generator._generate_html_report(data, "report")
    html = open(html_path, encoding="utf-8").read()

    assert data["uncertainty"]["bootstrap"][0]["metric"] == "RMSE"
    assert "Uncertainty-aware Evaluation" in html
    assert "indistinguishable" in html
