from __future__ import annotations

import base64
import io
import logging
import re
from pathlib import Path

import numpy as np
from PIL import Image

from openbench.util.report import ReportGenerator

_EMBEDDED_IMG = re.compile(r'<img src="data:([^;"]+);base64,([^"]+)" alt="([^"]+)"')


def _generator(case_dir: Path) -> ReportGenerator:
    return ReportGenerator(
        {"evaluation_items": ["Runoff"], "metrics": {}, "scores": {}, "comparisons": {}},
        str(case_dir),
    )


def _embedded_images(html: str) -> dict[str, tuple[str, bytes]]:
    return {alt: (mime, base64.b64decode(data)) for mime, data, alt in _EMBEDDED_IMG.findall(html)}


def test_report_writes_standalone_html_with_downscaled_embedded_figures(tmp_path):
    case_dir = tmp_path / "case"
    metrics_dir = case_dir / "metrics"
    scores_dir = case_dir / "scores"
    metrics_dir.mkdir(parents=True)
    scores_dir.mkdir(parents=True)

    # A 300 dpi map: opaque RGBA as Matplotlib writes it, too noisy for PNG to stay small.
    pixels = np.random.default_rng(0).integers(0, 256, (1600, 3200, 3), dtype=np.uint8)
    Image.fromarray(pixels).convert("RGBA").save(metrics_dir / "Runoff_bias map#1.png")
    legend = Image.new("RGBA", (200, 100), (255, 0, 0, 255))
    legend.paste((0, 0, 0, 0), (0, 0, 100, 100))
    legend.save(metrics_dir / "Runoff_legend.png")
    svg = b'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><rect width="10" height="10"/></svg>'
    (scores_dir / "Runoff_score.svg").write_bytes(svg)

    result = _generator(case_dir).generate_report()

    reports_dir = case_dir / "reports"
    assert result == {
        "html": str(reports_dir / "evaluation_report.html"),
        "standalone_html": str(reports_dir / "evaluation_report_standalone.html"),
    }
    assert not list(reports_dir.glob("*.pdf"))
    assert "figures/metrics/Runoff_bias%20map%231.png" in Path(result["html"]).read_text(encoding="utf-8")

    standalone = Path(result["standalone_html"]).read_text(encoding="utf-8")
    assert 'src="figures/' not in standalone
    images = _embedded_images(standalone)
    assert set(images) == {"Runoff_bias map#1.png", "Runoff_legend.png", "Runoff_score.svg"}

    mime, data = images["Runoff_bias map#1.png"]
    assert mime == "image/jpeg"
    with Image.open(io.BytesIO(data)) as image:
        assert image.size == (1200, 600)

    mime, data = images["Runoff_legend.png"]
    assert mime == "image/png"
    with Image.open(io.BytesIO(data)) as image:
        assert image.getchannel("A").getextrema() == (0, 255)

    assert images["Runoff_score.svg"] == ("image/svg+xml", svg)


def test_standalone_report_embeds_unreadable_figures_as_is_and_keeps_missing_links(tmp_path, caplog):
    case_dir = tmp_path / "case"
    figure_dir = case_dir / "reports" / "figures" / "metrics"
    figure_dir.mkdir(parents=True)
    (figure_dir / "Runoff_fake.jpg").write_bytes(b"fake jpg")
    html_path = case_dir / "reports" / "case.html"
    html_path.write_text(
        '<img src="figures/metrics/Runoff_fake.jpg" alt="fake">\n<img src="figures/metrics/gone.png" alt="gone">\n',
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING, logger="openbench.util.report"):
        standalone_path = _generator(case_dir)._generate_standalone_html_report(str(html_path), "case")

    standalone = Path(standalone_path).read_text(encoding="utf-8")
    assert _embedded_images(standalone) == {"fake": ("image/jpeg", b"fake jpg")}
    assert '<img src="figures/metrics/gone.png" alt="gone">' in standalone
    assert "without resizing" in caplog.text
    assert "missing figure" in caplog.text
