#!/usr/bin/env python3
"""Render the paper's figure PDFs into the project-page assets.

Usage:
    python figures/extract_paper_figures.py /path/to/paper-repo/images
"""

from pathlib import Path
import subprocess
import sys
import tempfile

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
MAX_WIDTH = 2400

# paper figure PDF -> project-page asset name
FIGURES = {
    "overview": "overview",
    "framework": "framework",
    "composition_scenario": "scenarios",
    "composition_specialization": "personas",
    "mode_distribution": "modes",
    "state_prompting_ablation": "pilot",
    "scaling_in_domain_1": "scaling",
    "scaling_zero_shot_merged": "transfer",
}


def render(source: Path, output: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        stem = Path(tmp) / output.stem
        subprocess.run(
            ["pdftocairo", "-png", "-r", "300", "-singlefile", str(source), str(stem)],
            check=True,
        )
        image = Image.open(stem.with_suffix(".png")).convert("RGB")

    width, height = image.size
    if width > MAX_WIDTH:
        image = image.resize((MAX_WIDTH, round(height * MAX_WIDTH / width)), Image.LANCZOS)
    image.save(output, optimize=True)


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    images = Path(sys.argv[1]).expanduser().resolve()

    for source_stem, asset_stem in FIGURES.items():
        source = images / f"{source_stem}.pdf"
        if not source.exists():
            sys.exit(f"missing figure: {source}")
        output = ASSETS / f"{asset_stem}.png"
        render(source, output)
        print(f"{source.name} -> {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
