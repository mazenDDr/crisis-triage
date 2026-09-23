import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("build_readme", Path("scripts/build_readme.py"))
build_readme = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_readme)


def test_readme_and_svg_match_a_fresh_build():
    # run `make readme` after changing the template or any results file
    readme, svg = build_readme.build()
    assert Path("README.md").read_text() == readme
    assert Path("docs/assets/desk.svg").read_text() == svg


def test_svg_uses_no_web_fonts_or_scripts():
    _, svg = build_readme.build()
    assert "<script" not in svg and "@import" not in svg and "url(http" not in svg
