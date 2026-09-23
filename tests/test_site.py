import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("build_site", Path("scripts/build_site.py"))
site = importlib.util.module_from_spec(spec)
spec.loader.exec_module(site)


def test_committed_page_matches_a_fresh_build():
    # run `make site` after changing the template or any results file
    assert Path("site/index.html").read_text() == site.build()


def test_page_data_is_valid_json_for_the_browser():
    page = site.build()
    assert "NaN" not in page and "Infinity" not in page
