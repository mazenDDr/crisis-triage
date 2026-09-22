import pandas as pd

from crisis_triage.data import (
    drop_test_copies,
    event_year,
    guess_creole_or_french,
    norm,
    time_split,
)


def test_event_year_finds_prefix_and_suffix_years():
    assert event_year("2014_chile_earthquake_esp") == 2014
    assert event_year("kerala_floods_2018") == 2018
    assert event_year("2014-2015_worldwide_landslides") == 2014
    assert event_year("disaster_events") is None


def test_time_split_boundary_is_inclusive():
    assert time_split(2017, 2017) == "dev"
    assert time_split(2018, 2017) == "test"
    assert time_split(None, 2017) is None


def test_norm_ignores_retweet_prefix_links_and_case():
    assert norm("RT @bob: Help us NOW! http://t.co/x") == norm("help us now")


def test_creole_vs_french():
    assert guess_creole_or_french("Mwen bezwen manje ak dlo nan kay la") == "ht"
    assert guess_creole_or_french("Nous avons besoin de tentes pour les enfants") == "fr"
    assert guess_creole_or_french("Eske pwoteksyon sivil la bay yon bilan") == "unk"
    assert guess_creole_or_french("please send water to Leogane") == "unk"


def test_drop_test_copies_removes_leaks_and_duplicates():
    df = pd.DataFrame(
        {
            "text": ["help us", "RT @a: help us", "new text", "new text", "other"],
            "split": ["dev", "test", "test", "test", "test"],
        }
    )
    kept = drop_test_copies(df)["text"].tolist()
    assert kept == ["help us", "new text", "other"]
