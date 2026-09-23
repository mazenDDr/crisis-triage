import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("export_demo", Path("scripts/export_demo.py"))
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def test_clean_removes_names_links_and_fixes_accents():
    assert demo.clean("RT @bob: help at @red_cross http://t.co/x now") == "help at @user now"
    assert demo.clean("se cayeron las casas http…") == "se cayeron las casas"
    assert demo.clean("murieron por caÃ­das") == "murieron por caídas"
    assert demo.clean("déjà vu") == "déjà vu"
    assert demo.clean("Â¿Lo dices? â€œhola…") == "¿Lo dices? “hola…"


def test_clean_masks_phone_numbers_and_emails_but_keeps_counts():
    assert demo.clean("Contact: Priya 9929004404 now") == "Contact: Priya [number] now"
    assert demo.clean("call +1 (555) 123-4567") == "call [number]"
    assert demo.clean("mail a.b@example.org") == "mail [email]"
    assert (
        demo.clean("44 dead, 2018 fire, 250,000 evacuated")
        == "44 dead, 2018 fire, 250,000 evacuated"
    )
