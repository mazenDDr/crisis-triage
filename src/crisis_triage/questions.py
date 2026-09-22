"""The typed questions Laya answers, per track, in every wording that T03 compares on dev.

Laya builds one sequence per question: instructions + options must fit in 192 tokens, so option
descriptions stay short (with 10+ options a long list cuts the instruction to 8 tokens;
`head_tokens` measures this). All variants of a message go in one `predict` call: questions
do not see each other, so this only saves time.
"""

from __future__ import annotations

# Short plain-language descriptions of every class the tracks use.
HUMANITARIAN = {
    "caution_and_advice": "warnings, advice, safety instructions",
    "displaced_people_and_evacuations": "people evacuated, displaced, in shelters",
    "displaced_and_evacuations": "people evacuated, displaced, in shelters",
    "infrastructure_and_utility_damage": "damaged buildings, roads, power, water supply",
    "infrastructure_and_utilities_damage": "damaged buildings, roads, power, water supply",
    "injured_or_dead_people": "people injured or killed",
    "missing_or_found_people": "people missing or found",
    "missing_and_found_people": "people missing or found",
    "not_humanitarian": "not about the disaster",
    "other_relevant_information": "other news about the disaster",
    "requests_or_urgent_needs": "someone asks for urgently needed help or supplies",
    "requests_or_needs": "someone asks for urgently needed help or supplies",
    "rescue_volunteering_or_donation_effort": "rescue work, volunteering, donations",
    "donation_and_volunteering": "volunteering, donations, fundraising",
    "sympathy_and_support": "prayers, sympathy, emotional support",
    "affected_individual": "people affected by the disaster",
    "personal_update": "personal news about oneself or family",
    "physical_landslide": "landslide or avalanche reports",
    "response_efforts": "government or agency response",
    "disease_related": "disease cases, outbreaks, health risks",
    "terrorism_related": "attacks, terrorism",
}

# CrisisBench merges sources that each used their own classes (deaths are `affected_individual`
# in some, `injured_or_dead_people` in others), so the fine classes are not one consistent
# question. Both tweet tracks are also scored on this shared coarse set.
COARSE = {
    "requests_or_needs": "someone asks for help, food, water or supplies",
    "people_affected": "people injured, killed, missing or affected",
    "infrastructure_damage": "damaged buildings, roads, power or water supply",
    "displaced_or_evacuated": "people evacuated, displaced or in shelters",
    "donations_and_rescue_work": "rescue work, volunteering, donations, aid response",
    "caution_and_advice": "warnings, advice, safety instructions",
    "sympathy_and_support": "prayers, sympathy, emotional support",
    "other_or_not_humanitarian": "other news, or not about the disaster",
}
TO_COARSE = {
    "requests_or_urgent_needs": "requests_or_needs",
    "requests_or_needs": "requests_or_needs",
    "injured_or_dead_people": "people_affected",
    "missing_or_found_people": "people_affected",
    "missing_and_found_people": "people_affected",
    "affected_individual": "people_affected",
    "infrastructure_and_utility_damage": "infrastructure_damage",
    "infrastructure_and_utilities_damage": "infrastructure_damage",
    "displaced_people_and_evacuations": "displaced_or_evacuated",
    "displaced_and_evacuations": "displaced_or_evacuated",
    "rescue_volunteering_or_donation_effort": "donations_and_rescue_work",
    "donation_and_volunteering": "donations_and_rescue_work",
    "response_efforts": "donations_and_rescue_work",
    "caution_and_advice": "caution_and_advice",
    "sympathy_and_support": "sympathy_and_support",
    "other_relevant_information": "other_or_not_humanitarian",
    "not_humanitarian": "other_or_not_humanitarian",
    "personal_update": "other_or_not_humanitarian",
    "physical_landslide": "other_or_not_humanitarian",
    "disease_related": "other_or_not_humanitarian",
    "terrorism_related": "other_or_not_humanitarian",
}

HAITI_NEEDS = {
    "rescue": "search and rescue, people trapped or missing",
    "medical": "injured or sick people, doctors, medicine",
    "water_food": "drinking water or food",
    "shelter": "tents, housing, a place to stay",
}

HUMSET_SECTORS = {
    "Health": "health care, disease, hospitals",
    "Protection": "violence, rights, safety of people",
    "Livelihoods": "jobs, income, markets",
    "Food Security": "access to food",
    "WASH": "water, sanitation, hygiene",
    "Education": "schools, learning",
    "Shelter": "housing, shelter, household items",
    "Nutrition": "malnutrition, feeding children",
    "Agriculture": "crops, livestock, farming",
    "Logistics": "transport, supply chains, access",
}

# Urgency wordings compared on dev; the score is read as an expected level in [0, 2].
URGENCY = {
    "urgency_plain": {
        "type": "score",
        "instructions": "How urgent is this message for emergency responders?",
        "criteria": ["not urgent", "needs help soon", "life in danger right now"],
    },
    "urgency_concrete": {
        "type": "score",
        "instructions": "Does someone in this message need emergency help?",
        "criteria": [
            "no one needs help",
            "people need supplies or services",
            "people are injured, trapped or missing",
        ],
    },
}

# Stand-in labels for urgency (no dataset labels it): the classes that describe a threat to life.
URGENT_CLASSES = {
    "haiti_sms": {"rescue", "medical"},
    "humaid": {"requests_or_urgent_needs", "injured_or_dead_people", "missing_or_found_people"},
    "crisisbench_ml": {"requests_or_needs", "injured_or_dead_people", "missing_and_found_people"},
}


def readable(label: str) -> str:
    return label.replace("_", " ")


def choice_question(labels: list[str], wording: str, descriptions: dict = HUMANITARIAN) -> dict:
    """One `choice` over humanitarian classes; option keys are readable names."""
    if wording == "names":
        criteria: list[str] | dict[str, str] = [readable(c) for c in labels]
    else:
        criteria = {readable(c): descriptions[c] for c in labels}
    return {
        "type": "choice",
        "instructions": "Which kind of information does this crisis message give?",
        "criteria": criteria,
    }


def noul_questions(labels: dict[str, str], template: str, wording: str) -> dict[str, dict]:
    """One `noul` per label; `{what}` in `template` becomes the name, or name + description."""
    out = {}
    for key, desc in labels.items():
        what = readable(key).lower() if wording == "short" else f"{readable(key).lower()} ({desc})"
        out[key] = {"type": "noul", "instructions": template.format(what=what)}
    return out


def haiti_variants() -> dict[str, dict]:
    """Question sets for the SMS needs. Keys are `<variant>/<question>`."""
    template = "Does the sender ask for {what}?"
    variants: dict[str, dict] = {}
    for wording in ("short", "described"):
        variants[f"noul_{wording}"] = noul_questions(HAITI_NEEDS, template, wording)
    variants["choice_described"] = {
        "need": {
            "type": "choice",
            "instructions": "What does the sender of this message need most?",
            "criteria": {**HAITI_NEEDS, "none": "no request for these needs"},
        }
    }
    return variants


def humset_variants() -> dict[str, dict]:
    template = "Is this humanitarian report excerpt about {what}?"
    return {
        f"noul_{w}": noul_questions(HUMSET_SECTORS, template, w) for w in ("short", "described")
    }


def class_variants(labels: list[str]) -> dict[str, dict]:
    """Fine variants over the track's own classes, coarse variants over COARSE."""
    out = {f"choice_{w}": {"label": choice_question(labels, w)} for w in ("names", "described")}
    for w in ("names", "described"):
        out[f"coarse_{w}"] = {"label": choice_question(list(COARSE), w, COARSE)}
    return out


def flatten(variants: dict[str, dict], extra: dict[str, dict] | None = None) -> dict[str, dict]:
    """All variants as one question dict for a single predict call, keys `<variant>/<question>`."""
    flat = {f"{v}/{q}": qdef for v, qs in variants.items() for q, qdef in qs.items()}
    for key, qdef in (extra or {}).items():
        flat[f"{key}/score"] = qdef
    return flat


def head_tokens(tok, qdef: dict) -> dict[str, int]:
    """Tokens the instruction and options take, and whether Laya would cut the instruction.

    Mirrors laya.common.build_sequence: options get up to 48 tokens each, the head budget is 192;
    when options leave fewer than 16 tokens they are cut, and the instruction keeps what is left
    (at least 8 tokens).
    """
    from laya.agent import Agent
    from laya.common import render_options

    q = Agent._to_internal(qdef)
    ins = len(tok(f"{q['t']} question: {q['ins']}", add_special_tokens=False)["input_ids"])
    opts = [
        1 + min(48, len(tok(" " + o, add_special_tokens=False)["input_ids"]))
        for o in render_options(q)
    ]
    budget = 192 - sum(opts)
    if budget < 16:
        per = max(4, (192 - 16) // max(1, len(opts)))
        opts = [min(o, per) for o in opts]
        budget = 192 - sum(opts)
    kept = min(ins, max(8, budget))
    return {"instruction": ins, "options": sum(opts), "instruction_kept": kept, "cut": kept < ins}


# Chosen on the dev sample (results/t03_dev.json, "decisions"): per track, the text column,
# the checkpoint (None = Laya's router) and the question variant. Ties keep the simpler or
# default option. Haiti is also run on the original text: translation is a pipeline step, and
# the human translation here is an upper bound for it.
CHOSEN = {
    "haiti_sms": {"text": "text_en", "model": "typed-decisions", "variant": "noul_described"},
    "haiti_sms_original": {"text": "text", "model": "multilingual", "variant": "noul_described"},
    "humaid": {"text": "text", "model": "typed-decisions", "variant": "coarse_described"},
    "crisisbench_ml": {"text": "text", "model": None, "variant": "coarse_described"},
    "humset": {"text": "text", "model": None, "variant": "noul_described"},
}
URGENCY_CHOSEN = "urgency_concrete"
