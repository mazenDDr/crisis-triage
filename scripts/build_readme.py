"""Build README.md and docs/assets/desk.svg from docs/README-template.md and results/*.json.

Every number in the README tables and every word on the animated slip comes from the
committed results, so the README cannot drift from them (tests/test_readme.py checks it).
"""

import json
import textwrap
from html import escape
from pathlib import Path

RESULTS = Path("results")
SYSTEMS = [
    ("laya-ft", "**Laya, fine-tuned**"),
    ("e5lr", "e5 + LR (trained)"),
    ("qwen3-4b", "Qwen3-4B"),
    ("gemma3-4b", "Gemma-3-4B"),
    ("laya", "Laya, out of the box"),
]
RUNS = {
    "haiti_original": "Haiti SMS, Creole/French original",
    "haiti_mt": "Haiti SMS, NLLB translation",
    "haiti_human": "Haiti SMS, human translation",
    "humaid": "HumAID tweets, 2018–19 disasters",
    "crisisbench_ml": "CrisisBench tweets, es/fr/it/pt/tl",
    "humset": "HumSet report excerpts",
}


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def pct(x, d=0) -> str:
    return "—" if x is None or x != x else f"{100 * x:.{d}f}%"


def ci(c: dict) -> str:
    return f"{c['value']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}]"


def desk_table(scores, gating) -> str:
    g, res, speed = gating["runs"]["humaid"], scores["results"]["humaid"], scores["speed"]
    ft = load("t07_finetune.json")["timing_runs"]
    ft_ms = [r["runs"]["humaid"]["p50_ms"] for r in ft]
    rows = [
        "| | Category score (macro-F1) | Sent on its own at the 95% target | …right among those | Trust its own “≥ 95% sure” | …right among those | Time per message |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, label in SYSTEMS:
        a = g["systems"][key]["at"]["0.95"]
        ms = (
            f"{min(ft_ms):.0f}–{max(ft_ms):.0f} ms"
            if key == "laya-ft"
            else f"{speed[key]['humaid']['p50_ms']:.0f} ms"
        )
        rows.append(
            f"| {label} | {ci(res[key]['macro_f1'])} | {pct(a['coverage_point'])} | "
            f"{pct(a['accuracy_covered'], 1) if a['coverage_point'] else '—'} | "
            f"{pct(a['naive']['coverage'])} | {pct(a['naive']['accuracy_covered'], 1) if a['naive']['coverage'] else '—'} | {ms} |"
        )
    return "\n".join(rows)


def all_runs_table(scores) -> str:
    res = scores["results"]
    head = "| Test set | n | Metric | " + " | ".join(label for _, label in SYSTEMS) + " |"
    rows = [head, "|" + "---|" * (3 + len(SYSTEMS))]
    for run, name in RUNS.items():
        systems = res[run]
        first = next(iter(systems.values()))
        metric = "mean AUC" if "mean_auc" in first else "macro-F1"
        cells = []
        for key, _ in SYSTEMS:
            if key not in systems:
                cells.append("—")
                continue
            v = (systems[key].get("mean_auc") or systems[key]["macro_f1"])["value"]
            best = max((s.get("mean_auc") or s["macro_f1"])["value"] for s in systems.values())
            cells.append(f"**{v:.3f}**" if v == best else f"{v:.3f}")
        rows.append(f"| {name} | {first['n']:,} | {metric} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def creole_table(scores) -> str:
    res = scores["results"]
    rows = [
        ("Laya, out of the box, reads the Creole", res["haiti_original"]["laya"]["mean_auc"]),
        ("Laya, after an NLLB-600M translation to English", res["haiti_mt"]["laya"]["mean_auc"]),
        ("Laya, fine-tuned, reads the Creole", res["haiti_original"]["laya-ft"]["mean_auc"]),
        ("Laya with a human translator (upper bound)", res["haiti_human"]["laya"]["mean_auc"]),
    ]
    out = ["| Haiti SMS, 996 test messages | Mean AUC over 4 needs [95% CI] |", "|---|---|"]
    out += [f"| {name} | {ci(c)} |" for name, c in rows]
    return "\n".join(out)


def facts(scores, gating, ft) -> dict:
    g = gating["runs"]["humaid"]["systems"]
    res = scores["results"]
    q = g["qwen3-4b"]["at"]["0.95"]["naive"]
    lf = g["laya-ft"]["at"]["0.95"]
    paired = scores["paired"]["humaid"]["laya-ft - e5lr"]
    return {
        "ft_sent": pct(lf["coverage_point"]),
        "ft_right": pct(lf["accuracy_covered"], 1),
        "ft_person": pct(1 - lf["coverage_point"]),
        "qwen_sent": pct(q["coverage"]),
        "qwen_right": pct(q["accuracy_covered"]),
        "qwen_wrong_per_100": f"{100 * q['coverage'] * (1 - q['accuracy_covered']):.0f}",
        "ft_minutes": f"{round(ft['epochs'][-1]['seconds'] / 60)}",
        "ft_examples": f"{ft['train_examples']:,}",
        "zs_f1": f"{res['humaid']['laya']['macro_f1']['value']:.2f}",
        "ft_f1": f"{res['humaid']['laya-ft']['macro_f1']['value']:.2f}",
        "ft_vs_e5": f"{paired['value']:+.3f} [{paired['lo']:+.3f}, {paired['hi']:+.3f}]",
        "haiti_zs": f"{res['haiti_original']['laya']['mean_auc']['value']:.2f}",
        "haiti_mt": f"{res['haiti_mt']['laya']['mean_auc']['value']:.2f}",
        "haiti_ft": f"{res['haiti_original']['laya-ft']['mean_auc']['value']:.2f}",
        "cb_reached": pct(
            gating["runs"]["crisisbench_ml"]["systems"]["laya-ft"]["at"]["0.95"][
                "accuracy_covered"
            ],
            1,
        ),
        "n_humaid": f"{gating['runs']['humaid']['n_test']:,}",
    }


# ---------------------------------------------------------------- the animated desk (SVG)

PAPER, SLIP, INK, FAINT, RULE = "#f2eee4", "#fffdf7", "#16140f", "#7d7667", "#d6cfbf"
SENT, PERSON, WRONG, METER = "#13733b", "#2446c8", "#d2361e", "#ddd5c4"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
CONDENSED = (
    "'Avenir Next Condensed', 'Arial Narrow', 'Roboto Condensed', sans-serif-condensed, sans-serif"
)


def opening_slips(demo, gating):
    """The first 'sent' and the first 'for a person' tweet of the page's sample (same as the page)."""
    cut = gating["runs"]["humaid"]["systems"]["laya-ft"]["at"]["0.95"]["dev_cut"]
    msgs = demo["tracks"]["humaid"]["messages"]
    sent = next(m for m in msgs if m["systems"]["laya-ft"][1] >= cut)
    person = next(m for m in msgs if m["systems"]["laya-ft"][1] < cut)
    return cut, [(sent, True), (person, False)]


def slip_group(m, sent: bool, cut: float, idx: int) -> str:
    conf = m["systems"]["laya-ft"][1]
    lines = textwrap.wrap(m["text"], 40)[:3]
    if len(textwrap.wrap(m["text"], 40)) > 3:
        lines[-1] = lines[-1][:37] + "…"
    text = "".join(
        f'<text x="20" y="{58 + 20 * i}" font-family="{MONO}" font-size="14" fill="{INK}">{escape(t)}</text>'
        for i, t in enumerate(lines)
    )
    stamp_color = SENT if sent else PERSON
    label = f"SENT · {round(100 * conf)}%" if sent else f"PERSON · {round(100 * conf)}%"
    event = escape(f"{m['lang'].upper()} · {m['event'].replace('_', ' ')}")
    return f"""
  <g class="slip s{idx}">
    <rect width="380" height="170" rx="4" fill="{SLIP}" stroke="{RULE}"/>
    <text x="20" y="30" font-family="{MONO}" font-size="11" letter-spacing="1" fill="{FAINT}">{event}</text>
    {text}
    <rect x="20" y="128" width="340" height="10" rx="2" fill="{METER}"/>
    <rect class="fill f{idx}" x="20" y="128" width="{340 * conf:.1f}" height="10" rx="2" fill="{INK}"/>
    <rect x="{20 + 340 * cut - 1.5:.1f}" y="122" width="3" height="22" fill="{WRONG}"/>
    <text x="20" y="158" font-family="{MONO}" font-size="11" fill="{FAINT}">how sure: {round(100 * conf)}% · safety line: {round(100 * cut)}%</text>
    <g class="stamp t{idx}" filter="url(#ink)">
      <rect x="-4" y="-22" width="{150 if sent else 180}" height="40" rx="6" fill="{SLIP}" stroke="{stamp_color}" stroke-width="4"/>
      <text x="{71 if sent else 86}" y="8" text-anchor="middle" font-family="{CONDENSED}" font-weight="800" font-size="24" letter-spacing="1" fill="{stamp_color}">{label}</text>
    </g>
  </g>"""


def desk_svg(demo, gating) -> str:
    cut, slips = opening_slips(demo, gating)
    groups = "".join(slip_group(m, s, cut, i) for i, (m, s) in enumerate(slips))
    # 10 s loop: slip 0 (sent) then slip 1 (for a person), then a long hold on the finished trays.
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 350" width="900" height="350" role="img" aria-label="A real test tweet is read, the model is sure, and it is stamped SENT; the next one is below the safety line and stamped FOR A PERSON.">
  <title>Sure → sent. Unsure → a person.</title>
  <defs>
    <filter id="ink" x="-5%" y="-10%" width="110%" height="120%">
      <feTurbulence type="fractalNoise" baseFrequency="0.8" numOctaves="2" seed="4" result="warp"/>
      <feDisplacementMap in="SourceGraphic" in2="warp" scale="2.2" result="rough"/>
      <feTurbulence type="fractalNoise" baseFrequency="1.9" numOctaves="1" seed="9" result="grain"/>
      <feColorMatrix in="grain" type="matrix" values="0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 -1.1 1.35" result="holes"/>
      <feComposite in="rough" in2="holes" operator="in"/>
    </filter>
  </defs>
  <style>
    .slip {{ opacity: 0; transform-box: fill-box; }}
    .s0 {{ animation: slip0 10s infinite; }}
    .s1 {{ animation: slip1 10s infinite; }}
    .fill {{ transform-box: fill-box; transform-origin: left center; }}
    .f0 {{ animation: fill0 10s infinite; }}
    .f1 {{ animation: fill1 10s infinite; }}
    .stamp {{ transform-box: fill-box; transform-origin: center; opacity: 0; }}
    .t0 {{ animation: stamp0 10s infinite; }}
    .t1 {{ animation: stamp1 10s infinite; }}
    .mini0 {{ animation: mini0 10s infinite; }}
    .mini1 {{ animation: mini1 10s infinite; }}
    .c0 {{ animation: mini0 10s infinite; }}
    .c1 {{ animation: mini1 10s infinite; }}
    @keyframes slip0 {{ 0% {{ opacity: 0; transform: translate(260px, 112px); }} 4%, 30% {{ opacity: 1; transform: translate(260px, 132px); }} 38%, 100% {{ opacity: 0; transform: translate(40px, 200px) scale(0.35); }} }}
    @keyframes slip1 {{ 0%, 40% {{ opacity: 0; transform: translate(260px, 112px); }} 44%, 70% {{ opacity: 1; transform: translate(260px, 132px); }} 78%, 100% {{ opacity: 0; transform: translate(700px, 200px) scale(0.35); }} }}
    @keyframes fill0 {{ 0%, 6% {{ transform: scaleX(0); }} 16%, 100% {{ transform: scaleX(1); }} }}
    @keyframes fill1 {{ 0%, 46% {{ transform: scaleX(0); }} 56%, 100% {{ transform: scaleX(1); }} }}
    @keyframes stamp0 {{ 0%, 19% {{ opacity: 0; transform: translate(222px, -8px) rotate(-7deg) scale(2.2); }} 22%, 100% {{ opacity: 1; transform: translate(222px, -8px) rotate(-7deg) scale(1); }} }}
    @keyframes stamp1 {{ 0%, 59% {{ opacity: 0; transform: translate(192px, -8px) rotate(-7deg) scale(2.2); }} 62%, 100% {{ opacity: 1; transform: translate(192px, -8px) rotate(-7deg) scale(1); }} }}
    @keyframes mini0 {{ 0%, 37% {{ opacity: 0; }} 40%, 100% {{ opacity: 1; }} }}
    @keyframes mini1 {{ 0%, 77% {{ opacity: 0; }} 80%, 100% {{ opacity: 1; }} }}
    @media (prefers-reduced-motion: reduce) {{
      .slip, .fill, .stamp, .mini0, .mini1, .c0, .c1 {{ animation: none; }}
      .s1 {{ opacity: 1; transform: translate(260px, 132px); }}
      .t1 {{ opacity: 1; transform: translate(192px, -8px) rotate(-7deg); }}
    }}
  </style>
  <rect width="900" height="350" rx="10" fill="{PAPER}"/>
  <text x="30" y="46" font-family="{CONDENSED}" font-weight="800" font-size="34" fill="{INK}">SURE → <tspan fill="{SENT}">SENT.</tspan></text>
  <text x="30" y="84" font-family="{CONDENSED}" font-weight="800" font-size="34" fill="{INK}">UNSURE → <tspan fill="{PERSON}">A PERSON.</tspan></text>
  <g transform="translate(30 150)">
    <rect width="200" height="150" rx="4" fill="#e8e2d4" stroke="{INK}" stroke-width="2"/>
    <text x="14" y="30" font-family="{CONDENSED}" font-weight="800" font-size="20" fill="{SENT}">SENT</text>
    <g class="mini0" transform="rotate(-2 100 72)"><rect x="16" y="52" width="168" height="40" rx="3" fill="{SLIP}" stroke="{RULE}"/><rect x="26" y="62" width="120" height="4" rx="2" fill="{RULE}"/><rect x="26" y="72" width="140" height="4" rx="2" fill="{RULE}"/><rect x="26" y="82" width="90" height="4" rx="2" fill="{RULE}"/></g>
    <text class="c0" x="170" y="34" text-anchor="end" font-family="{CONDENSED}" font-weight="800" font-size="30" fill="{SENT}">1</text>
  </g>
  <g transform="translate(670 150)">
    <rect width="200" height="150" rx="4" fill="#e8e2d4" stroke="{INK}" stroke-width="2"/>
    <text x="14" y="30" font-family="{CONDENSED}" font-weight="800" font-size="20" fill="{PERSON}">FOR A PERSON</text>
    <g class="mini1" transform="rotate(2 100 72)"><rect x="16" y="52" width="168" height="40" rx="3" fill="{SLIP}" stroke="{RULE}"/><rect x="26" y="62" width="120" height="4" rx="2" fill="{RULE}"/><rect x="26" y="72" width="140" height="4" rx="2" fill="{RULE}"/><rect x="26" y="82" width="90" height="4" rx="2" fill="{RULE}"/></g>
    <text class="c1" x="170" y="34" text-anchor="end" font-family="{CONDENSED}" font-weight="800" font-size="30" fill="{PERSON}">1</text>
  </g>{groups}
</svg>
"""


def build() -> tuple[str, str]:
    scores, gating, ft = load("t05_test.json"), load("t06_gating.json"), load("t07_finetune.json")
    demo = load("demo_messages.json")
    fill = {
        "desk_table": desk_table(scores, gating),
        "all_runs_table": all_runs_table(scores),
        "creole_table": creole_table(scores),
        **facts(scores, gating, ft),
    }
    readme = Path("docs/README-template.md").read_text()
    for key, value in fill.items():
        readme = readme.replace("{{" + key + "}}", value)
    assert "{{" not in readme, "unfilled placeholder in the README template"
    return readme, desk_svg(demo, gating)


def main():
    readme, svg = build()
    Path("README.md").write_text(readme)
    Path("docs/assets").mkdir(parents=True, exist_ok=True)
    Path("docs/assets/desk.svg").write_text(svg)
    print("wrote README.md and docs/assets/desk.svg")


if __name__ == "__main__":
    main()
