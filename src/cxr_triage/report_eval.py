"""Evaluate guardrailed report drafting on held-out test cases.

    python -m cxr_triage.report_eval runs/<run> --backends template,gemini --n 100

For each backend it measures how often the first draft passes the verifier, the rate of each
violation type (hallucinated finding, omitted finding, missing uncertainty disclosure,
overclaiming), how often the repair step or the template fallback was needed, latency, tokens,
and estimated cost. Cases are stratified across triage levels so rare REFER/URGENT cases are
represented. Writes report_eval.json and report_eval.md into the run directory.

    python -m cxr_triage.report_eval runs/<run> --backends gemini --bundle-samples

pre-generates reports for the demo's bundled sample images and stores them in bundle.pt, so the
public demo shows LLM-drafted reports for samples without calling any API at serving time.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .inference import make_prediction
from .report import make_generator, write_report

# USD per million tokens (input, output). Gemini free tier costs nothing within its limits.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}
TRIAGE_LEVELS = ["URGENT", "REFER_UNCERTAIN", "NON_URGENT_FINDING", "NO_FINDING_DETECTED"]


def load_bundle(run: Path) -> dict:
    return torch.load(run / "bundle.pt", map_location="cpu", weights_only=False)


def load_cases(run: Path, n: int, seed: int = 0) -> list[dict]:
    """Predictions for n test cases, stratified as evenly as possible across triage levels."""
    b = load_bundle(run)
    data = np.load(run / "test_predictions.npz")
    preds = [
        make_prediction(
            p,
            b["labels"],
            np.asarray(b["thresholds"]),
            set(b["urgent_labels"]),
            b["refer_uncertainty"],
        )
        for p in data["probs"]
    ]
    rng = np.random.default_rng(seed)
    groups = {t: [i for i, p in enumerate(preds) if p["triage"] == t] for t in TRIAGE_LEVELS}
    for g in groups.values():
        rng.shuffle(g)
    chosen: list[int] = []
    while len(chosen) < min(n, len(preds)):
        for t in TRIAGE_LEVELS:  # round-robin over levels until n cases are picked
            if groups[t] and len(chosen) < n:
                chosen.append(groups[t].pop())
    return [{**preds[i], "case_id": int(i)} for i in chosen]


def _rate(xs) -> float:
    xs = list(xs)
    return float(np.mean(xs)) if xs else float("nan")


def summarize(name: str, results: list[dict], cases: list[dict]) -> dict:
    first = [r["violations_per_attempt"][0] if r["violations_per_attempt"] else [] for r in results]
    usage = [u for r in results for u in r["usage"]]
    tok_in = sum(u.get("input_tokens") or 0 for u in usage)
    tok_out = sum(u.get("output_tokens") or 0 for u in usage)
    model = next((u.get("model") for u in usage if u.get("model")), None)
    price = next((v for k, v in PRICES.items() if model and model.startswith(k)), None)
    refer = [i for i, c in enumerate(cases) if c["triage"] == "REFER_UNCERTAIN"]
    lat = [r["latency_s"] for r in results]
    return {
        "backend": name,
        "model": model,
        "n": len(results),
        "first_draft_pass_rate": _rate(not v for v in first),
        "first_draft_hallucination_rate": _rate(
            any(x.startswith("hallucinated") for x in v) for v in first
        ),
        "first_draft_omission_rate": _rate(any(x.startswith("omitted") for x in v) for v in first),
        "first_draft_overclaim_rate": _rate(
            any(x.startswith("overclaim") for x in v) for v in first
        ),
        "refer_disclosure_miss_rate": _rate(
            any(x.startswith("missing uncertainty") for x in first[i]) for i in refer
        ),
        "generator_error_rate": _rate(
            any(x.startswith("generator error") for x in v) for v in first
        ),
        "repaired_rate": _rate(r["source"].endswith("_repaired") for r in results),
        "template_fallback_rate": _rate(r["source"] == "template_fallback" for r in results),
        "final_verified_rate": _rate(r["verified"] for r in results),
        "sources": dict(Counter(r["source"] for r in results)),
        "latency_mean_s": float(np.mean(lat)),
        "latency_p95_s": float(np.quantile(lat, 0.95)),
        "input_tokens": tok_in,
        "output_tokens": tok_out,
        "est_cost_usd": (tok_in * price[0] + tok_out * price[1]) / 1e6 if price else 0.0,
    }


def run_backend(backend: str, cases: list[dict], rpm: float = 0) -> tuple[dict, list[dict]]:
    gen = make_generator(backend)
    results = []
    for i, case in enumerate(cases):
        t0 = time.time()
        results.append(write_report(case, gen))
        if rpm and gen.name != "template":  # stay under free-tier per-minute limits
            time.sleep(max(0.0, 60.0 / rpm * len(results[-1]["usage"] or [1]) - (time.time() - t0)))
        if (i + 1) % 10 == 0:
            print(f"  [{gen.name}] {i + 1}/{len(cases)}")
    return summarize(gen.name, results, cases), results


def to_markdown(summaries: list[dict], examples: dict[str, list[dict]]) -> str:
    pct = lambda x: "n/a" if x != x else f"{x:.0%}"  # noqa: E731  (NaN-safe)
    rows = [
        "| Backend | Model | n | 1st-draft pass | Hallucination | Omission | Overclaim | "
        "REFER disclosure miss | Repaired | Template fallback | Final verified | "
        "Latency (mean) | Est. cost |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        rows.append(
            f"| {s['backend']} | {s['model'] or '-'} | {s['n']} | "
            f"{pct(s['first_draft_pass_rate'])} | "
            f"{pct(s['first_draft_hallucination_rate'])} | {pct(s['first_draft_omission_rate'])} | "
            f"{pct(s['first_draft_overclaim_rate'])} | {pct(s['refer_disclosure_miss_rate'])} | "
            f"{pct(s['repaired_rate'])} | {pct(s['template_fallback_rate'])} | "
            f"{pct(s['final_verified_rate'])} | {s['latency_mean_s']:.2f}s | "
            f"${s['est_cost_usd']:.3f} |"
        )
    out = ["# Report drafting evaluation", "", *rows, ""]
    for name, exs in examples.items():
        out += [f"## Examples: {name}", ""]
        for r in exs:
            out += [
                f"Source: `{r['source']}`, violations per attempt: `{r['violations_per_attempt']}`",
                "",
                "```",
                r["text"],
                "```",
                "",
            ]
    return "\n".join(out)


def bundle_samples(run: Path, backend: str) -> None:
    from PIL import Image

    from .inference import Predictor

    pred = Predictor(run / "bundle.pt")
    gen = make_generator(backend)
    reports = [write_report(pred.predict(Image.fromarray(s["image"])), gen) for s in pred.samples]
    b = load_bundle(run)
    b["sample_reports"] = reports
    torch.save(b, run / "bundle.pt")
    print(f"stored {len(reports)} sample reports ({[r['source'] for r in reports]})")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument(
        "--backends",
        default="template,auto",
        help="comma-separated: template, gemini, claude, auto",
    )
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--rpm", type=float, default=10, help="max LLM requests/minute (0 = no limit)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--bundle-samples",
        action="store_true",
        help="pre-generate demo sample reports into bundle.pt instead of evaluating",
    )
    args = ap.parse_args(argv)
    run = Path(args.run_dir)
    backends = args.backends.split(",")
    if args.bundle_samples:
        bundle_samples(run, backends[-1])
        return
    cases = load_cases(run, args.n, args.seed)
    print(f"{len(cases)} cases: {dict(Counter(c['triage'] for c in cases))}")
    summaries, examples, detail = [], {}, {}
    seen = set()
    for backend in backends:
        name = make_generator(backend).name
        if name in seen:  # e.g. "auto" resolved to the template with no API key set
            continue
        seen.add(name)
        summary, results = run_backend(backend, cases, args.rpm)
        summaries.append(summary)
        failing = [r for r in results if r["violations_per_attempt"][:1] != [[]]]
        examples[summary["backend"]] = (failing[:2] + results)[:3]
        detail[summary["backend"]] = [
            {"case_id": c["case_id"], "triage": c["triage"], **r}
            for c, r in zip(cases, results, strict=True)
        ]
    (run / "report_eval.json").write_text(
        json.dumps({"summaries": summaries, "cases": detail}, indent=2, default=str)
    )
    md = to_markdown(summaries, examples)
    (run / "report_eval.md").write_text(md)
    print(md.split("## Examples")[0])


if __name__ == "__main__":
    main()
