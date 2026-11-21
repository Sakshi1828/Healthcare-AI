"""Guardrailed LLM report drafting: the LLM writes, code verifies.

The language model never sees the image. It receives only the classifier's structured output
(findings above threshold, calibrated probabilities, triage level, uncertainty) and turns it into
readable report prose. Every draft then passes a deterministic verifier:

* no finding mentioned that is not above threshold (hallucination), including negated mentions
  like "no pneumothorax", which would claim a certainty the classifier does not have;
* every above-threshold finding is mentioned (omission);
* referral cases must disclose uncertainty and recommend radiologist review;
* no overclaiming language ("diagnosis", "confirmed", "definitely", ...).

A failing draft gets one repair attempt with the violations as feedback; after that the system
falls back to a deterministic template. The disclaimer and the raw model outputs are appended by
code, never by the LLM.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections import OrderedDict
from typing import Protocol

from .inference import DISCLAIMER

DISPLAY = {
    "atelectasis": "atelectasis",
    "cardiomegaly": "cardiomegaly",
    "effusion": "pleural effusion",
    "infiltration": "infiltration",
    "mass": "mass",
    "nodule": "nodule",
    "pneumonia": "pneumonia",
    "pneumothorax": "pneumothorax",
    "consolidation": "consolidation",
    "edema": "pulmonary edema",
    "emphysema": "emphysema",
    "fibrosis": "fibrosis",
    "pleural": "pleural thickening",
    "hernia": "hernia",
}

# How each finding can appear in prose. "pleural" alone is not a finding: "pleural effusion"
# must count as effusion, not as pleural thickening.
PATTERNS = {
    "atelectasis": r"atelecta(?:sis|ses|tic)",
    "cardiomegaly": r"cardiomegaly|(?:enlarged|enlargement of the) (?:cardiac|heart)"
    r"|cardiac (?:silhouette )?enlargement",
    "effusion": r"effusions?",
    "infiltration": r"infiltrat(?:e|es|ion|ions|ive)",
    "mass": r"\bmass(?:es)?\b",
    "nodule": r"nodul(?:e|es|ar)",
    "pneumonia": r"pneumonia|pneumonic",
    "pneumothorax": r"pneumothora(?:x|ces)",
    "consolidation": r"consolidat(?:ion|ions|ive|ed)",
    "edema": r"o?edema",
    "emphysema": r"emphysema(?:tous)?",
    "fibrosis": r"fibros(?:is|es)|fibrotic",
    "pleural": r"pleural thickening|thickening of the pleura",
    "hernia": r"hernia(?:s|tion)?",
}
OVERCLAIM = re.compile(
    r"\b(diagnos(?:is|es|ed|e)|confirm(?:s|ed)?|definite(?:ly)?|certain(?:ly)?|"
    r"rule[sd]? out|ruled out|consistent with a diagnosis)\b",
    re.I,
)
REFERRAL = re.compile(r"uncertain|radiologist", re.I)

SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"finding": {"type": "string"}, "sentence": {"type": "string"}},
                "required": ["finding", "sentence"],
                "additionalProperties": False,
            },
        },
        "impression": {"type": "string"},
        "recommendation": {"type": "string"},
    },
    "required": ["findings", "impression", "recommendation"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You draft short preliminary chest X-ray triage notes from the structured output \
of an image classifier. A radiologist reads every case; your note helps them prioritize.

You do not see the image. Write only what the classifier output supports:
- Write one finding entry for each listed finding, and no others. Use the finding's name as given.
- Never mention any other finding, not even to say it is absent: the classifier cannot rule \
anything out, so "no pneumothorax" would be an unsupported claim.
- Use hedged screening language ("suggests", "possible", "flagged for"). Never say diagnosis, \
confirmed, definite, certain, or ruled out.
- If the triage level is REFER_UNCERTAIN, the impression or recommendation must say the model \
is uncertain and that radiologist review is needed.
- If the triage level is URGENT, the recommendation should ask for prompt radiologist review.
- If there are no listed findings, write an empty findings list and say that nothing was \
flagged above threshold; do not describe the lungs as normal.
- Keep it brief: one sentence per finding, one or two sentences each for impression and \
recommendation. Plain clinical English, no markdown."""


def build_context(pred: dict) -> dict:
    """The only information the LLM receives."""
    findings = [
        {
            "finding": DISPLAY[f],
            "probability": round(pred["probabilities"][f], 3),
            "threshold": round(pred["thresholds"][f], 3),
            "urgent": f in pred.get("urgent_labels", []),
        }
        for f in sorted(pred["findings"], key=lambda f: -pred["probabilities"][f])
    ]
    return {
        "triage": pred["triage"],
        "uncertainty_bits": round(pred["uncertainty"], 3),
        "uncertainty_level": "high" if pred["triage"] == "REFER_UNCERTAIN" else "normal",
        "findings_above_threshold": findings,
    }


def mentioned(text: str) -> set[str]:
    return {k for k, p in PATTERNS.items() if re.search(p, text, re.I)}


def draft_text(draft: dict) -> str:
    parts = [f"{x.get('finding', '')}: {x.get('sentence', '')}" for x in draft.get("findings", [])]
    return "\n".join(parts + [draft.get("impression", ""), draft.get("recommendation", "")])


def verify(draft: dict, pred: dict) -> list[str]:
    """Returns a list of guardrail violations (empty = pass)."""
    text = draft_text(draft)
    allowed = set(pred["findings"])
    said = mentioned(text)
    v = [f"hallucinated: {DISPLAY[f]}" for f in sorted(said - allowed)]
    v += [f"omitted: {DISPLAY[f]}" for f in sorted(allowed - said)]
    if pred["triage"] == "REFER_UNCERTAIN" and not REFERRAL.search(
        draft.get("impression", "") + " " + draft.get("recommendation", "")
    ):
        v.append("missing uncertainty disclosure")
    v += [f"overclaim: '{m.group(0)}'" for m in OVERCLAIM.finditer(text)]
    return v


class Generator(Protocol):
    name: str

    def draft(self, context: dict, feedback: list[str] | None = None) -> tuple[dict, dict]:
        """Returns (draft, usage)."""
        ...


class TemplateGenerator:
    """Deterministic baseline; also the fallback when the LLM fails verification."""

    name = "template"

    def draft(self, context: dict, feedback: list[str] | None = None) -> tuple[dict, dict]:
        fs = context["findings_above_threshold"]
        findings = [
            {
                "finding": f["finding"],
                "sentence": f"Flagged for {f['finding']} (model probability "
                f"{f['probability']:.0%}, threshold {f['threshold']:.0%}).",
            }
            for f in fs
        ]
        triage = context["triage"]
        if triage == "URGENT":
            urgent = ", ".join(f["finding"] for f in fs if f["urgent"])
            impression = f"Possible urgent finding: {urgent}."
            rec = "Prioritize for prompt radiologist review."
        elif triage == "REFER_UNCERTAIN":
            impression = "The model is uncertain about this image."
            rec = "Route to a radiologist for full review; do not rely on the model output."
        elif fs:
            impression = "Possible non-urgent finding(s): " + ", ".join(f["finding"] for f in fs)
            impression += "."
            rec = "Routine radiologist review."
        else:
            impression = "Nothing was flagged above threshold by the model."
            rec = "Routine radiologist review; a radiologist still reads every case."
        return {"findings": findings, "impression": impression, "recommendation": rec}, {}


class ClaudeGenerator:
    """Drafts with Claude using structured outputs (JSON schema), so the response always parses."""

    name = "claude"

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium", client=None):
        import anthropic

        self.client = client or anthropic.Anthropic(max_retries=2, timeout=60.0)
        self.model = model
        self.effort = effort

    def draft(self, context: dict, feedback: list[str] | None = None) -> tuple[dict, dict]:
        content = "Classifier output:\n" + json.dumps(context, indent=2, sort_keys=True)
        if feedback:
            content += (
                "\n\nYour previous draft broke these rules; write a corrected draft:\n- "
                + "\n- ".join(feedback)
            )
        resp = self.client.beta.messages.create(
            model=self.model,
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": SCHEMA},
            },
            # On a safety-classifier decline, the API reruns the request on a fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        usage = {
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
            "model": resp.model,
        }
        if resp.stop_reason in ("refusal", "max_tokens"):
            raise RuntimeError(f"LLM stopped with {resp.stop_reason}")
        text = next(b.text for b in resp.content if b.type == "text")
        return json.loads(text), usage


def render(draft: dict, pred: dict) -> str:
    lines = [f"TRIAGE: {pred['triage'].replace('_', ' ')}", "", "FINDINGS:"]
    lines += [f"- {x['sentence']}" for x in draft["findings"]] or ["- None above threshold."]
    lines += [
        "",
        f"IMPRESSION: {draft['impression']}",
        f"RECOMMENDATION: {draft['recommendation']}",
    ]
    lines += ["", f"Model uncertainty: {pred['uncertainty']:.2f} bits.", DISCLAIMER]
    return "\n".join(lines)


def write_report(pred: dict, generator: Generator, max_repairs: int = 1) -> dict:
    """Draft, verify, repair once, then fall back to the template if still failing."""
    context = build_context(pred)
    usage, history, feedback, draft = [], [], None, None
    t0 = time.perf_counter()
    for _ in range(1 + max_repairs):
        try:
            draft, u = generator.draft(context, feedback)
        except Exception as err:  # API error, refusal, malformed JSON: fall back, never crash
            history.append([f"generator error: {type(err).__name__}: {err}"])
            break
        usage.append(u)
        violations = verify(draft, pred)
        history.append(violations)
        if not violations:
            source = generator.name if len(history) == 1 else f"{generator.name}_repaired"
            break
        feedback = violations
    if draft is None or history[-1]:
        draft, _ = TemplateGenerator().draft(context)
        source = "template" if generator.name == "template" else "template_fallback"
    return {
        "text": render(draft, pred),
        "draft": draft,
        "source": source,
        "verified": not verify(draft, pred),
        "violations_per_attempt": history,
        "usage": usage,
        "latency_s": round(time.perf_counter() - t0, 3),
    }


class ReportService:
    """Serving wrapper: caches identical contexts (demo samples get clicked repeatedly) and caps
    LLM calls per hour so a public demo cannot run up an API bill; over the cap it uses the
    template."""

    def __init__(self, generator: Generator, max_llm_calls_per_hour: int = 60, cache_size=256):
        self.generator = generator
        self.cap = max_llm_calls_per_hour
        self.calls: list[float] = []
        self.cache: OrderedDict[str, dict] = OrderedDict()
        self.cache_size = cache_size
        self.lock = threading.Lock()

    def __call__(self, pred: dict) -> dict:
        key = hashlib.sha256(json.dumps(build_context(pred), sort_keys=True).encode()).hexdigest()
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return {**self.cache[key], "cached": True}
            now = time.time()
            self.calls = [t for t in self.calls if now - t < 3600]
            use_llm = self.generator.name != "template" and len(self.calls) < self.cap
            if use_llm:
                self.calls.append(now)
        report = write_report(pred, self.generator if use_llm else TemplateGenerator())
        if not use_llm and self.generator.name != "template":
            report["source"] = "template_rate_limited"
        with self.lock:
            self.cache[key] = report
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return {**report, "cached": False}
