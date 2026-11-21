import json
from types import SimpleNamespace

import numpy as np
import pytest

from cxr_triage import LABELS
from cxr_triage.inference import make_prediction
from cxr_triage.report import (
    ClaudeGenerator,
    GeminiGenerator,
    ReportService,
    TemplateGenerator,
    make_generator,
    mentioned,
    verify,
    write_report,
)

URGENT = ["pneumothorax", "effusion", "pneumonia", "edema", "consolidation"]


def pred(findings, triage="URGENT", unc=0.2):
    probs = {name: (0.8 if name in findings else 0.01) for name in LABELS}
    return {
        "triage": triage,
        "findings": findings,
        "uncertainty": unc,
        "probabilities": probs,
        "thresholds": dict.fromkeys(LABELS, 0.3),
        "urgent_labels": URGENT,
    }


def draft(sentences, impression="Possible finding.", rec="Prompt radiologist review."):
    return {
        "findings": [{"finding": f, "sentence": s} for f, s in sentences],
        "impression": impression,
        "recommendation": rec,
    }


# ---------- verifier ----------


def test_pleural_effusion_is_effusion_not_pleural_thickening():
    assert mentioned("Small left pleural effusion.") == {"effusion"}
    assert mentioned("Apical pleural thickening.") == {"pleural"}


def test_verifier_flags_negated_mentions_omissions_and_overclaims():
    p = pred(["effusion"])
    d = draft([("effusion", "Possible pleural effusion.")], impression="No pneumothorax.")
    assert verify(d, p) == ["hallucinated: pneumothorax"]
    assert "omitted: pleural effusion" in verify(draft([]), p)
    d = draft([("effusion", "Pleural effusion, diagnosis confirmed.")])
    assert {v for v in verify(d, p) if v.startswith("overclaim")} == {
        "overclaim: 'diagnosis'",
        "overclaim: 'confirmed'",
    }


def test_uncertain_is_not_an_overclaim_and_referral_needs_disclosure():
    p = pred([], triage="REFER_UNCERTAIN", unc=0.9)
    assert verify(draft([], "Nothing flagged.", "Routine review."), p) == [
        "missing uncertainty disclosure"
    ]
    assert verify(draft([], "The model is uncertain.", "Radiologist review."), p) == []


def test_template_always_passes_the_verifier():
    rng = np.random.default_rng(0)
    for _ in range(300):
        probs = rng.random(len(LABELS)) ** 3
        p = make_prediction(probs, LABELS, np.full(len(LABELS), 0.5), set(URGENT), 0.5)
        report = write_report(p, TemplateGenerator())
        assert report["verified"] and report["source"] == "template", report


# ---------- LLM backends (fake clients, no network) ----------


class FakeClaude:
    def __init__(self, drafts, stop_reason="end_turn"):
        self.drafts, self.stop_reason, self.calls = list(drafts), stop_reason, []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.calls.append(kw)
        text = json.dumps(self.drafts.pop(0))
        return SimpleNamespace(
            usage=SimpleNamespace(input_tokens=500, output_tokens=300),
            model=kw["model"],
            stop_reason=self.stop_reason,
            content=[SimpleNamespace(type="text", text=text)],
        )


class FakeGemini:
    def __init__(self, drafts):
        self.drafts, self.calls = list(drafts), []
        self.models = SimpleNamespace(generate_content=self.generate_content)

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return SimpleNamespace(
            text=json.dumps(self.drafts.pop(0)),
            usage_metadata=SimpleNamespace(prompt_token_count=400, candidates_token_count=150),
        )


GOOD = draft([("pleural effusion", "Findings suggest a possible pleural effusion.")])
BAD = draft(
    [("pleural effusion", "Possible pleural effusion.")], impression="No pneumothorax is seen."
)


def test_claude_repair_loop_and_request_shape():
    client = FakeClaude([BAD, GOOD])
    report = write_report(pred(["effusion"]), ClaudeGenerator(client=client))
    assert report["source"] == "claude_repaired" and report["verified"]
    assert report["violations_per_attempt"] == [["hallucinated: pneumothorax"], []]
    first, second = client.calls
    assert first["model"] == "claude-opus-5-5"
    assert first["output_config"]["format"]["type"] == "json_schema"
    assert first["fallbacks"] == "default"
    assert "hallucinated: pneumothorax" in second["messages"][0]["content"]
    assert "probability" in first["messages"][0]["content"]  # numbers only, never the image


def test_persistent_failure_falls_back_to_template():
    report = write_report(pred(["effusion"]), ClaudeGenerator(client=FakeClaude([BAD, BAD])))
    assert report["source"] == "template_fallback" and report["verified"]


def test_refusal_falls_back_to_template():
    client = FakeClaude([GOOD], stop_reason="refusal")
    report = write_report(pred(["effusion"]), ClaudeGenerator(client=client))
    assert report["source"] == "template_fallback"
    assert report["violations_per_attempt"][0][0].startswith("generator error")


def test_gemini_backend_uses_json_schema_output():
    client = FakeGemini([GOOD])
    report = write_report(pred(["effusion"]), GeminiGenerator(client=client))
    assert report["source"] == "gemini" and report["verified"]
    cfg = client.calls[0]["config"]
    assert cfg.response_mime_type == "application/json" and cfg.response_json_schema
    assert "chest X-ray" in cfg.system_instruction


def test_service_caches_and_caps_llm_calls():
    client = FakeClaude([GOOD] * 5)
    svc = ReportService(ClaudeGenerator(client=client), max_llm_calls_per_hour=1)
    a = svc(pred(["effusion"]))
    b = svc(pred(["effusion"]))
    assert a["source"] == "claude" and b["cached"] and len(client.calls) == 1
    c = svc(pred(["mass"], triage="NON_URGENT_FINDING"))
    assert c["source"] == "template_rate_limited" and len(client.calls) == 1


def test_auto_backend_without_keys_is_template(monkeypatch):
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert make_generator("auto").name == "template"
    with pytest.raises(ValueError):
        make_generator("nope")


# ---------- offline evaluation + demo integration ----------


def test_report_eval_and_pregenerated_sample_reports(trained_run):
    from fastapi.testclient import TestClient

    from cxr_triage.report_eval import main
    from cxr_triage.serve import create_app

    run, _ = trained_run
    main([str(run), "--backends", "template", "--n", "20"])
    out = json.loads((run / "report_eval.json").read_text())
    s = out["summaries"][0]
    assert s["n"] == 20 and s["final_verified_rate"] == 1.0 and s["est_cost_usd"] == 0.0
    assert (run / "report_eval.md").exists()

    main([str(run), "--backends", "template", "--bundle-samples"])
    client = TestClient(create_app(str(run / "bundle.pt")))
    r = client.get("/samples/0/predict?report=true").json()
    assert r["report"]["pregenerated"] and r["report"]["verified"]
    assert "ground_truth" in r
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.zeros((64, 64), dtype=np.uint8)).save(buf, format="PNG")
    up = client.post("/predict?report=true", files={"file": ("x.png", buf.getvalue(), "image/png")})
    assert up.json()["report"]["source"] == "template"


def test_auto_backend_picks_gemini_on_vertex(monkeypatch):
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "demo-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")
    assert make_generator("auto").name == "gemini"
