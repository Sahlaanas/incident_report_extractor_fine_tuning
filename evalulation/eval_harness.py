"""
Scores a model (base or fine-tuned) against data/test.jsonl, field by
field, plus JSON-validity rate and full-exact-match rate.

Designed to score ANY model given a HF model id/path, so the same script
produces both the "base" and "fine-tuned" rows of your results table --
run it twice, once per model, rather than needing separate scripts.

Usage (run from the repo root, after training -- needs a GPU, so this
is meant to run in the same Colab session right after training, or
locally if you've downloaded the adapter):

    # Base model, zero-shot prompted:
    python evaluation/eval_harness.py --model unsloth/Qwen3-4B-bnb-4bit --label "base"

    # Fine-tuned (base + LoRA adapter):
    python evaluation/eval_harness.py --model unsloth/Qwen3-4B-bnb-4bit \\
        --adapter ./lora_adapter --label "fine-tuned"

    # Or a model already pushed to the Hub:
    python evaluation/eval_harness.py --model your-username/incident-report-qwen3-lora --label "fine-tuned"
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from schema import SYSTEM_PROMPT_FIELDS, IncidentReport, parse_model_output  # noqa: E402

SYSTEM_PROMPT = f"""You are an incident-report structuring assistant. Given a freeform incident report, extract the following fields as a single JSON object and output ONLY that JSON object, nothing else:

{SYSTEM_PROMPT_FIELDS}
"""

FIELDS = list(IncidentReport.model_fields.keys())

# "reported_by" is only ever given to the data-generation writer model as
# a voice/perspective instruction, not something the text is guaranteed
# to state explicitly -- real incident reports mostly don't self-announce
# the author's job title. It tends to sit near chance level regardless of
# model quality, and silently dominates a strict all-8-fields exact-match
# score. Tracking a second, fairer metric that excludes it alongside the
# strict one, rather than just reporting one potentially misleading number.
FIELDS_EXCL_REPORTED_BY = [f for f in FIELDS if f != "reported_by"]


def find_default_test_path() -> Path:
    """
    Looks in a few likely locations so this works both when run from the
    full repo structure (evaluation/eval_harness.py with data/ as a
    sibling of evaluation/'s parent) and when this file was uploaded flat
    into a single directory alongside test.jsonl (e.g. a Colab session,
    where subfolder structure typically isn't preserved on upload).
    """
    candidates = [
        Path(__file__).parent.parent / "data" / "test.jsonl",  # full repo structure
        Path(__file__).parent / "data" / "test.jsonl",
        Path(__file__).parent / "test.jsonl",                   # flat upload, same dir as this script
        Path.cwd() / "test.jsonl",                               # flat upload, cwd
        Path.cwd() / "data" / "test.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # Nothing found — return the most likely intended path anyway, so the
    # resulting FileNotFoundError at least points somewhere sensible.
    return candidates[0]


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_model(model_name: str, adapter_path: str | None):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True,
    )
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)

    FastLanguageModel.for_inference(model)
    return model, tokenizer


def run_inference(model, tokenizer, report_text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": report_text},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        enable_thinking=False,  # Qwen3 defaults to a <think>...</think> block otherwise
        return_tensors="pt",
    ).to("cuda")
    outputs = model.generate(input_ids=inputs, max_new_tokens=200, temperature=0.1, do_sample=True)
    return tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)


def score(predicted: IncidentReport | None, gold: IncidentReport) -> dict:
    if predicted is None:
        return {field: False for field in FIELDS} | {
            "_parsed": False, "_exact_match": False, "_exact_match_excl_reported_by": False,
        }

    field_results = {field: (getattr(predicted, field) == getattr(gold, field)) for field in FIELDS}
    exact_match = all(field_results.values())
    exact_match_excl_reported_by = all(field_results[f] for f in FIELDS_EXCL_REPORTED_BY)
    return field_results | {
        "_parsed": True,
        "_exact_match": exact_match,
        "_exact_match_excl_reported_by": exact_match_excl_reported_by,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model id or path")
    parser.add_argument("--adapter", default=None, help="Path or HF id of a LoRA adapter to apply on top of --model")
    parser.add_argument("--label", required=True, help="Label for this run in the output (e.g. 'base', 'fine-tuned')")
    parser.add_argument("--test-file", default=None, help="Defaults to data/test.jsonl relative to repo root")
    args = parser.parse_args()

    test_path = Path(args.test_file) if args.test_file else find_default_test_path()
    test_rows = load_jsonl(test_path)
    print(f"Loaded {len(test_rows)} test examples from {test_path}")

    print(f"Loading model: {args.model}" + (f" + adapter {args.adapter}" if args.adapter else " (base, zero-shot)"))
    model, tokenizer = load_model(args.model, args.adapter)

    results = []
    total_latency = 0.0

    for i, row in enumerate(test_rows):
        gold = IncidentReport(**row["label"])

        start = time.monotonic()
        raw_output = run_inference(model, tokenizer, row["report_text"])
        elapsed = time.monotonic() - start
        total_latency += elapsed

        predicted = parse_model_output(raw_output)
        row_scores = score(predicted, gold)
        results.append(row_scores)

        status = "✓" if row_scores["_exact_match"] else ("partial" if row_scores["_parsed"] else "unparseable")
        print(f"  [{i + 1}/{len(test_rows)}] {status} ({elapsed:.1f}s)")

    # ── Aggregate ────────────────────────────────────────────────────
    n = len(results)
    json_validity_rate = sum(r["_parsed"] for r in results) / n
    exact_match_rate = sum(r["_exact_match"] for r in results) / n
    exact_match_excl_reported_by_rate = sum(r["_exact_match_excl_reported_by"] for r in results) / n
    avg_latency = total_latency / n

    field_accuracy = {
        field: sum(r[field] for r in results) / n
        for field in FIELDS
    }

    print(f"\n── Results: {args.label} ({args.model}" + (f" + {args.adapter}" if args.adapter else "") + f") " + "─" * 20)
    print(f"JSON validity rate: {json_validity_rate:.1%}")
    print(f"Full exact-match rate (all 8 fields): {exact_match_rate:.1%}")
    print(f"Exact-match rate (7 fields, excl. reported_by): {exact_match_excl_reported_by_rate:.1%}")
    print(f"Avg inference latency: {avg_latency:.2f}s")
    print("Per-field accuracy:")
    for field, acc in field_accuracy.items():
        print(f"  {field:20s} {acc:.1%}")

    # Also dump machine-readable results for building the README table later
    out_path = Path(__file__).parent / f"results_{args.label.replace(' ', '_')}.json"
    with open(out_path, "w") as f:
        json.dump({
            "label": args.label,
            "model": args.model,
            "adapter": args.adapter,
            "n_examples": n,
            "json_validity_rate": json_validity_rate,
            "exact_match_rate": exact_match_rate,
            "exact_match_excl_reported_by_rate": exact_match_excl_reported_by_rate,
            "avg_latency_seconds": avg_latency,
            "field_accuracy": field_accuracy,
        }, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()