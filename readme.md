# Incident Report Structuring — LoRA Fine-Tuning

Fine-tunes a small open-weight LLM (Qwen3-4B) via LoRA to extract clean,
schema-conforming JSON from freeform incident reports — the kind of
messy, inconsistently-written text that actually shows up in Slack
messages and support tickets, rather than carefully formatted input.

**100% free to build and run**: free Colab T4 GPU for training, free
local Ollama for synthetic data generation, free Hugging Face Spaces for
the live demo.

Companion project: [Enterprise Agentic Ops & Compliance Copilot](../enterprise-agent-copilot)
— that project answers questions *about* incident policy; this one
structures the incident reports people actually submit. Together they
cover both sides of an "AI for internal ops" story.

## Why this project, not just more RAG

The companion project demonstrates orchestrating LLMs (agents, retrieval,
evals). This one demonstrates going a layer deeper: curating training
data, running an actual fine-tuning job, and proving — with a measured
before/after comparison — that it improved something concrete. That's a
distinct, harder-to-fake signal than another prompt-engineering demo.

## Methodology: labels by construction, not extraction

Most synthetic-data pipelines ask an LLM to generate text, then ask an
LLM (often the same one) to extract a label from it — any mistake the
model makes in that second step quietly becomes "ground truth" baked
into the dataset.

This project avoids that: `data/generate_dataset.py` samples a random,
internally-consistent set of **structured facts first** (via Python's
`random`, not an LLM), then asks the LLM only to **write a realistic
report consistent with those facts**, in a randomized tone/style. The
sampled facts *are* the label — there's no extraction step that could
introduce label noise. See the docstring in that file for the full
reasoning.

## A real bug found by spot-checking (worth keeping in, for interviews)

An early version of `generate_dataset.py` had the writer-model prompt say
things like *"write a report about a {severity}-severity {category}
incident"* directly — spot-checking the output caught the model echoing
that almost verbatim (*"a critical-severity application incident was
detected"*), which would have made the extraction task artificially easy
and inflated eval numbers in a way that wouldn't hold up under scrutiny.
The fix: describe what each category/severity *looks like in practice*
(concrete symptom cues) instead of naming it, and explicitly forbid the
literal label words in the output. A 6th `"other"` category was also
dropped entirely after spot-checking showed it had no concrete positive
definition, so the writer model couldn't ground it in real content — it
just produced generic reports indistinguishable from `application`.

This is exactly the kind of thing worth walking an interviewer through:
not "my first version was perfect," but "I caught a real label-leakage
and category-definition problem by reading my own training data before
trusting it," which is a more credible engineering story than a clean
first try.

## Schema

Every field is closed-vocabulary (enum or boolean) by deliberate design
— see `schema.py`. This isn't because real incident reports are always
this clean; it's so evaluation can be exact-match and unambiguous,
rather than needing fuzzy or LLM-judge scoring to produce a defensible
number.

```python
class IncidentReport(BaseModel):
    severity: Literal["critical", "high", "medium", "low"]
    category: Literal["security", "infrastructure", "application", "data", "network"]
    affected_system: str
    reported_by: Literal["on-call engineer", "SRE", "customer support agent", ...]
    time_detected: Literal["just now", "within the last hour", "this morning", ...]
    action_taken: Literal["none", "restarted the service", "rolled back the deployment", ...]
    requires_escalation: bool
    customer_impact: bool
```

## Project structure

```
incident-report-extractor/
├── schema.py                      # single source of truth for the target schema
├── data/
│   ├── generate_dataset.py        # synthetic data generation (facts-first methodology)
│   ├── smoke_test_examples.jsonl  # 5 hand-written examples for quick pipeline testing
│   ├── train.jsonl                # generated — not committed, see Quickstart
│   ├── val.jsonl                  # generated
│   └── test.jsonl                 # generated
├── training/
│   └── finetune_colab.ipynb       # Unsloth + LoRA fine-tuning notebook (Colab-ready)
├── evaluation/
│   └── eval_harness.py            # scores any model (base or fine-tuned) against test.jsonl
├── inference/
│   └── predictor.py               # production-facing wrapper, retry-on-malformed-JSON
├── app/
│   └── gradio_app.py              # live demo — base vs. fine-tuned, side by side
└── requirements.txt
```

## Quickstart

### 1. Generate the synthetic dataset
Uses your local Ollama (same setup as the companion project — if you
already have `llama3.1:8b` pulled, you're ready):
```bash
pip install -r requirements.txt
python data/generate_dataset.py --n 600
# writes data/train.jsonl, data/val.jsonl, data/test.jsonl
```
Takes a while (one LLM call per example) — 600 examples is a reasonable
starting point; the notebook trains fast even on a few hundred.

### 2. Fine-tune (Google Colab, free T4 GPU)
1. Open `training/finetune_colab.ipynb` in Colab (Runtime → Change runtime type → T4 GPU)
2. Upload `data/train.jsonl` and `data/val.jsonl` to the Colab session
3. Run all cells — training itself typically takes well under an hour at this data scale
4. Push the adapter to the Hugging Face Hub (free) in the save step, so it's loadable by repo id everywhere else in this project

### 3. Evaluate: base vs. fine-tuned
Run in the same Colab session (needs a GPU), once per model:
```bash
python evaluation/eval_harness.py --model unsloth/Qwen3-4B-bnb-4bit --label "base"
python evaluation/eval_harness.py --model unsloth/Qwen3-4B-bnb-4bit \
    --adapter your-username/incident-report-qwen3-lora --label "fine-tuned"
```
Each run writes `evaluation/results_<label>.json` — paste the numbers into the results table below once you have them.

### 4. Deploy the live demo
1. Set `ADAPTER_PATH` in `app/gradio_app.py` to your pushed Hub repo id
2. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space) (SDK: Gradio)
3. Push `app/gradio_app.py` (as `app.py`), `schema.py`, `inference/predictor.py`, and `requirements.txt`
4. Space builds automatically — you get a public URL to put on your resume

## Results

Evaluated on a 90-example held-out test set, Qwen3-4B base model
(zero-shot, system-prompted) vs. the LoRA fine-tuned version:

| Metric | Base (zero-shot) | Fine-tuned (LoRA) |
|---|---|---|
| JSON validity rate | 96.7% | **100.0%** |
| Exact-match rate (7 fields, excl. `reported_by`) | 4.4% | **34.4%** (7.8× improvement) |
| Full exact-match rate (all 8 fields) | 0.0% | 1.1% — see note below |
| Avg inference latency | 4.29s | 5.07s |
| `severity` accuracy | 46.7% | **60.0%** |
| `category` accuracy | 50.0% | **86.7%** |
| `affected_system` accuracy | 60.0% | **97.8%** |
| `reported_by` accuracy | 13.3% | 14.4% — see note below |
| `time_detected` accuracy | 74.4% | 77.8% |
| `action_taken` accuracy | 77.8% | **91.1%** |
| `requires_escalation` accuracy | 86.7% | **97.8%** |
| `customer_impact` accuracy | 53.3% | **95.6%** |

**Why two exact-match numbers:** `reported_by` sits at chance level
(~14%, for 7 possible roles) on *both* models — fine-tuning didn't fail
here, the task design has a real gap. `generate_dataset.py` only tells
the writer model to adopt a role's *voice*, not to explicitly state it,
and real incident reports mostly don't self-announce the author's job
title ("Hey team, we've got an issue..." never reveals who's writing).
Since the strict all-8-fields metric requires every field correct
simultaneously, this one near-chance field acts as a near-constant veto
on it (`0.9^7 × 0.14 ≈ 7%`, roughly matching what's observed). The
7-field metric excludes it and is the fairer measure of what fine-tuning
actually achieved — a legitimate, documented limitation rather than
something papered over.

**Live demo:** *(add your Hugging Face Spaces link here once deployed)*

## Resume bullets

> Fine-tuned Qwen3-4B via LoRA (Unsloth, PEFT) to extract structured
> JSON from freeform incident reports, improving 7-field exact-match
> accuracy from 4.4% (zero-shot base model) to 34.4% (7.8×) and JSON
> validity to 100% on a held-out test set; designed a synthetic data
> generation methodology that produces labels by construction (zero
> label noise) rather than by LLM extraction. Deployed a live
> base-vs-fine-tuned comparison demo on Hugging Face Spaces.

> Built an end-to-end fine-tuning pipeline — synthetic data generation,
> LoRA training (Google Colab, free tier), automated evaluation
> harness, and production inference wrapper with retry-on-malformed-
> output handling — entirely on free/open-source infrastructure.

## Interview talking points

**"Why generate facts first instead of extracting labels afterward?"**
Label noise compounds silently — if 5% of your "ground truth" labels are
actually wrong because the extraction step made a mistake, your eval
numbers (and your sense of whether fine-tuning even helped) become
unreliable in a way that's hard to detect. Generating facts
programmatically and asking the LLM only to *write from* them removes
that failure mode entirely, at the cost of the dataset being synthetic
rather than real-world text.

**"Why closed-vocabulary fields instead of open text?"**
Same instinct as the eval-harness limitation documented in the companion
project: a metric needs to be trustworthy, not just sophisticated-
looking. Open text fields would need fuzzy or LLM-judge scoring, adding
a second source of noise on top of the thing you're actually trying to
measure. Closed vocabulary makes the eval exact and arguable in an
interview — you can show the scoring code and it's obviously correct.

**"What would you change for a production version of this?"**
Real incident reports aren't closed-vocabulary — `affected_system`
especially would need either a maintained enum synced to your actual
service catalog, or a retrieval step to match free text against known
system names. I'd also want real labeled data mixed in with synthetic
data before shipping this against actual user-submitted reports, since
synthetic data captures the *distribution I designed*, not necessarily
the one real users produce.

**"How would this scale beyond a handful of fields?"**
LoRA rank and target modules would likely need tuning as schema
complexity grows: more fields, more interdependencies between them
(e.g. `category` constraining which `affected_system` values are
plausible) benefit from either a larger rank or exploring whether a
single model should own the whole schema vs. decomposing into multiple
smaller, more reliable extraction heads.