"""
Live demo: paste a freeform incident report, see it structured in real
time -- side by side with the base (non-fine-tuned) model's attempt at
the same task.

Built for Hugging Face Spaces' free ZeroGPU tier, NOT the free CPU-only
tier -- a 4B-parameter model with 4-bit quantization needs a CUDA GPU
just to load, let alone run, so CPU Basic Spaces will fail outright.
ZeroGPU gives free dynamically-attached GPU time (~5 min/day on a free
account), which is enough for a demo but tight enough that this file:
  - uses plain transformers + peft for serving (not Unsloth -- Unsloth
    loads weights straight onto a CUDA device at import time, which
    doesn't work with ZeroGPU's "no GPU exists until a @spaces.GPU
    function actually runs" model; Unsloth is still the right tool for
    *training*, see training/finetune_colab.ipynb -- this is a
    deliberate train-with-X, serve-with-Y split, not an inconsistency)
  - loads the model ONCE, lazily, on first request, and caches it --
    reloading a multi-GB model on every click would burn through the
    daily quota in a handful of interactions
  - keeps generation short (max_new_tokens=200) to conserve quota

Deploy:
  1. Create a new Space at huggingface.co/new-space -- SDK: Gradio,
     Hardware: ZeroGPU (shown as an option once SDK is Gradio)
  2. Push this file as app.py, plus schema.py and
     app/requirements.txt (as requirements.txt at the Space root)
  3. Set ADAPTER_PATH below to your pushed adapter's HF repo id
  4. Space builds automatically -- public URL appears on the Space page

Runs locally too (falls back to local CUDA if available, errors clearly
if not -- this file assumes a GPU exists somewhere, unlike the training
notebook's CPU-friendly 4-bit loading path):
    python app/gradio_app.py
"""
import sys
from pathlib import Path

import gradio as gr
import spaces
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Robust to either layout: schema.py as a sibling of this file (flat
# deployment -- e.g. both uploaded directly to a Space's repo root) or
# as a sibling of this file's parent directory (nested -- the local
# project structure, schema.py at repo root with this file in app/).
# Same class of bug as eval_harness.py's path lookup on Colab -- fixing
# it defensively here rather than requiring the upload to be exactly right.
for _candidate in (Path(__file__).parent, Path(__file__).parent.parent):
    if (_candidate / "schema.py").exists():
        sys.path.insert(0, str(_candidate))
        break

from schema import SYSTEM_PROMPT_FIELDS, parse_model_output  # noqa: E402

MODEL_NAME = "Qwen/Qwen3-4B"
ADAPTER_PATH = "Sahla/incident-report-qwen3-lora" # set after pushing to the Hub

SYSTEM_PROMPT = f"""You are an incident-report structuring assistant. Given a freeform incident report, extract the following fields as a single JSON object and output ONLY that JSON object, nothing else:

{SYSTEM_PROMPT_FIELDS}
"""

EXAMPLE_REPORTS = [
    "Customers are reporting they can't log in since about 30 minutes ago. "
    "Looks like the auth-service is returning timeouts. I've already "
    "restarted the service once, didn't help. Escalating now.",

    "hey just noticed the internal wiki search hasn't been returning results "
    "all morning, not urgent but figured i'd flag it, nobody's complained yet",

    "URGENT -- unusual query patterns in the customer-db logs over the last "
    "hour, possible unauthorized access. Have not touched anything, waiting "
    "for security team before doing anything that might destroy evidence.",
]

# Lazy-loaded, cached globals -- populated on first request inside the
# @spaces.GPU function below, NOT at import time (no GPU exists yet when
# this module is imported under ZeroGPU).
_tokenizer = None
_base_model = None
_fine_tuned_model = None


def _ensure_models_loaded():
    global _tokenizer, _base_model, _fine_tuned_model
    if _tokenizer is not None:
        return  # already loaded, reuse across requests

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    quant_config = BitsAndBytesConfig(load_in_4bit=True)
    _base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=quant_config, device_map="auto",
    )
    _fine_tuned_model = PeftModel.from_pretrained(_base_model, ADAPTER_PATH)


def _generate(model, report_text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": report_text},
    ]
    inputs = _tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        enable_thinking=False, return_tensors="pt",
    ).to(model.device)
    outputs = model.generate(input_ids=inputs, max_new_tokens=200, temperature=0.1, do_sample=True)
    return _tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)


@spaces.GPU  # GPU attaches only for the duration of this function call
def run_comparison(report_text: str):
    if not report_text.strip():
        return "", ""

    _ensure_models_loaded()

    try:
        base_raw = _generate(_base_model, report_text)
        base_parsed = parse_model_output(base_raw)
        base_json = base_parsed.model_dump_json(indent=2) if base_parsed else f"(failed to parse)\n{base_raw}"
    except Exception as e:
        base_json = f"(error during generation)\n{e}"

    try:
        ft_raw = _generate(_fine_tuned_model, report_text)
        ft_parsed = parse_model_output(ft_raw)
        fine_tuned_json = ft_parsed.model_dump_json(indent=2) if ft_parsed else f"(failed to parse)\n{ft_raw}"
    except Exception as e:
        fine_tuned_json = f"(error during generation)\n{e}"

    return base_json, fine_tuned_json


with gr.Blocks(title="Incident Report Structuring") as demo:
    gr.Markdown(
        "# Incident Report Structuring — Base vs. Fine-Tuned\n"
        "Paste a freeform incident report below. See how a LoRA-fine-tuned "
        "small model extracts clean, schema-conforming JSON compared to the "
        "same base model with only a system-prompt instruction.\n\n"
        "_Running on free Hugging Face ZeroGPU — first request may take a "
        "little longer while the model loads._"
    )

    report_input = gr.Textbox(
        label="Incident report (freeform text)",
        placeholder="Paste or type an incident report here...",
        lines=5,
    )

    gr.Examples(examples=EXAMPLE_REPORTS, inputs=report_input, label="Try an example")

    run_button = gr.Button("Extract structured fields", variant="primary")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Base model (zero-shot)")
            base_output = gr.Code(language="json", label=None)
        with gr.Column():
            gr.Markdown("### Fine-tuned model (LoRA)")
            fine_tuned_output = gr.Code(language="json", label=None)

    run_button.click(fn=run_comparison, inputs=report_input, outputs=[base_output, fine_tuned_output])

if __name__ == "__main__":
    demo.launch()