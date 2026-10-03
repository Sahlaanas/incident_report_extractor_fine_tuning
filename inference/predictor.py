"""
Inference wrapper around the fine-tuned model. Loads once, exposes a
single `extract()` call that returns a validated IncidentReport (or
raises a clear error after exhausting retries) -- the kind of thin,
production-facing layer that should sit between "a model" and "a thing
an application actually calls."

Includes one retry with a stricter instruction if the first generation
doesn't parse as valid JSON against the schema -- small models
occasionally wrap output in prose or markdown fences despite
instructions not to, and a single retry resolves most of those cases
without meaningfully hurting latency.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from schema import SYSTEM_PROMPT_FIELDS, IncidentReport, parse_model_output  # noqa: E402

SYSTEM_PROMPT = f"""You are an incident-report structuring assistant. Given a freeform incident report, extract the following fields as a single JSON object and output ONLY that JSON object, nothing else:

{SYSTEM_PROMPT_FIELDS}
"""

RETRY_SUFFIX = (
    "\n\nIMPORTANT: Output ONLY the raw JSON object. No markdown code "
    "fences, no explanation, no extra text before or after."
)


class IncidentExtractor:
    def __init__(self, model_name: str, adapter_path: str | None = None):
        """
        model_name: HF model id (e.g. "unsloth/Qwen3-4B-bnb-4bit")
        adapter_path: local path or HF id of the LoRA adapter. Omit to
                      run the base model (useful for A/B comparisons).
        """
        from unsloth import FastLanguageModel

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=512,
            dtype=None,
            load_in_4bit=True,
        )
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        FastLanguageModel.for_inference(self.model)
        self.is_fine_tuned = adapter_path is not None

    def _generate(self, report_text: str, strict: bool = False) -> str:
        prompt = SYSTEM_PROMPT + (RETRY_SUFFIX if strict else "")
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": report_text},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=False,  # Qwen3 defaults to a <think>...</think> block otherwise
            return_tensors="pt",
        ).to("cuda")
        outputs = self.model.generate(
            input_ids=inputs, max_new_tokens=200, temperature=0.1, do_sample=True
        )
        return self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

    def extract(self, report_text: str) -> IncidentReport:
        """
        Returns a validated IncidentReport. Raises ValueError if the
        model's output doesn't parse as valid schema-conforming JSON
        even after one stricter retry.
        """
        raw = self._generate(report_text, strict=False)
        parsed = parse_model_output(raw)
        if parsed is not None:
            return parsed

        # One retry with a stricter instruction before giving up.
        raw_retry = self._generate(report_text, strict=True)
        parsed_retry = parse_model_output(raw_retry)
        if parsed_retry is not None:
            return parsed_retry

        raise ValueError(
            f"Model output did not parse as valid IncidentReport JSON "
            f"after 1 retry. Last raw output: {raw_retry!r}"
        )


if __name__ == "__main__":
    # Quick manual smoke test — point ADAPTER_PATH at your downloaded
    # adapter, or leave it None to test the base model.
    ADAPTER_PATH = None  # e.g. "../training/lora_adapter" or a HF repo id

    extractor = IncidentExtractor(
        model_name="unsloth/Qwen3-4B-bnb-4bit",
        adapter_path=ADAPTER_PATH,
    )

    sample_report = (
        "Customers are reporting they can't log in since about 30 minutes ago. "
        "Looks like the auth-service is returning timeouts. I've already "
        "restarted the service once, didn't help. Escalating now."
    )

    result = extractor.extract(sample_report)
    print(result.model_dump_json(indent=2))