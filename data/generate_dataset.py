"""
Generates synthetic (report_text, label) training pairs for the
incident-structuring task.

Methodology (the part worth explaining in an interview): most synthetic
data pipelines ask an LLM to EXTRACT a label from text it also wrote,
which means label errors from the generation step silently become
"ground truth." This script avoids that by generating the STRUCTURED
FACTS first (via Python's random module, not an LLM), then asking the
LLM only to render those facts as a realistic, naturally-written report
in a randomized tone/style. The facts are the label — there's no
extraction step to get wrong, so the dataset has zero label noise by
construction.

Uses a local Ollama model (same free/local approach as the companion
"Enterprise Agentic Ops & Compliance Copilot" project) to do the
writing. Swap `OLLAMA_MODEL` below if you'd rather use a different
local model.

Usage:
    python data/generate_dataset.py --n 600
    # writes data/train.jsonl (70%), data/val.jsonl (15%), data/test.jsonl (15%)
"""
import argparse
import json
import random
import sys
from pathlib import Path

from ollama import Client

sys.path.insert(0, str(Path(__file__).parent.parent))
from schema import (  # noqa: E402
    AFFECTED_SYSTEMS,
    ActionTaken,
    Category,
    IncidentReport,
    ReportedBy,
    Severity,
    TimeDetected,
)

OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_HOST = "http://localhost:11434"

SEVERITIES: list[Severity] = ["critical", "high", "medium", "low"]
CATEGORIES: list[Category] = ["security", "infrastructure", "application", "data", "network"]
REPORTED_BY: list[ReportedBy] = [
    "on-call engineer", "SRE", "customer support agent",
    "security analyst", "devops engineer", "qa tester", "unspecified",
]
TIMES_DETECTED: list[TimeDetected] = [
    "just now", "within the last hour", "this morning",
    "this afternoon", "overnight", "yesterday", "unspecified",
]
# Systems customers never interact with directly -- an incident here
# should essentially never be framed as customer-facing, regardless of
# how the random sampler would otherwise roll customer_impact.
INTERNAL_ONLY_SYSTEMS = {"ci-cd-pipeline", "internal-wiki", "backup-storage", "vpn-gateway"}

ACTIONS_TAKEN: list[ActionTaken] = [
    "none", "restarted the service", "rolled back the deployment",
    "escalated to on-call", "applied a temporary workaround",
    "disabled the affected feature", "notified the vendor",
]
CATEGORY_HINTS = {
    "security": "unauthorized access, suspicious login attempts, a potential data breach, unusual account activity, or signs of an exploited vulnerability",
    "infrastructure": "a server, network hardware, storage, or cloud infrastructure problem -- not an application code bug",
    "application": "a bug, crash, or incorrect behavior in a specific application feature or piece of software",
    "data": "incorrect, missing, corrupted, duplicated, or inconsistent data records",
    "network": "connectivity issues, latency, packet loss, DNS problems, or routing failures",
}

SEVERITY_HINTS = {
    "critical": "a full outage or major business-critical failure with widespread impact, needing immediate attention right now",
    "high": "a serious problem affecting many users or a core function, needing prompt attention soon",
    "medium": "a real but contained problem affecting a limited scope -- annoying, not urgent",
    "low": "a minor, low-impact issue that can be handled in the normal course of business, no rush",
}

WRITING_STYLES = [
    "terse and technical, like a quick status update",
    "a bit panicked and rambling",
    "calm, detailed, and methodical",
    "written by a non-technical end user who doesn't know the jargon",
    "formal incident-ticket style with full sentences",
    "a quick, casual Slack message to the team channel",
    "written by someone clearly annoyed and frustrated",
]


def sample_fact_set() -> IncidentReport:
    """Randomly samples a plausible, internally-consistent set of facts."""
    severity = random.choice(SEVERITIES)
    # Escalation correlates with severity but isn't deterministic — keeps
    # the task from being trivially inferable from severity alone.
    requires_escalation = (
        severity in ("critical", "high") and random.random() < 0.85
    ) or (severity in ("medium", "low") and random.random() < 0.1)

    category = random.choice(CATEGORIES)
    affected_system = random.choice(AFFECTED_SYSTEMS)

    if affected_system in INTERNAL_ONLY_SYSTEMS:
        # Customers never interact with these systems directly, so a
        # "customer impact" framing would be factually wrong regardless
        # of category/severity — don't even roll the dice on it.
        customer_impact = False
    else:
        customer_impact = (
            category in ("security", "application", "data") and random.random() < 0.6
        ) or random.random() < 0.15

    return IncidentReport(
        severity=severity,
        category=category,
        affected_system=affected_system,
        reported_by=random.choice(REPORTED_BY),
        time_detected=random.choice(TIMES_DETECTED),
        action_taken=random.choice(ACTIONS_TAKEN),
        requires_escalation=requires_escalation,
        customer_impact=customer_impact,
    )


def build_writer_prompt(facts: IncidentReport, style: str) -> str:
    reported_by_clause = (
        "" if facts.reported_by == "unspecified"
        else f"Written from the perspective of a {facts.reported_by}. "
    )
    time_clause = (
        "" if facts.time_detected == "unspecified"
        else f"It was detected {facts.time_detected}. "
    )
    action_clause = (
        "No remediation has been attempted yet. " if facts.action_taken == "none"
        else f"The following was already tried, described in your own words rather than "
             f"quoted directly: {facts.action_taken}. "
    )
    escalation_clause = (
        "The tone should make clear this needs to be escalated to more senior/specialized "
        "people beyond whoever is currently handling it. "
        if facts.requires_escalation
        else "The tone should make clear the current team has this handled without needing "
             "to pull in anyone else. "
    )
    if facts.customer_impact:
        impact_clause = "Customer-facing impact should be evident from the report. "
    elif facts.affected_system in INTERNAL_ONLY_SYSTEMS:
        impact_clause = (
            f"'{facts.affected_system}' is a purely internal/backend system that "
            f"customers never interact with directly — frame this as an internal or "
            f"employee-facing issue, not a customer-facing one. Do not mention customer "
            f"impact at all. "
        )
    else:
        impact_clause = (
            "There is no customer-facing impact here — frame this as a purely internal "
            "issue. Do not mention customers being affected. "
        )

    return (
        f"Write a short, realistic incident report (60-120 words) affecting the "
        f"'{facts.affected_system}' system.\n\n"
        f"The underlying problem should be the kind of thing associated with: "
        f"{CATEGORY_HINTS[facts.category]}.\n\n"
        f"The impact and urgency should feel like: {SEVERITY_HINTS[facts.severity]}.\n\n"
        f"{reported_by_clause}{time_clause}{action_clause}{escalation_clause}"
        f"{impact_clause}"
        f"Write in this style: {style}.\n\n"
        f"CRITICAL RULES:\n"
        f"- Do NOT use a structured/labeled format (no 'Severity:', 'Category:' etc. fields).\n"
        f"- Do NOT use the literal words '{facts.severity}' or '{facts.category}' anywhere "
        f"in the text, and do not use phrases like '{facts.severity}-severity' or "
        f"'{facts.category} incident'.\n"
        f"- Let the symptoms and tone imply the severity and category naturally — the "
        f"way a real person describing this problem would actually write it, without "
        f"labeling or categorizing it themselves.\n"
        f"- Output ONLY the report text, nothing else."
    )


def generate_example(client: Client) -> dict:
    facts = sample_fact_set()
    style = random.choice(WRITING_STYLES)
    prompt = build_writer_prompt(facts, style)

    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.9},  # high temperature — we want varied phrasing, not varied facts
    )
    report_text = response["message"]["content"].strip()

    return {"report_text": report_text, "label": facts.model_dump()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=600, help="Total number of examples to generate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    client = Client(host=OLLAMA_HOST)

    examples = []
    print(f"Generating {args.n} synthetic examples via {OLLAMA_MODEL}...")
    for i in range(args.n):
        try:
            examples.append(generate_example(client))
        except Exception as e:
            print(f"  [WARN] Skipped example {i} due to error: {e}")
            continue
        if (i + 1) % 25 == 0:
            print(f"  ...{i + 1}/{args.n}")

    random.shuffle(examples)
    n = len(examples)
    train_end = int(n * 0.70)
    val_end = train_end + int(n * 0.15)

    splits = {
        "train": examples[:train_end],
        "val": examples[train_end:val_end],
        "test": examples[val_end:],
    }

    out_dir = Path(__file__).parent
    for name, rows in splits.items():
        path = out_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"Wrote {len(rows)} examples to {path}")


if __name__ == "__main__":
    main()