# Incident Report Structuring — LoRA Fine-Tuning

Fine-tunes a small open-weight LLM (Qwen3-4B) via LoRA to extract clean,
schema-conforming JSON from freeform incident reports — the kind of
messy, inconsistently-written text that actually shows up in Slack
messages and support tickets, rather than carefully formatted input.

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




## Project structure

```
incident-report-extractor/
├── schema.py                     
├── data/
│   ├── generate_dataset.py        
│   ├── smoke_test_examples.jsonl  
│   ├── train.jsonl                
│   ├── val.jsonl         
│   └── test.jsonl                 
├── training/
│   └── finetune_colab.ipynb       
├── evaluation/
│   └── eval_harness.py            
├── inference/
│   └── predictor.py               
├── app/
│   └── gradio_app.py             
└── requirements.txt
```



### 4. Deployed the live demo
<img width="1634" height="754" alt="image" src="https://github.com/user-attachments/assets/70b55e88-c40c-4059-8506-cf66fa01fd24" />

