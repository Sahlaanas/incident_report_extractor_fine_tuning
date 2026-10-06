# Incident Report Structuring — LoRA Fine-Tuning

Fine-tunes a small open-weight LLM (Qwen3-4B) via LoRA to extract clean,
schema-conforming JSON from freeform incident reports — the kind of
messy, inconsistently-written text that actually shows up in Slack
messages and support tickets, rather than carefully formatted input.


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

