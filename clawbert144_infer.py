"""
clawbert144_infer.py — standalone scorer for clawbert-144, the
multi-OCR 11-class court-filing page-type classifier.

This is the OCR-robust successor to ClawBERT (clawbert_infer.py): ModernBERT-base
fine-tuned on the same corpus OCR'd by FIVE engines (PP-OCRv5, Tesseract, docTR,
Hunyuan VLM, Windows OCR), so predictions are stable across OCR engines. Adds an
11th class, `transcript`.

Reproduces the training-time input construction EXACTLY: full-page text, plain
truncation at meta.json's maxlen (no head+tail — ModernBERT's window fits whole
pages), label order from meta.json.

The tokenizer files saved beside the weights were written by a newer transformers
than 4.52 and may not deserialize; we fall back to the identical base tokenizer
(`answerdotai/ModernBERT-base`) — same vocab, so predictions are unaffected.

Public API (mirrors clawbert_infer):
  LABELS            -> list[str] (11 page types, index-aligned to the model)
  score_texts(texts, batch=16) -> (pred_labels:list[str], probs:np.ndarray[N,11])
"""
import os, json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

_HERE = os.path.dirname(os.path.abspath(__file__))
_LOCAL = os.path.join(_HERE, "model")
# local ./model dir if present, else the Hugging Face Hub repo
MODEL_DIR = os.environ.get("CLAWBERT144_DIR",
    _LOCAL if os.path.isfile(os.path.join(_LOCAL, "config.json")) else "RayJackson30/clawbert-144")
BASE_TOKENIZER = "answerdotai/ModernBERT-base"
MAXLEN = 1536
LABELS = None  # filled from config.id2label at load time

_tok = _model = _device = None

def pick_device():
    if not torch.cuda.is_available():
        return torch.device("cpu")
    # the 5090 (sm_120) lacks kernels in this torch build; probe for a usable card
    for i in range(torch.cuda.device_count()):
        try:
            x = torch.zeros(8, device=f"cuda:{i}"); _ = (x + 1).sum().item()
            return torch.device(f"cuda:{i}")
        except Exception:
            continue
    return torch.device("cpu")

def _load():
    global _tok, _model, _device
    if _model is not None:
        return
    try:
        _tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    except Exception:
        _tok = AutoTokenizer.from_pretrained(BASE_TOKENIZER)
    _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    global LABELS
    LABELS = [_model.config.id2label[i] for i in range(_model.config.num_labels)]
    _device = pick_device()
    _model.to(_device).eval()

@torch.no_grad()
def score_texts(texts, batch=16):
    _load()
    use_amp = _device.type == "cuda"
    preds, probs = [], []
    for s in range(0, len(texts), batch):
        enc = _tok([t or "" for t in texts[s:s + batch]], truncation=True,
                   max_length=MAXLEN, padding=True, return_tensors="pt").to(_device)
        if use_amp:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = _model(**enc).logits
        else:
            logits = _model(**enc).logits
        p = torch.softmax(logits.float(), -1).cpu().numpy()
        probs.append(p); preds.extend(p.argmax(1).tolist())
    probs = np.concatenate(probs) if probs else np.zeros((0, len(LABELS)))
    return [LABELS[i] for i in preds], probs

if __name__ == "__main__":
    _load()
    print("device:", _device, "| maxlen:", MAXLEN, "| labels:", LABELS)
    pl, pr = score_texts(["PROOF OF SERVICE\nI declare under penalty of perjury...",
                          "REPORTER'S TRANSCRIPT OF PROCEEDINGS\nTHE COURT: Good morning.\nMR. SMITH: Good morning, your honor."])
    for l, p in zip(pl, pr):
        print(f"  {l:20s} max={p.max():.3f}")
