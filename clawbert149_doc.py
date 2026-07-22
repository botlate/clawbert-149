"""
clawbert149_doc.py — whole-document scorer for clawbert-149.

The per-page model (clawbert149_infer.py) classifies one page at a time. This
runs the shipped document-context layer on top: each page is embedded by the
clawbert-149 backbone (mean-pooled last hidden state), then a small bidirectional
transformer (docxf, 1.8M params) runs over the whole document's page sequence —
every page attends to every other page — before per-page labels come out.
Ambiguous pages (mid-filing caption pages, TOC pages) get decided with document
context. Pooled test macro-F1 0.937 -> 0.949; subsequent_cover_page F1 0.68 -> 0.77.

Input is ONE document's pages, in page order. Don't feed it an appendix or a
combined record — split first, classify second.

Public API:
  score_document(pages: list[str], batch=16) -> (labels: list[str], probs: np.ndarray[N, 11])
"""
import os, json, math
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

REPO = os.environ.get("CLAWBERT149_DIR", "RayJackson30/clawbert-149")
BASE_TOKENIZER = "answerdotai/ModernBERT-base"
MAXLEN, EMB_DIM, D_MODEL, N_HEADS, N_LAYERS = 1536, 768, 256, 4, 2
WINDOW, STRIDE = 256, 128          # sliding window for very long documents

LABELS = None
_tok = _enc = _net = _device = None


class DocTransformer(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(EMB_DIM, D_MODEL), nn.ReLU(), nn.Dropout(0.0))
        layer = nn.TransformerEncoderLayer(D_MODEL, N_HEADS, dim_feedforward=4 * D_MODEL,
                                           dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, N_LAYERS)
        self.head = nn.Linear(D_MODEL, n_classes)

    def _pos(self, T, device):
        pe = torch.zeros(T, D_MODEL, device=device)
        p = torch.arange(T, device=device).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, D_MODEL, 2, device=device).float()
                        * (-math.log(10000.0) / D_MODEL))
        pe[:, 0::2] = torch.sin(p * div); pe[:, 1::2] = torch.cos(p * div)
        return pe.unsqueeze(0)

    def forward(self, x):
        h = self.proj(x) + self._pos(x.shape[1], x.device)
        return self.head(self.encoder(h))


def _pick_device():
    if not torch.cuda.is_available():
        return torch.device("cpu")
    for i in range(torch.cuda.device_count()):
        try:
            x = torch.zeros(8, device=f"cuda:{i}"); _ = (x + 1).sum().item()
            return torch.device(f"cuda:{i}")
        except Exception:
            continue
    return torch.device("cpu")


def _load():
    global _tok, _enc, _net, _device, LABELS
    if _net is not None:
        return
    try:
        _tok = AutoTokenizer.from_pretrained(REPO)
    except Exception:
        _tok = AutoTokenizer.from_pretrained(BASE_TOKENIZER)
    _enc = AutoModel.from_pretrained(REPO)
    if os.path.isdir(REPO):
        wpath = os.path.join(REPO, "docxf", "docxf.pt")
        mpath = os.path.join(REPO, "docxf", "meta.json")
    else:
        from huggingface_hub import hf_hub_download
        wpath = hf_hub_download(REPO, "docxf/docxf.pt")
        mpath = hf_hub_download(REPO, "docxf/meta.json")
    LABELS = json.load(open(mpath, encoding="utf-8"))["labels"]
    _net = DocTransformer(len(LABELS))
    _net.load_state_dict(torch.load(wpath, map_location="cpu", weights_only=True))
    _device = _pick_device()
    _enc.to(_device).eval(); _net.to(_device).eval()


@torch.no_grad()
def _embed(pages, batch):
    out = np.zeros((len(pages), EMB_DIM), np.float32)
    order = np.argsort([len(p or "") for p in pages], kind="stable")
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        e = _tok([pages[i] or "" for i in idx], truncation=True, max_length=MAXLEN,
                 padding=True, return_tensors="pt").to(_device)
        if _device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = _enc(**e).last_hidden_state
        else:
            h = _enc(**e).last_hidden_state
        m = e["attention_mask"].unsqueeze(-1).float()
        out[idx] = ((h.float() * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy()
    return out


@torch.no_grad()
def score_document(pages, batch=16):
    """pages: one document's page texts, in page order."""
    _load()
    E = _embed(pages, batch)
    T = len(pages)
    lsum = np.zeros((T, len(LABELS))); cnt = np.zeros((T, 1))
    s = 0
    while True:
        e = min(T, s + WINDOW)
        X = torch.from_numpy(E[s:e]).unsqueeze(0).to(_device)
        lsum[s:e] += _net(X)[0].cpu().numpy(); cnt[s:e] += 1
        if e == T:
            break
        s += STRIDE
    probs = torch.softmax(torch.from_numpy(lsum / cnt), -1).numpy()
    return [LABELS[i] for i in probs.argmax(1)], probs


if __name__ == "__main__":
    labels, probs = score_document([
        "SUPERIOR COURT OF THE STATE OF CALIFORNIA\nCOUNTY OF SACRAMENTO\nVERIFIED PETITION FOR WRIT OF MANDATE",
        "1\nPetitioner alleges as follows:\n2\n1. Petitioner is a resident of the County of Sacramento.",
        "PROOF OF SERVICE\nI declare under penalty of perjury...",
    ])
    for l, p in zip(labels, probs):
        print(f"  {l:22s} {p.max():.3f}")
