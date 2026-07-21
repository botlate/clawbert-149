# clawbert-144

A page-type classifier for California court filings. Give it the OCR text of one
page, and it tells you what that page *is* — one of 11 structural types:

`body · cover_page · subsequent_cover_page · toc · toa · exhibit_cover ·
proof_of_service · verification · judicial_form · transcript · unknown_other`

One of each, straight from filed documents (individual images in
[docs/examples/](docs/examples/)):

![one example page per class](docs/page_types.jpg)

## It's not keyword matching

A few held-out pages picked specifically to fool it — a declaration that reads
like a verification, a petition that says PROOF OF SERVICE in the text, a
citation-dense argument page that reads like a TOA, and a mid-filing caption
page that reads like a cover. Real predictions, real confidences:

![lookalike pages with model predictions](docs/lookalikes.jpg)

## What it's for

Twofold, but I would only rely on it for the first:

1. **Routing pages to the right OCR.** A pleading cover page goes to a model that
   analyzes the regions and extracts critical document information; a table of
   contents goes to an OCR that recognizes the structure of the headings; a
   judicial form can go to a form-trained OCR; and so on. Classify first, route
   second.
2. **Per-page metadata for chunking.** Tell the LLM working with the text "this
   is a pleading cover page, it falls on page 1", or "this is a pleading cover
   page falling on page 17, after an exhibit cover page" — stuff like that lowers
   the risk of AI docket-context mistakes.

## The one problem it actually solves

The first version of this model looked great — 98% accuracy — until I changed OCR
engines. Same pages, same model: 53% on Tesseract text, 37% on VLM text. It had
quietly memorized *how one OCR engine writes* instead of what pages say.

The fix wasn't a bigger model. It was OCRing the same 13.6k labeled pages with
**five different engines** and training on all of it:

![accuracy per OCR engine across three model generations](docs/robustness.png)

![label stability comparison](docs/stability.png)

## Specs

| | |
|---|---|
| Base | ModernBERT-base, 149.6M params, full-page input (1,536 tokens) |
| Trained on | ~13.6k human-labeled pages from 653 CA filings × 5 OCR engines (PP-OCRv5, Tesseract, docTR, Hunyuan VLM, Windows OCR) |
| Splits | document-disjoint (no filing crosses train/test) |
| Held-out test | macro-F1 0.937 · accuracy 0.974 pooled across engines (`eval/modernbert11_metrics.json`) |
| Calibration | ECE ≈ 0.02 on every engine tested — the confidence is usable for triage |
| Weights | on the Hugging Face Hub (598 MB safetensors), not in this repo |

## Use it

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

repo = "REPLACE_ME/clawbert-144"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForSequenceClassification.from_pretrained(repo).eval()

enc = tok(page_text, truncation=True, max_length=1536, return_tensors="pt")
probs = torch.softmax(model(**enc).logits, -1)[0]
print(model.config.id2label[int(probs.argmax())], float(probs.max()))
```

Or use the bundled scorer, which batches and picks a working GPU on its own:

```python
from clawbert144_infer import score_texts
labels, probs = score_texts([page1_text, page2_text])
```

## Know what you're getting

- **It assumes the PDF is already one document.** It won't work on an appendix or
  a combined record — document order is information, especially with various
  OCRs. Split first, classify second.
- **California, probably only.** California helpfully still uses the antiquated
  pleading line numbers — annoying for OCR, but it turns out to be super helpful
  for classification. I doubt it will work in other states unmodified.
- **More categories are needed and forthcoming:** appellate cover pages, and
  court-originating documents (orders, notifications, minute orders). For now
  appellate materials land in `unknown_other` — a convention, not a failure.
- Robustness is demonstrated for the five trained engines; I haven't tested it on
  other OCR models.
- `subsequent_cover_page` is genuinely ambiguous from one page alone — if you
  have whole documents, sequence models on top of these embeddings fix that.
- Text only. Anything purely visual — a struck-through "[PROPOSED]", a signature
  — is invisible to it.

## License

Apache-2.0, same as the base model.
