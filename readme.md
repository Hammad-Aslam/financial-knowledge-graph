# Financial Knowledge Graph

End-to-end Financial Named Entity Recognition and Knowledge Graph system — built with FinBERT, FastAPI, Next.js, and D3.js.

## What it does

Extracts named entities from financial text and builds an interactive knowledge graph showing relationships between companies, people, locations, money, and dates.

**Input:**
```
Apple CEO Tim Cook announced that Apple reported revenue of $94.9 billion in Q1 2024.
Microsoft acquired Activision Blizzard for $68.7 billion in January 2023.
Goldman Sachs headquartered in New York reported net income of $2.1 billion.
```

**Output:**
- Entities extracted with confidence scores
- Relations identified between entities
- Interactive D3.js force-directed knowledge graph

## Entity Types

| Entity | Example | Confidence |
|---|---|---|
| ORG | `Apple`, `Microsoft`, `Goldman Sachs` | ~100% |
| PER | `Tim Cook`, `Elon Musk`, `Warren Buffett` | ~99.8% |
| LOC | `New York`, `California`, `London` | ~99.8% |
| MONEY | `$94.9 billion`, `$68.7 billion` | ~90% |
| DATE | `Q1 2024`, `January 2023`, `FY2024` | ~100% |

## Relation Types

| Relation | Example |
|---|---|
| ACQUIRED | Microsoft → ACQUIRED → Activision Blizzard |
| REPORTED_REVENUE | Apple → REPORTED_REVENUE → $94.9 billion |
| LOCATED_IN | Goldman Sachs → LOCATED_IN → New York |
| CEO_OF | Tim Cook → CEO_OF → Apple |
| FINED | JPMorgan → FINED → $200 million |
| INVESTED_IN | Berkshire → INVESTED_IN → Bank of America |
| PARTNERED_WITH | Microsoft → PARTNERED_WITH → OpenAI |

## Model Performance

### V2 (current) vs V1

| Metric | V1 | V2 | Improvement |
|---|---|---|---|
| Overall F1 | 82.15% | 91.12% | +8.97% |
| PER F1 | 92.48% | 95.72% | +3.24% |
| ORG F1 | 69.22% | 81.41% | +12.19% |
| LOC F1 | 78.32% | 92.30% | +13.98% |

**V2 training data:**
- flare-ner — 408 real SEC filing sentences
- WikiANN English — 20,000 sentences
- MultiNERD English — 131,280 sentences
- Synthetic financial annotations — 1,150 sentences
- Total: **152,838 training examples**

Base model: `ProsusAI/finbert` — BERT pre-trained on 4.9B financial tokens

## Architecture

```
Financial text
      ↓
FinBERT NER (token classification)
      ↓
Entity span extraction + confidence scores
      ↓
Money/Date post-processing (decimal + magnitude fix)
      ↓
Rule-based relation extraction (13 patterns)
      ↓
NetworkX knowledge graph (nodes + edges)
      ↓
FastAPI REST API (/extract, /graph, /health)
      ↓
Next.js + D3.js force-directed graph visualization
```

## Tech Stack

| Layer | Technology |
|---|---|
| NER Model | FinBERT (HuggingFace Transformers) |
| Relation Extraction | Rule-based regex patterns |
| Knowledge Graph | NetworkX (in-memory) |
| Backend | FastAPI + uvicorn |
| Frontend | Next.js 14 + TypeScript + Tailwind |
| Graph Visualization | D3.js v7 force simulation |
| Experiment Tracking | MLflow |
| Training | PyTorch + HuggingFace Trainer on Colab T4 |
| Data Pipeline | pandas, feedparser, BeautifulSoup |

## Data Sources

| Source | Type | Size |
|---|---|---|
| SEC EDGAR (10-K filings) | Real financial filings | 15 companies |
| Reuters/Yahoo Finance RSS | Financial news | 74 articles |
| FinGPT sentiment | Financial sentences | 74,963 texts |
| Finance Alpaca | Financial QA | 56,659 texts |
| Twitter Financial News | Social finance | 9,297 texts |
| **Total corpus** | | **6.35M words** |

## Project Structure

```
financial_kg/
├── backend/
│   └── app/
│       └── main.py              ← FastAPI + FinBERT + NetworkX
├── frontend/
│   └── financial-kg-ui/
│       └── app/
│           ├── page.tsx         ← Next.js + D3.js graph UI
│           └── globals.css
├── notebooks/
│   ├── phase1_data_collection.ipynb
│   └── phase2_cleaning.ipynb
├── data/
│   ├── raw/                     ← not in git
│   ├── processed/               ← not in git
│   └── stats/
├── models/                      ← not in git (too large)
│   └── financial_ner_finbert_v2/
│       ├── model.safetensors
│       ├── config.json
│       ├── tokenizer.json
│       └── label_map.json
├── mlruns/                      ← not in git
├── scripts/
└── reports/
```

## Setup

### 1 — Create conda environment

```bash
conda create -n ner_env python=3.10 -y
conda activate ner_env
pip install fastapi uvicorn transformers torch networkx sentencepiece
```

### 2 — Download model weights

Train using Colab notebook (see `notebooks/`) or download pre-trained weights and place at:
```
models/financial_ner_finbert_v2/
```

### 3 — Start MLflow

```bash
cd G:\financial_kg
mlflow server --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:///mlruns/mlflow.db \
  --default-artifact-root ./mlruns/artifacts
```

MLflow UI → http://127.0.0.1:5000

### 4 — Start FastAPI backend

```bash
cd backend/app
uvicorn main:app --reload --host 127.0.0.1 --port 8002
```

API docs → http://127.0.0.1:8002/docs

### 5 — Start Next.js frontend

```bash
cd frontend/financial-kg-ui
npm install
npm run dev -- --port 3001
```

UI → http://localhost:3001

## API Reference

### GET /health

```json
{
  "status": "ok",
  "model": "FinBERT NER + NetworkX KG",
  "version": "2.0.0",
  "entities": ["PER", "ORG", "LOC", "MONEY", "DATE"]
}
```

### POST /extract

Request:
```json
{
  "text": "Apple CEO Tim Cook reported $94.9 billion revenue in Q1 2024."
}
```

Response:
```json
{
  "entities": [
    {"text": "Apple",         "label": "ORG",   "confidence": 100.0},
    {"text": "Tim Cook",      "label": "PER",   "confidence": 99.8},
    {"text": "$94.9 billion", "label": "MONEY", "confidence": 90.4},
    {"text": "Q1 2024",       "label": "DATE",  "confidence": 100.0}
  ],
  "entity_count": 4
}
```

### POST /graph

Request:
```json
{
  "text": "Microsoft acquired Activision Blizzard for $68.7 billion in January 2023."
}
```

Response:
```json
{
  "nodes": [
    {"id": "Microsoft",           "type": "ORG"},
    {"id": "Activision Blizzard", "type": "ORG"},
    {"id": "$68.7 billion",       "type": "MONEY"},
    {"id": "January 2023",        "type": "DATE"}
  ],
  "edges": [
    {"source": "Microsoft", "target": "Activision Blizzard", "relation": "ACQUIRED"}
  ],
  "stats": {"nodes": 4, "edges": 1, "density": 0.083}
}
```

## Training on Google Colab

The model was trained on Google Colab free tier (T4 GPU):

```
Training time  : ~50 minutes
GPU            : Tesla T4
Epochs         : 5
Batch size     : 32
Learning rate  : 2e-5
FP16           : enabled
Best epoch     : 5 (F1 = 91.12%)
```

## MLflow Experiment Tracking

All experiments tracked with MLflow:
- Phase 1: Data collection metrics (corpus size, sources, word counts)
- Phase 2: Data cleaning metrics (before/after quality)
- Phase 3: Model training (loss curves, F1 per epoch, per-entity metrics)

## Author

Hammad Aslam — Data Engineer + ML Engineer

2 years Data Engineering experience + specializations in Machine Learning, Deep Learning, NLP, GANs, PyTorch