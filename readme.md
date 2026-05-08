Here are all the commands — open 4 separate Anaconda Prompt windows:

**Window 1 — MLflow:**
```
conda activate ner_env
cd G:\financial_kg
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlruns/mlflow.db --default-artifact-root ./mlruns/artifacts
```

**Window 2 — Financial NER FastAPI:**
```
conda activate ner_env
cd G:\financial_kg\backend\app
uvicorn main:app --reload --host 127.0.0.1 --port 8002
```

**Window 3 — Financial KG Frontend:**
```
cd G:\financial_kg\frontend\financial-kg-ui
npm run dev -- --port 3001
```

**Window 4 — Medical NER FastAPI (if needed):**
```
conda activate ner_env
cd G:\multi_domain_ner\backend\app
uvicorn main:app --reload --host 127.0.0.1 --port 8001
```

**Window 5 — Medical NER Frontend (if needed):**
```
cd G:\multi_domain_ner\frontend\ner-ui
npm run dev
```

**All services:**
```
MLflow UI         → http://127.0.0.1:5000
Financial API     → http://127.0.0.1:8002
Financial API docs→ http://127.0.0.1:8002/docs
Financial UI      → http://localhost:3001
Medical API       → http://127.0.0.1:8001
Medical UI        → http://localhost:3000
```

Send a screenshot when all are running and we continue.