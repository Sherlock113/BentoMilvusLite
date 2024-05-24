# BentoMilvusLite

## Install dependencies

```bash
pip install -U pymilvus bentoml
```

## Run the RAG app

1. Clone the entire repo.
2. Deploy an embedding and a large language model on BentoCloud, then retrieve the `BENTO_EMBEDDING_MODEL_END_POINT` and `BENTO_LLM_END_POINT`.
3. Run the following:

   ```bash
   python rag_service.py
   ```
