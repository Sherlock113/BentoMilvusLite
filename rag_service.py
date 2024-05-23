import bentoml
from embedding_service import get_embeddings, milvus_client, COLLECTION_NAME

BENTO_LLM_END_POINT = "your_llm_endpoint_here"
llm_client = bentoml.SyncHTTPClient(BENTO_LLM_END_POINT)

def dorag(question: str, context: str):
    prompt = (f"You are a helpful assistant. The user has a question. Answer the user question based only on the context: {context}. \n"
              f"The user question is {question}")
    results = llm_client.generate(
        max_tokens=1024,
        prompt=prompt,
    )
    return "".join(results)

# Function to ask a question
def ask_a_question(question):
    embeddings = get_embeddings([question])
    res = milvus_client.search(
        collection_name=COLLECTION_NAME,
        data=embeddings,
        anns_field="embedding",
        limit=5,
        output_fields=["sentence"]
    )

    sentences = [hit['entity']['sentence'] for hits in res for hit in hits]
    context = ". ".join(sentences)
    return context

# Example question
question = "What state is Cambridge in?"
context = ask_a_question(question=question)
print(dorag(question=question, context=context))
