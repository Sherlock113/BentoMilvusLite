import bentoml
import os

BENTO_EMBEDDING_MODEL_END_POINT = "your_embedding_endpoint_here"

embedding_client = bentoml.SyncHTTPClient(BENTO_EMBEDDING_MODEL_END_POINT)

# Function to chunk text
def chunk_text(filename: str) -> list:
    with open(filename, "r") as f:
        text = f.read()
    sentences = text.split("\n")
    return sentences

# Process files and chunk text
cities = os.listdir("city_data")
city_chunks = []
for city in cities:
    chunked = chunk_text(f"city_data/{city}")
    cleaned = [chunk for chunk in chunked if len(chunk) > 7]
    mapped = {
        "city_name": city.split(".")[0],
        "chunks": cleaned
    }
    city_chunks.append(mapped)

# Function to get embeddings
def get_embeddings(texts: list) -> list:
    if len(texts) > 25:
        splits = [texts[x:x+25] for x in range(0, len(texts), 25)]
        embeddings = []
        for split in splits:
            embedding_split = embedding_client.encode(sentences=split)
            if embedding_split.size > 0:  # Check if the split is not empty
                embeddings.extend(embedding_split)
            else:
                print(f"Warning: No embeddings returned for split: {split}")
        return embeddings
    return embedding_client.encode(sentences=texts)

# Prepare entries with embeddings
entries = []
for city_dict in city_chunks:
    embedding_list = get_embeddings(city_dict["chunks"])
    for i, embedding in enumerate(embedding_list):
        entry = {
            "embedding": embedding,
            "sentence": city_dict["chunks"][i],
            "city": city_dict["city_name"]
        }
        entries.append(entry)

# Print entries to verify
# print(entries)

from pymilvus import MilvusClient, DataType

COLLECTION_NAME = "Bento_Milvus_RAG"
DIMENSION = 384

# Initialize a Milvus Lite client
milvus_client = MilvusClient("./milvus_demo.db")

# Create schema
schema = MilvusClient.create_schema(
    auto_id=True,
    enable_dynamic_field=True,
)
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=DIMENSION)

# Prepare index parameters and add index
index_params = milvus_client.prepare_index_params()
index_params.add_index(
    field_name="embedding",
    index_type="AUTOINDEX",
    metric_type="COSINE",
)

# Create collection and insert data
milvus_client.create_collection(
    collection_name=COLLECTION_NAME,
    schema=schema,
    index_params=index_params
)
milvus_client.insert(collection_name=COLLECTION_NAME, data=entries)
