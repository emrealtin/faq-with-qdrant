from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import json

app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')
qdrant = QdrantClient(host="localhost", port=6333)

COLLECTION_NAME = "faq_collection"

# Koleksiyon oluştur
try:
    qdrant.get_collection(COLLECTION_NAME)
except Exception:
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

# JSON dosyasından FAQ verilerini yükle
def load_faq_data():
    try:
        with open('faq_data.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
            return [(item["question"], item["answer"]) for item in data["faqs"]]
    except FileNotFoundError:
        print("FAQ veri dosyası bulunamadı!")
        return []
    except json.JSONDecodeError:
        print("FAQ veri dosyası geçerli JSON formatında değil!")
        return []

# FAQ verilerini yükle
faq = load_faq_data()

# Soru embeddinglerini hesapla ve Qdrant'a yükle
for idx, (q, a) in enumerate(faq):
    vector = model.encode(q).tolist()
    point = PointStruct(id=idx, vector=vector, payload={"question": q, "answer": a})
    qdrant.upsert(collection_name=COLLECTION_NAME, points=[point])

class Query(BaseModel):
    question: str

@app.post("/query")
def query_api(query: Query):
    q_vector = model.encode(query.question).tolist()
    search_result = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=q_vector,
        limit=1
    )
    if search_result:
        answer = search_result[0].payload.get("answer", "Cevap bulunamadı.")
        score = search_result[0].score
        return {"answer": answer, "score": score}
    return {"answer": "Cevap bulunamadı.", "score": 0}