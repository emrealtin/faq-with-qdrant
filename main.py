from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

app = FastAPI()
# Gelen soruları embeddinge çevirmek için sentence transformer modeli kullanıldı
model = SentenceTransformer('all-MiniLM-L6-v2')

# Qdrant client
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

# Örnek soru-cevaplar
faq = [
    ("Siparişim nerede?", "Siparişiniz kargoya verildi."),
    ("Ürünüm hasar gördü.", "Hasar için lütfen destekle iletişime geçin."),
    ("İade süreci nasıl işliyor?", "İade için form doldurmanız gerekiyor."),
]

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