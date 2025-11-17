import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from bson import ObjectId

from database import db, create_document, get_documents
from schemas import BlogPost

app = FastAPI(title="AI Data Scientist Blog API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "AI Blog Backend Running"}

@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response

# ---------- AI Utilities (simple heuristics, no external dependencies) ----------
class AnalyzeRequest(BaseModel):
    text: str

class AnalyzeResponse(BaseModel):
    summary: str
    tags: List[str]
    sentiment: str
    reading_time_minutes: int

KEYWORDS = {
    "machine learning": ["ml", "scikit-learn", "xgboost"],
    "deep learning": ["pytorch", "tensorflow", "cnn", "transformer"],
    "nlp": ["bert", "gpt", "token", "llm", "language"],
    "data engineering": ["airflow", "spark", "etl", "kafka"],
    "mle": ["pipeline", "deployment", "monitoring", "feature store"],
    "statistics": ["bayesian", "hypothesis", "regression", "inference"],
}

POSITIVE = ["great", "excellent", "amazing", "win", "improved", "fast", "robust", "efficient"]
NEGATIVE = ["bug", "fail", "slow", "issue", "problem", "worse", "bad", "error"]


def simple_summary(text: str, max_sentences: int = 2) -> str:
    sentences = [s.strip() for s in text.replace("\n", " ").split('.') if s.strip()]
    return '. '.join(sentences[:max_sentences]) + ('.' if sentences else '')


def extract_tags(text: str) -> List[str]:
    text_l = text.lower()
    found = set()
    for tag, terms in KEYWORDS.items():
        if any(t in text_l for t in [tag] + terms):
            found.add(tag)
    # also pick top nouns-ish words by length/frequency as fallback
    words = [w.strip('.,()[]{}:;!"\'') for w in text_l.split()]
    freq = {}
    for w in words:
        if len(w) >= 6 and w.isalpha():
            freq[w] = freq.get(w, 0) + 1
    for w, _ in sorted(freq.items(), key=lambda x: (-x[1], -len(x[0])))[:3]:
        found.add(w)
    return sorted(list(found))[:8]


def sentiment(text: str) -> str:
    tl = text.lower()
    pos = sum(t in tl for t in POSITIVE)
    neg = sum(t in tl for t in NEGATIVE)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


def reading_time(text: str) -> int:
    words = len(text.split())
    return max(1, int(round(words / 200)))


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze_text(req: AnalyzeRequest):
    sm = simple_summary(req.text)
    tg = extract_tags(req.text)
    st = sentiment(req.text)
    rt = reading_time(req.text)
    return AnalyzeResponse(summary=sm, tags=tg, sentiment=st, reading_time_minutes=rt)


# ---------- Blog Endpoints ----------
class CreatePostRequest(BaseModel):
    title: str
    content: str
    hero_image: Optional[str] = None

class PostResponse(BaseModel):
    id: str
    title: str
    content: str
    summary: Optional[str]
    tags: List[str]
    sentiment: Optional[str]
    reading_time_minutes: Optional[int]
    hero_image: Optional[str]


def _collection_name(model_cls) -> str:
    return model_cls.__name__.lower()


@app.post("/api/posts", response_model=PostResponse)
def create_post(req: CreatePostRequest):
    # Auto analyze
    analysis = analyze_text(AnalyzeRequest(text=req.content))

    post = BlogPost(
        title=req.title,
        content=req.content,
        summary=analysis.summary,
        tags=analysis.tags,
        sentiment=analysis.sentiment,
        reading_time_minutes=analysis.reading_time_minutes,
        hero_image=req.hero_image,
    )
    collection = _collection_name(BlogPost)
    inserted_id = create_document(collection, post)

    return PostResponse(
        id=inserted_id,
        title=post.title,
        content=post.content,
        summary=post.summary,
        tags=post.tags,
        sentiment=post.sentiment,
        reading_time_minutes=post.reading_time_minutes,
        hero_image=post.hero_image,
    )


@app.get("/api/posts", response_model=List[PostResponse])
def list_posts(limit: int = 20):
    collection = _collection_name(BlogPost)
    docs = get_documents(collection, limit=limit)
    results: List[PostResponse] = []
    for d in docs:
        results.append(
            PostResponse(
                id=str(d.get("_id")),
                title=d.get("title", ""),
                content=d.get("content", ""),
                summary=d.get("summary"),
                tags=d.get("tags", []),
                sentiment=d.get("sentiment"),
                reading_time_minutes=d.get("reading_time_minutes"),
                hero_image=d.get("hero_image"),
            )
        )
    return results


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
