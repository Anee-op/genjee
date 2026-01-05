from django.shortcuts import render
from django.http import Http404
import os

import google.generativeai as genai
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction


# ---------------- ENV SETUP ----------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY / GOOGLE_API_KEY not set")

genai.configure(api_key=GEMINI_API_KEY)


# ---------------- COLLEGES ----------------

AVAILABLE_COLLEGES = {
    "nit-hamirpur": "NIT Hamirpur",
}


# ---------------- HOME VIEW ----------------

def home_view(request):
    return render(
        request,
        "home.html",
        {
            "colleges": AVAILABLE_COLLEGES,
            "page_title": "JEECollege RAG - Select a College",
        },
    )


# ---------------- CHROMA SETUP ----------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma")

chroma_client = chromadb.Client(
    Settings(
        persist_directory=CHROMA_DIR,
        anonymized_telemetry=False
    )
)

# ---------------- RAG CORE ----------------

def generate_rag_response(user_query: str, college_slug: str) -> str:
    try:
        collection = chroma_client.get_collection (
            name=college_slug,
            embedding_function=GoogleGenerativeAiEmbeddingFunction(
                api_key=GEMINI_API_KEY,
                model_name="models/text-embedding-004",
            ),
        )

        results = collection.query(
            query_texts=[user_query],
            n_results=3,
            include=["documents"],
        )

        retrieved_docs = results["documents"][0]

        if not retrieved_docs:
            return "No relevant information found."

    except Exception as e:
        print("ChromaDB error:", e)
        return "Database error."

    context_text = "\n---\n".join(retrieved_docs)

    system_prompt = (
        f"You are an assistant answering questions about "
        f"{AVAILABLE_COLLEGES[college_slug]} using ONLY the provided context."
    )

    prompt = f"""
Question:
{user_query}

Context:
{context_text}

Answer clearly and factually using only the context above.
"""

    try:
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=system_prompt,
        )

        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=512,
            ),
        )

        return response.text.strip()

    except Exception as e:
        print("Gemini error:", e)
        return "AI generation failed."


# ---------------- QA VIEW ----------------

def college_qa_view(request, college_slug):
    college_name = AVAILABLE_COLLEGES.get(college_slug)
    if not college_name:
        raise Http404("College not found")

    user_question = ""
    ai_answer = None

    if request.method == "POST":
        user_question = request.POST.get("question", "").strip()
        if user_question:
            ai_answer = generate_rag_response(user_question, college_slug)

    return render(
        request,
        "qa.html",
        {
            "college_name": college_name,
            "college_slug": college_slug,
            "user_question": user_question,
            "ai_answer": ai_answer,
        },
    )

    
  