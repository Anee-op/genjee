
from django.shortcuts import render
import os
from google.generativeai import genai
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction


# ---------------- COLLEGES ----------------

AVAILABLE_COLLEGES = {
    'nit-hamirpur': 'NIT Hamirpur',
    
    
}

# ---------------- HOME VIEW ----------------

def home_view(request):
    context = {

        'colleges': AVAILABLE_COLLEGES,
        'page_title': "JEECollege RAG - Select a College"
    }
    return render(request, 'home.html', context)

# ---------------- GEMINI SETUP ----------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")



if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not set")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ---------------- CHROMA SETUP ----------------
chroma_client = chromadb.Client(
    Settings(
        persist_directory="./chroma",
        anonymized_telemetry=False
    )
)

# ---------------- RAG FUNCTION ----------------

def generate_rag_response(user_query: str, college_slug: str) -> str:

    try:
        print("Collections:", chroma_client.list_collections())

        
        collection = chroma_client.get_collection(
    name=college_slug,
    embedding_function=GoogleGenerativeAiEmbeddingFunction(
        api_key=GEMINI_API_KEY,
        model_name="models/text-embedding-004"
    ),
    
)


        results = collection.query(
            query_texts=[user_query],
            n_results=2,
            include=["documents"]
        )

        retrieved_documents = results["documents"][0]

        if not retrieved_documents:
            return "No relevant data found in the database."

    except Exception as e:
        print("Chroma error:", e)
        return "Database retrieval failed."

    context_text = "\n---\n".join(retrieved_documents)

    system_prompt = (
        f"You answer questions about {AVAILABLE_COLLEGES[college_slug]} "
        "using ONLY the given context."
    )

    augmented_prompt = f"""
Question: {user_query}

Context:
{context_text}

Answer clearly using only the context.
"""

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=augmented_prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.2
            )
        )
        return response.text

    except Exception as e:
        print("Gemini error:", e)
        return "AI generation failed."

# ---------------- QA VIEW ----------------

def college_qa_view(request, college_slug):

    college_name = AVAILABLE_COLLEGES.get(college_slug)
    if not college_name:
        from django.http import Http404
        raise Http404("College not found")

    user_question = ""
    ai_answer = None

    if request.method == "POST":
        user_question = request.POST.get("question", "").strip()
        if user_question:
            ai_answer = generate_rag_response(user_question, college_slug)

    context = {
        "college_name": college_name,
        "college_slug": college_slug,
        "user_question": user_question,
        "ai_answer": ai_answer
    }

    return render(request, "qa.html", context)



    
  