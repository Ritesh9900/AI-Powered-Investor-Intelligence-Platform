import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from azure.search.documents.models import VectorizedQuery
from langchain_openai import AzureOpenAIEmbeddings

from vectorstore.azure_ai_search import AzureAISearchVectorStore
from llm.azure_openai import get_openai_client

router = APIRouter()


class ChatRequest(BaseModel):
    question: str
    company: str | None = None
    year: int | None = None


@router.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Initialize vector store
        vector_store = AzureAISearchVectorStore(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            api_key=os.getenv("AZURE_SEARCH_API_KEY"),
            index_name=os.getenv("AZURE_SEARCH_INDEX_NAME")
        )

        embeddings = AzureOpenAIEmbeddings(
            model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_EMBEDDING_VERSION", "2023-05-15")
        )

        # Build filter expression
        filters = []
        if request.company:
            filters.append(f"company eq '{request.company}'")
        if request.year:
            filters.append(f"year eq '{request.year}'")
        filter_expr = " and ".join(filters) if filters else None

        # Generate query vector for hybrid search
        query_vector = embeddings.embed_query(request.question)
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=10,
            fields="content_vector"
        )

        # Perform Hybrid Search (Vector + Keyword)
        results = vector_store.client.search(
            search_text=request.question,
            vector_queries=[vector_query],
            filter=filter_expr,
            top=10
        )

        # Deduplicate retrieved chunks
        docs = []
        seen = set()
        for r in results:
            content = r.get("content", "").strip()
            if content and content not in seen:
                seen.add(content)
                docs.append(content)

        context = "\n\n".join(docs)

        # Build chat prompt – include retrieved context and the user question
        prompt = (
            f"You are an expert financial analyst. Use the following context from corporate reports to answer "
            f"the user's question. If the context does not contain relevant information, politely indicate that "
            f"you do not have enough data.\n\nContext:\n{context}\n\nUser Question: {request.question}\n\nAnswer:"
        )

        client = get_openai_client()
        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
