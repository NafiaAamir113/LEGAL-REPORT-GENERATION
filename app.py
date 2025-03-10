import streamlit as st
import requests
import pinecone
from sentence_transformers import SentenceTransformer, CrossEncoder

# Streamlit page setup
st.set_page_config(page_title="Legal RAG System", layout="wide")

# Load secrets
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
TOGETHER_AI_API_KEY = st.secrets["TOGETHER_AI_API_KEY"]

# Pinecone setup
INDEX_NAME = "lawdata-2-index"
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# Check if index exists
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    st.error(f"❌ Index '{INDEX_NAME}' not found.")
    st.stop()

# Initialize Pinecone index
index = pc.Index(INDEX_NAME)

# Load embedding models
embedding_model = SentenceTransformer("BAAI/bge-large-en")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

st.title("📚 Legal Retrieval-Augmented Generation (RAG) System")
query = st.text_input("🔍 Enter your legal question:")

if query:
    with st.spinner("🔎 Searching..."):
        query_embedding = embedding_model.encode(query, normalize_embeddings=True).tolist()

        # ✅ Query Pinecone with error handling
        try:
            search_results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
        except Exception as e:
            st.error(f"❌ Pinecone query failed: {e}")
            st.stop()

        if not search_results or "matches" not in search_results or not search_results["matches"]:
            st.warning("No relevant results found. Try rephrasing your query.")
            st.stop()

        # Extract text chunks from results
        context_chunks = [match["metadata"]["text"] for match in search_results["matches"]]

        # Rerank results
        rerank_scores = reranker.predict([(query, chunk) for chunk in context_chunks])
        ranked_results = sorted(zip(context_chunks, rerank_scores), key=lambda x: x[1], reverse=True)

        context_text = "\n\n".join([r[0] for r in ranked_results[:5]])

        # Construct LLM prompt
        # prompt = f"""You are a legal assistant. Answer the question based on the retrieved legal documents.
        prompt = f"""You are a legal assistant. Given the retrieved legal documents, provide a detailed answer.


        Context:
        {context_text}

        Question: {query}

        Answer:"""
        

        # Query Together AI
        response = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers={"Authorization": f"Bearer {TOGETHER_AI_API_KEY}", "Content-Type": "application/json"},
            json={"model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                  "messages": [{"role": "system", "content": "You are an expert in legal matters."},
                               {"role": "user", "content": prompt}], "temperature": 0.2}
        )

        answer = response.json()["choices"][0]["message"]["content"]
        st.success("💡 AI Response:")
        st.write(answer)

st.markdown("🚀 Built with **Streamlit**, **Pinecone**, and **Llama-3.3-70B-Turbo** on **Together AI**.")




