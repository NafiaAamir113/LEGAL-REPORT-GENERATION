# import streamlit as st
# import requests
# import pinecone
# import torch  # ✅ Import torch to check for GPU
# from sentence_transformers import SentenceTransformer, CrossEncoder

# st.set_page_config(page_title="Legal RAG System", layout="wide")

# PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
# PINECONE_ENV = st.secrets["PINECONE_ENV"]
# TOGETHER_AI_API_KEY = st.secrets["TOGETHER_AI_API_KEY"]

# INDEX_NAME = "lawdata-2-index"
# pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# if INDEX_NAME not in [index_info["name"] for index_info in pc.list_indexes()]:
#     st.error(f"❌ Index '{INDEX_NAME}' not found.")
#     st.stop()

# # ✅ Move models to GPU if available
# device = "cuda" if torch.cuda.is_available() else "cpu"
# embedding_model = SentenceTransformer("BAAI/bge-large-en", device=device)
# reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)  # ✅ Move reranker to GPU

# st.title("📚 Legal Retrieval-Augmented Generation (RAG) System")

# query = st.text_input("🔍 Enter your legal question:")

# if query:
#     with st.spinner("🔎 Searching..."):
#         query_embedding = embedding_model.encode(query, convert_to_tensor=True).cpu().tolist()  # ✅ Ensure tensor conversion
#         search_results = index.query(vector=query_embedding, top_k=10, include_metadata=True)

#         if search_results.get("matches"):
#             context_chunks = [match["metadata"]["text"] for match in search_results["matches"]]
#             rerank_scores = reranker.predict([(query, chunk) for chunk in context_chunks])

#             ranked_results = sorted(zip(context_chunks, rerank_scores), key=lambda x: x[1], reverse=True)
#             context_text = "\n\n".join([r[0] for r in ranked_results[:5]])

#             prompt = f"""You are a legal assistant. Answer the question based on the retrieved legal documents.

#             Context:
#             {context_text}

#             Question: {query}

#             Answer:"""

#             response = requests.post(
#                 "https://api.together.xyz/v1/chat/completions",
#                 headers={"Authorization": f"Bearer {TOGETHER_AI_API_KEY}", "Content-Type": "application/json"},
#                 json={"model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
#                       "messages": [{"role": "system", "content": "You are an expert in legal matters."},
#                                    {"role": "user", "content": prompt}], "temperature": 0.2}
#             )

#             answer = response.json()["choices"][0]["message"]["content"]
#             st.success("💡 AI Response:")
#             st.write(answer)

# st.markdown("🚀 Built with **Streamlit**, **Pinecone**, and **Llama-3.3-70B-Turbo** on **Together AI**.")


import streamlit as st
import requests
import pinecone
import torch  
from sentence_transformers import SentenceTransformer, CrossEncoder

# ✅ Streamlit page config
st.set_page_config(page_title="Legal RAG System", layout="wide")

# ✅ Load API keys securely
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENV"]
TOGETHER_AI_API_KEY = st.secrets["TOGETHER_AI_API_KEY"]

# ✅ Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# ✅ Define the index name
INDEX_NAME = "lawdata-2-index"

# ✅ Check if index exists before querying
if INDEX_NAME not in pinecone.list_indexes():
    st.error(f"❌ Pinecone index '{INDEX_NAME}' not found. Please check your Pinecone dashboard.")
    st.stop()

# ✅ Initialize Pinecone index correctly
index = pinecone.Index(INDEX_NAME)

# ✅ Move models to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer("BAAI/bge-large-en", device=device)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)

st.title("📚 Legal Retrieval-Augmented Generation (RAG) System")
query = st.text_input("🔍 Enter your legal question:")

if query:
    with st.spinner("🔎 Searching..."):
        # ✅ Encode query properly
        query_embedding = embedding_model.encode(query, convert_to_tensor=True).cpu().numpy()

        # ✅ Query Pinecone with proper index reference
        search_results = index.query(vector=query_embedding, top_k=10, include_metadata=True)

        if search_results.get("matches"):
            # ✅ Extract text chunks
            context_chunks = [match["metadata"]["text"] for match in search_results["matches"]]

            # ✅ Rerank results using CrossEncoder
            rerank_scores = reranker.predict([(query, chunk) for chunk in context_chunks])

            # ✅ Sort results based on reranking scores
            ranked_results = sorted(zip(context_chunks, rerank_scores), key=lambda x: x[1], reverse=True)
            context_text = "\n\n".join([r[0] for r in ranked_results[:5]])

            # ✅ Construct AI prompt
            prompt = f"""You are a legal assistant. Answer the question based on the retrieved legal documents.

            Context:
            {context_text}

            Question: {query}

            Answer:"""

            # ✅ Call Together AI
            response = requests.post(
                "https://api.together.xyz/v1/chat/completions",
                headers={"Authorization": f"Bearer {TOGETHER_AI_API_KEY}", "Content-Type": "application/json"},
                json={"model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                      "messages": [{"role": "system", "content": "You are an expert in legal matters."},
                                   {"role": "user", "content": prompt}], "temperature": 0.2}
            )

            # ✅ Parse response
            answer = response.json()["choices"][0]["message"]["content"]

            st.success("💡 AI Response:")
            st.write(answer)

st.markdown("🚀 Built with **Streamlit**, **Pinecone**, and **Llama-3.3-70B-Turbo** on **Together AI**.")
