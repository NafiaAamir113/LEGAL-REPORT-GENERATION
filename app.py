# Generate embeddings for the query
query_embedding = embedding_model.encode(query).tolist()

# Retrieve top 10 similar legal documents from Pinecone
search_results = index.query(vector=query_embedding, top_k=10, include_metadata=True)

if search_results.get("matches"):
    # Extract the text from retrieved documents
    context_chunks = [match["metadata"]["text"] for match in search_results["matches"]]

    # Use CrossEncoder to rerank the results based on query relevance
    rerank_scores = reranker.predict([(query, chunk) for chunk in context_chunks])

    # Sort by highest relevance score
    ranked_results = sorted(zip(context_chunks, rerank_scores), key=lambda x: x[1], reverse=True)

    # Use top 5 most relevant chunks as context for the final AI response
    context_text = "\n\n".join([r[0] for r in ranked_results[:5]])
