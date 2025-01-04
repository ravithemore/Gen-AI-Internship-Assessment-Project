import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import google.generativeai as genai

# Qdrant details
QDRANT_URL = "https://807708a6-1d41-4ecb-a1f3-8a41fcd48ec3.us-east4-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "J3LJcoG3q_njIvu9OzjooR2VBD-tx_Zz553gGwMoUD_xzdYz1tFufA"
QDRANT_COLLECTION_NAME = "courses-data"

# Google Gemini API details
GEMINI_API_KEY = "AIzaSyCUEWMuE3mPH3Aui55MK0PKaJxYyBPrFMY"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# Initialize Qdrant client
qdrant_client = QdrantClient(url=QDRANT_URL, prefer_grpc=True, api_key=QDRANT_API_KEY)

# Load the SentenceTransformer model
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Vector size = 384


def vector_search(query, collection_name, top_k):
    """Perform a vector search on the Qdrant collection."""
    query_vector = embedder.encode(query).tolist()
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k
    )
    results = []
    for result in search_result:
        chunk_text = result.payload.get('page_content', 'No text found')
        results.append(chunk_text)
    return results


def gemini(query, chunks):
    """Generates an answer using Google's Generative AI (Gemini)."""
    context = "\n".join([f"{i+1}. {chunk}" for i, chunk in enumerate(chunks)])
    prompt = f"""
    You are a highly knowledgeable assistant. Based on the given context, please provide a well-crafted answer to the query below. Use the provided information from the context as reference material.
    
    ### Context:
    {context}
    
    ### Query:
    {query}
    
    Based on the context, provide a list of courses - course names and a short description.
    Provide a concise, clear, and informative response based on the query.
    """
    # Make the request to generate text
    response = model.generate_content(prompt)

    # Check if the response contains valid content
    if response.candidates and len(response.candidates) > 0:
        return response.text    # Return the generated text as a string
    else:
        return "No valid content was returned. Please adjust your prompt or try again."


def getResult(input_query):
    context = vector_search(input_query, QDRANT_COLLECTION_NAME, top_k=5)
    return gemini(input_query, context)


# Streamlit App
st.title("Course Finder using RAG System")
st.write("Search for courses using a query. The system retrieves and generates relevant course details.")

# Search bar for user input
query = st.text_input("Enter your query:", "")

# Display the result when the user enters a query
if st.button("Search"):
    if query.strip():
        with st.spinner("Searching and generating results..."):
            result = getResult(query)
        st.subheader("Results:")
        st.write(result)
    else:
        st.warning("Please enter a valid query!")
