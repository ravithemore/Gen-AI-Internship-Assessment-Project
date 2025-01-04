# Smart Search Tool for Analytics Vidhya Free Courses

## Objective
The project aims to create an intelligent search tool to help users efficiently find relevant free courses on the Analytics Vidhya platform. By leveraging a Retrieval-Augmented Generation (RAG) approach, the tool interprets natural language queries or keywords to deliver precise and meaningful results. It is publicly deployed on Huggingface Spaces for evaluation.

---

## Approach and Methodology

### 1. Data Collection
- **Process**: Scraped or manually collected data such as course titles, descriptions, and curriculum from Analytics Vidhya's free courses section.
- **Storage**: Stored the course data in a structured format as text files for further processing.

---

### 2. System Design
The tool consists of several components to ensure effective search functionality:

#### A. Data Chunking and Embedding
- Split course data into manageable chunks using **RecursiveCharacterTextSplitter** from LangChain, ensuring each chunk is concise yet contextually complete.
- Generated vector embeddings for the chunks using the **SentenceTransformer model (all-MiniLM-L6-v2)**, known for its compact size and high-quality representations.

#### B. Vector Database
- **Qdrant** was used as the vector database to store embeddings. It enables efficient vector similarity searches, ensuring rapid retrieval of relevant data.

#### C. Retrieval Mechanism
- Developed a vector similarity search with Qdrant’s API to retrieve the top 5 most relevant chunks based on cosine similarity for a given user query.

#### D. Generative Model for Response
- Integrated a generative AI model like **Google's Gemini API** or **OpenAI's GPT-3.5** to process retrieved chunks and user queries, generating clear, user-friendly responses.

#### E. Deployment
- The tool was deployed using **Streamlit** on Huggingface Spaces for accessible and interactive user experiences.

---

## Implementation Details

### Libraries and Tools
- **LangChain**: For text chunking and managing the RAG pipeline.
- **SentenceTransformers**: To generate embeddings for text data.
- **Qdrant**: For storing and retrieving embeddings efficiently.
- **Streamlit**: To build the user-friendly frontend interface.
- **Google Gemini API**: (Optional) To enhance the quality of generated responses.

---

### Key Features

#### 1. Data Processing
- **`read_txt_files(directory)`**: Reads and processes course data from structured text files.
- **`upload_chunks_to_QDrant(documents, collectionName)`**: Creates embeddings for text chunks and uploads them to the Qdrant vector database.

#### 2. Search and Retrieval
- **`vector_search(query, collection_name, top_k)`**: Retrieves the top `k` most relevant chunks based on the similarity score from Qdrant.

#### 3. Generative Answering
- **`gemini(query, chunks)`**: Generates concise, human-like responses based on the user query and the retrieved chunks.

#### 4. Frontend Interaction
- Built with **Streamlit**, enabling users to input queries and view results seamlessly.

#### 5. Deployment
- Hosted on Huggingface Spaces for public accessibility and easy user interaction.

---

## How It Works

1. **Input Query**: Users enter a natural language query or keyword (e.g., "Python course for beginners").
2. **Search Process**:
   - The system retrieves the most relevant course information chunks from the vector database.
   - The generative model processes the retrieved data to create a coherent response.
3. **Output**: The tool displays a list of relevant courses with brief descriptions and links to full course details.

---

## Methodology and Models

### Embedding Model
- **SentenceTransformer (all-MiniLM-L6-v2)**:
  - Compact size with a vector dimension of 384.
  - Optimized for sentence-level similarity tasks, ensuring quick retrieval and efficient storage.

### Generative Model
- **Google Gemini API** (Optional: OpenAI GPT-3.5):
  - Processes contextual information from retrieved chunks and user queries.
  - Delivers high-quality, natural language responses.

---

## Evaluation Metrics

1. **Relevance**: Measures how closely the retrieved chunks match the user’s query.
2. **Clarity**: Assesses the coherence and informativeness of generated responses.
3. **Speed**: Evaluates the responsiveness of the search and generation process.

---

## Future Enhancements

1. **Direct Course Links**: Automatically include clickable links to course pages in search results.
2. **Dynamic Chunking**: Optimize chunk sizes dynamically to capture course details more effectively.
3. **Advanced Query Understanding**: Incorporate query expansion and other techniques for better search accuracy.

---

This tool represents a significant step toward improving user experiences in navigating the rich library of free courses on Analytics Vidhya. Its modular design, efficient architecture, and reliance on advanced technologies ensure scalability and robust performance.
