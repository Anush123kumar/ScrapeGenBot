# Install the required libraries
!pip install faiss-cpu transformers sentence-transformers beautifulsoup4 requests gradio

# Import necessary modules
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from bs4 import BeautifulSoup
import requests
import gradio as gr

# Load the sentence transformer model for embedding generation
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the text generation model for generating longer responses
text_generator = pipeline("text-generation", model="gpt2")

# FAISS index initialization
dim = 384  # Dimension of embeddings from 'all-MiniLM-L6-v2'
index = faiss.IndexFlatL2(dim)  # Use L2 distance for similarity search

# In-memory text store for storing scraped text
text_store = {}

# Function to scrape web pages and get the text content
def scrape_webpage(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')
    page_text = " ".join([para.get_text() for para in paragraphs])
    return page_text

# Function to add a scraped page to FAISS and store embeddings
def add_to_faiss(url):
    page_text = scrape_webpage(url)
    page_embedding = embedding_model.encode(page_text).astype('float32')  # Convert to float32 for FAISS compatibility

    # Add the embedding to FAISS index
    index.add(np.array([page_embedding]))  # FAISS expects a 2D array
    # Store the text in memory for context generation
    text_store[url] = page_text
    return f"Data loaded successfully from {url}"

# Function to generate a detailed answer from the context using GPT-2
def generate_paragraph_answer(context, question):
    max_context_length = 600  # Truncate the context to a reasonable length
    if len(context) > max_context_length:
        context = context[:max_context_length]

    input_text = f"Given the context: '{context}', answer the question: '{question}'. Answer:"
    response = text_generator(input_text, max_new_tokens=150, num_return_sequences=1)
    return response[0]['generated_text'].strip().split('Answer: ')[-1]

# Function to query FAISS for relevant embeddings and return the text
def query_faiss(question):
    question_embedding = embedding_model.encode(question).astype('float32')

    # Search for the most similar embedding in FAISS
    distances, indices = index.search(np.array([question_embedding]), k=1)  # Find top 1 match

    if distances.size > 0:
        most_relevant_url = list(text_store.keys())[indices[0][0]]  # Get the URL using the index
        context = text_store[most_relevant_url]  # Retrieve stored text using URL as key
        return generate_paragraph_answer(context, question)

    return "No relevant data found."

# Create the Gradio UI components
def load_data_gradio(url):
    return add_to_faiss(url)

def query_data_gradio(question):
    return query_faiss(question)

# Create the Gradio interface with enhanced styling
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="green")) as interface:
    # Custom CSS embedded using gr.Markdown
    gr.Markdown("""
    <style>
    #url_input, #question_input {
        font-size: 16px;
        padding: 8px;
        border-radius: 6px;
        border: 2px solid #ccc;
    }

    #load_button, #query_button {
        font-size: 16px;
        width: 150px;
        height: 45px;
        cursor: pointer;
        color: white;
        background-color: #4CAF50;
        border-radius: 8px;
        padding: 10px;
    }

    .gradio-block, .gradio-tab {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
        margin: 20px;
    }

    .gradio-markdown h2 {
        color: #4CAF50;
    }
    </style>
    """)

    gr.Markdown("""
    <div style="text-align: center; font-size: 30px; font-weight: bold; color: #333;">
        🌐 WEBSAGE 🚀
    </div>
    <div style="text-align: center; font-size: 18px; color: #555;">

    </div>
    """)

    with gr.Tab("🔗 Load Web Data"):
        gr.Markdown("### Enter a URL to scrape and store content:")
        url_input = gr.Textbox(label="Website URL", placeholder="https://example.com", elem_id="url_input", interactive=True)
        load_button = gr.Button("🚀 Load Data", elem_id="load_button")
        load_output = gr.Textbox(label="Output", placeholder="Data loading status will be shown here.", interactive=True)###########

        load_button.click(load_data_gradio, inputs=url_input, outputs=load_output)

    with gr.Tab("🤔 Ask a Question"):
        gr.Markdown("### Ask a question based on the scraped data:")
        question_input = gr.Textbox(label="Your Question", placeholder="What do you want to ask?", elem_id="question_input", interactive=True)
        query_button = gr.Button("🔍 Get Answer", elem_id="query_button")
        query_output = gr.Textbox(label="Answer", placeholder="The answer will appear here.", interactive=False)

        query_button.click(query_data_gradio, inputs=question_input, outputs=query_output)

    with gr.Tab("📊 About"):
        gr.Markdown("""
        This app leverages:
        - **FAISS** for Facebook Ai Similarity Search(VECTOR DATABASE),
        - **Sentence Transformer** for embedding generation,
        - **GPT-2** for generating detailed answers.

        """)

# Launch the Gradio interface
interface.launch(share=True)