# ScrapeGenBot
ScrapeGenBot is an AI-based project that combines web scraping, vector embeddings, and generative AI to extract and analyze web content. The tool enables users to scrape data from any webpage, store it in a vector database (FAISS), and query the stored data with questions, generating insightful answers using GPT-2.

Table of Contents
----Installation
----Usage
Modules
1. Web Scraping
2. Embedding Generation
3. FAISS Vector Database
4. Generative AI
5. Gradio User Interface
----API Endpoints
----Technologies
----License

   Modules
1. Web Scraping
Functionality: Scrapes text content from a specified webpage using BeautifulSoup.
Key Function: scrape_webpage(url) - This function takes a URL as input and returns the scraped text content by extracting all paragraph elements.

2. Embedding Generation
Functionality: Converts the scraped text into vector embeddings for similarity search.
Key Class: SentenceTransformer from the sentence-transformers library is used to generate embeddings.
Key Function: add_to_faiss(url) - This function scrapes the webpage and adds the embeddings to the FAISS index.

3. FAISS Vector Database
Functionality: Manages the storage and retrieval of vector embeddings using FAISS.
Key Class: faiss.IndexFlatL2 is used for efficient nearest neighbor search based on L2 distance.
Key Function: query_faiss(question) - This function retrieves relevant embeddings for a given question and finds the most similar stored text.

4. Generative AI
Functionality: Generates answers to user queries based on the retrieved context.
Key Class: The pipeline function from the transformers library is used to load the GPT-2 model for text generation.
Key Function: generate_paragraph_answer(context, question) - This function takes a context and a question, and returns a generated answer.

5. Gradio User Interface
Functionality: Provides a user-friendly interface to interact with the WebSage-Bot.
Key Functions: The Gradio interface is set up with tabs for loading data and asking questions, making it easy for users to interact with the bot.


API Endpoints
Load Data:

Endpoint: /load
Method: POST
Description: Accepts a URL, scrapes the content, processes it, and loads it into the vector database.
Query Data:

Endpoint: /query
Method: POST
Description: Accepts a user question, retrieves relevant data, and generates an answer using the generative AI model.
Technologies
Python: Core programming language.
BeautifulSoup: For web scraping.
Sentence Transformers: For embedding generation.
FAISS: For efficient vector similarity search.
GPT-2: For generating detailed responses.
Gradio: To create a user-friendly web interface.


License
This project is licensed under the MIT License - see the LICENSE file for details.
