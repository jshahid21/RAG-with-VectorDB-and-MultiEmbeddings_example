# RAG-with-VectorDB-and-MultiEmbeddings

## Retrieval Augmented Generation (RAG) with Vector Databases and Multiple Embedding Models

RAG is a powerful technique that combines information retrieval with Large Language Models (LLMs) to enhance their responses with up-to-date and factual external knowledge.

## Overview

In this repo, we explore:

1.  **Text Embeddings**: The fundamental concept of converting text into numerical vectors that capture semantic meaning. We demonstrate this using:
    *   **OpenAI's `text-embedding-ada-002`**
    *   **Cohere's embedding models**
    *   **Hugging Face's `SentenceTransformer` models** (e.g., `all-mpnet-base-v2`)
2.  **Vector Databases/Similarity Search**: How libraries like **FAISS (Facebook AI Similarity Search)** enable efficient storage and retrieval of these high-dimensional embedding vectors to find relevant information quickly.
3.  **Retrieval Augmented Generation (RAG) Pattern**: We implement an end-to-end RAG system where:
    *   A user query is first embedded.
    *   This query embedding is used to retrieve semantically similar documents from a FAISS index.
    *   The retrieved documents (their original text content) are then provided as context to an LLM (e.g., OpenAI's `gpt-4o-mini`) to generate a more accurate, grounded, and informed response, mitigating issues like hallucination.

This code is ideal for aspiring AI/ML Engineers looking to build a solid foundation in modern NLP and Generative AI applications.

## Setup and Installation

To run this notebook, you'll need Python and `pip` installed. It's recommended to run this in a Google Colab environment for ease of setup and access to GPUs (though FAISS CPU is used here).

1.  **Clone the repository (if applicable) or open the notebook in Google Colab.**
2.  **Install necessary libraries:**
    ```bash
    !pip install faiss-cpu openai python-dotenv cohere sentence-transformers numpy
    ```

3.  **API Key Configuration:**
    *   **OpenAI**: Obtain an OpenAI API key. Store it securely in a `.env` file in the same directory as your notebook (e.g., `OPENAI_API_KEY="sk-..."`). The notebook uses `dotenv` to load this key.
    *   **Cohere**: Obtain a Cohere API key. In Google Colab, it's recommended to store this key in Colab's built-in **Secrets** manager (click the '🔑' icon on the left panel, add a new secret named `COHERE_KEY`). The notebook accesses this using `from google.colab import userdata`.

## Usage

Once the environment is set up and API keys are configured, simply run through the cells in the notebook sequentially.

The notebook flow is as follows:

1.  **Environment Setup**: Installs libraries and loads API keys.
2.  **Initial LLM Interaction**: Demonstrates an LLM's limitations without external knowledge.
3.  **OpenAI Embeddings & FAISS**: Generates embeddings for sample documents, builds a FAISS index, and performs semantic search.
4.  **RAG Implementation**: Combines retrieval (using embeddings + FAISS) with LLM generation to answer questions based on provided context.
5.  **Alternative Embeddings**: Shows how to use Cohere and Hugging Face `SentenceTransformer` models to generate embeddings.

Experiment with different queries and observe how the RAG system retrieves relevant information and uses it to formulate responses.
