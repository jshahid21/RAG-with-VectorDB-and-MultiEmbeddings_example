!pip install faiss-cpu

documents = [

             "The Eiffel Tower is in Hyderabad",
             "Python is popular programming language",
             "OpenAI develops advanced AI models like GPT",
             "The sun rises in the east and sets in the west"
]

import numpy as np
import os
import faiss
import openai
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)
response = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
    {
      "role": "user",
      "content": "can you give me  updates on the current IPL status"
    }
  ]
)
response.choices[0].message.content

def get_embeddings(text):
  response = openai.Embedding.create(
    input=text,
    model="text-embedding-ada-002"
  )
  return response['data'][0]['embedding']

doc_embeddings = [get_embeddings(doc) for doc in documents]

len(doc_embeddings)

embedding_dim = len(doc_embeddings[0])
embedding_dim

index = faiss.IndexFlatL2(embedding_dim)
index.add(np.array(doc_embeddings).astype('float32'))

def retrive_similar_docs(query,k=2):
  query_embedding = np.array(get_embeddings(query)).astype('float32').reshape(1,-1)
  distances, indices = index.search(query_embedding, k)
  return [documents[i] for i in indices[0]]

retrive_similar_docs("Eiffel",k=1)

#Build the Context

user_query = "Where is the Eiffel Tower located?"
retrieved_docs = retrive_similar_docs(user_query)
retrieved_docs

user_query="what is capital of india?"

def generate_response(query,context_docs):
  context = "\n".join(context_docs)
  prompt=f"Answer the question based on the context below :\n\nContext:\n{context}\n\nQuestion:{query}\nAnswer"
  completion = client.chat.completions.create(
    model="gpt-5",
    messages=[
      {
        "role": "user",
        "content": prompt
      }
    ]
  )
  return completion.choices[0].message.content

response = generate_response(user_query,retrieved_docs)
response

!pip install cohere

from google.colab import userdata
userdata.get('COHERE_KEY')

import cohere
from sentence_transformers import SentenceTransformer

def get_embed_cohere(text):
  response = co.embed(texts=[text],model="small")
  return response.embeddings[0]

cohere_emb = get_embed_cohere(text)
print(f"Cohere Embeddings length :{len(cohere_emb)}")
print(cohere_emb[:5])


#3Hugging Face

model = SentenceTransformer('all-mpnet-base-v2')
sent_emb = model.encode(text)
print(f"Hugging Face Embeddings length :{len(sent_emb)}")
print(sent_emb[:5])
