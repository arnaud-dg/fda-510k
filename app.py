import requests
import numpy as np
import json
import os
import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
import openai
import logging
import nltk

# def download_file(url, local_filename):
#     with requests.get(url, stream=True) as r:
#         r.raise_for_status()
#         with open(local_filename, 'wb') as f:            for chunk in r.iter_content(chunk_size=8192):
#                 f.write(chunk)

nltk.download('punkt')

# URLs des fichiers dans votre repo GitHub
base_url = "https://raw.githubusercontent.com/arnaud-dg/fda-510k/main/data/"
files = ["embeddings.npy", "metadatas.json", "ids.npy"]

# for file in files:
#     download_file(base_url + file, file)

# Charger les fichiers téléchargés
document_embeddings = np.load("embeddings.npy")
with open("metadatas.json", "r") as f:
    metadatas = json.load(f)
ids = np.load("ids.npy").tolist()

# Configurer l'API OpenAI
openai.api_key = st.secrets["OPENAI_API_KEY"]


# Initialiser le client Chroma
client = chromadb.Client()
collection_name = "documents"

# Charger les embeddings, les métadonnées et les identifiants depuis les fichiers locaux
document_embeddings = np.load("embeddings.npy")
with open("metadatas.json", "r") as f:
    metadatas = json.load(f)
ids = np.load("ids.npy").tolist()

# Vérifier si la collection existe, sinon la recréer et la charger
if collection_name not in [coll.name for coll in client.list_collections()]:
    collection = client.create_collection(collection_name)
    collection.add(
        embeddings=document_embeddings.tolist(), 
        metadatas=metadatas,
        ids=ids
    )
else:
    collection = client.get_collection(collection_name)

logging.basicConfig(level=logging.INFO)
logging.info("Collection prête à être utilisée")

# Charger le modèle de transformation de phrases
model = SentenceTransformer('all-MiniLM-L6-v2')

def retrieve_documents(query_embedding, top_k=5):
    logging.info("Recherche de documents pertinents")
    results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=top_k)
    
    logging.info(f"Résultats de la requête: {results}")

    # Supposons que la structure des résultats est {'results': [{'metadata': {'content': '...'}}]}
    retrieved_docs = [result['metadata']['content'] for result in results['results']]
    logging.info(f"Documents récupérés: {retrieved_docs}")
    return retrieved_docs

def generate_response(query, context):
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    logging.info("Génération de la réponse")
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=200
    )
    logging.info(f"Réponse générée: {response.choices[0].text.strip()}")
    return response.choices[0].text.strip()

# Streamlit app
st.title("Chatbot LLM-RAG")

# Utiliser st.session_state pour stocker l'historique du chat
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

user_message = st.text_input("Posez votre question:")

if user_message and st.button("Envoyer"):
    st.session_state.chat_history.append({"role": "user", "content": user_message})
    
    # Encoder la question de l'utilisateur
    query_embedding = model.encode([user_message])[0]
    
    # Récupérer les documents pertinents
    retrieved_docs = retrieve_documents(query_embedding)
    
    # Générer une réponse
    context = " ".join(retrieved_docs)
    response = generate_response(user_message, context)
    
    st.session_state.chat_history.append({"role": "assistant", "content": response})

# Afficher l'historique des chats mis à jour
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.write(f"**Vous:** {chat['content']}")
    else:
        st.write(f"**Assistant:** {chat['content']}")
