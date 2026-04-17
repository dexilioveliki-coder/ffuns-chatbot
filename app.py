import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os

# Naslov app-a
st.set_page_config(page_title="FFUNS Chatbot", page_icon="📚")
st.title("📚 FFUNS AI Asistent")
st.caption("Pitaj sve o upisu, studijskim programima, rokovima i pravilima na Filozofskom fakultetu u Novom Sadu")

# Učitaj Groq API key iz Secrets (ne piši ga ovde!)
groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    st.error("Groq API key nije podešen. Dodaj ga u Settings > Secrets na Hugging Face Spaces.")
    st.stop()

#
# Podešavanja (jednom)
@st.cache_resource(show_spinner=False)
def load_index():
    # Embeddings (besplatno na HF, dobar za srpski)
    Settings.embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-large")
    
    # LLM preko Groq
    Settings.llm = Groq(model="llama-3.1-70b-versatile", api_key=groq_key)
    
    # Učitaj sve fajlove iz data/ foldera
    documents = SimpleDirectoryReader("data").load_data()
    
    # Kreiraj indeks (RAG)
    index = VectorStoreIndex.from_documents(documents)
    return index.as_query_engine(similarity_top_k=5)

# Učitaj indeks
query_engine = load_index()

# Chat istorija
if "messages" not in st.session_state:
    st.session_state.messages = []

# Prikaz prethodnih poruka
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Npr: Koji su rokovi za upis na osnovne studije?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generiši odgovor
    with st.chat_message("assistant"):
        with st.spinner("Razmišljam... (koristim podatke sa sajta FFUNS)"):
            response = query_engine.query(prompt)
            st.markdown(str(response))
    
    st.session_state.messages.append({"role": "assistant", "content": str(response)})
