import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
def cargar_txt(ruta):
    preguntas = []
    respuestas = []

    with open(ruta, "r", encoding="utf-8") as f:
        contenido = f.read().split("---")

        for bloque in contenido:
            if "PREGUNTA:" in bloque and "RESPUESTA:" in bloque:
                partes = bloque.strip().split("\n")
                preg = partes[0].replace("PREGUNTA:", "").strip()
                resp = partes[1].replace("RESPUESTA:", "").strip()
                preguntas.append(preg)
                respuestas.append(resp)

    return pd.DataFrame({"Preguntas": preguntas, "Respuestas": respuestas})

@st.cache_resource
def cargar_modelo():
    return SentenceTransformer("distiluse-base-multilingual-cased-v2")

@st.cache_data
def preparar_embeddings(df, _modelo):  # ← el guion bajo evita error de hashing
    embeddings = _modelo.encode(df["Preguntas"].tolist(), convert_to_tensor=True)
    return embeddings

st.set_page_config(page_title="Chat Ferretería", layout="wide")

#se coloco este codigo CSS para el estilo del chat y alineación a la izquierda
st.markdown(
    """
    <style>
    .main {
        text-align: left;
        max-width: 1000px;
        margin-left: 2%;
    }
    .chat-bubble-user {
        background-color: #DCF8C6;
        padding: 8px 12px;
        border-radius: 12px;
        margin: 4px 0;
        text-align: right;
    }
    .chat-bubble-bot {
        background-color: #E6E6E6;
        padding: 8px 12px;
        border-radius: 12px;
        margin: 4px 0;
        text-align: left;
    }
    </style>
    """,
    unsafe_allow_html=True
)
df = cargar_txt("ferreteria.txt")
modelo = cargar_modelo()
embeddings_preguntas = preparar_embeddings(df, modelo)

st.sidebar.title("Chat de Ferretería")
if "historial" not in st.session_state:
    st.session_state.historial = []

pregunta = st.sidebar.text_input("Tu pregunta:")

if pregunta:
    if pregunta.lower() in ["salir", "exit", "quit", "chao"]:
        st.sidebar.info("Chat: ¡Hasta luego!")
    else:
        embedding_pregunta = modelo.encode(pregunta, convert_to_tensor=True)
        similitudes = util.semantic_search(embedding_pregunta, embeddings_preguntas, top_k=1)
        idx = similitudes[0][0]['corpus_id']
        respuesta = df["Respuestas"].iloc[int(idx)]

        #en esta parte se Guarda la convresacion
        st.session_state.historial.append(("user", pregunta))
        st.session_state.historial.append(("bot", respuesta))

# este fragmento de codigo muesrtra historial del chat
for rol, texto in st.session_state.historial:
    if rol == "user":
        st.sidebar.markdown(f"<div class='chat-bubble-user'><b>Tú:</b> {texto}</div>", unsafe_allow_html=True)
    else:
        st.sidebar.markdown(f"<div class='chat-bubble-bot'><b>Chat:</b> {texto}</div>", unsafe_allow_html=True)

#Aqui se coloco el titulo, que determinara el contenido a mostrar
st.title("Aquí lo que desea mostrar")
