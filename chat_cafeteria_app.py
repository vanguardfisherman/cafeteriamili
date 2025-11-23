import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import os
import google.generativeai as genai

# Configuración de la página
st.set_page_config(page_title="Cafetería Mili 🌸", page_icon="🍰", layout="wide")

# Configurar API Key de Gemini (usa secrets en producción)
API_KEY = st.secrets.get("GEMINI_API_KEY", "AIzaSyCahdcqm4qGOpQZN0WxXJ4iH3sot98B9o4")
genai.configure(api_key=API_KEY)

# Estilos CSS Personalizados (Pink / Hello Kitty Theme)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #FFF0F5; /* LavenderBlush */
    }
    .main {
        text-align: left;
        max-width: 1000px;
        margin-left: 2%;
    }
    .chat-bubble-user {
        background-color: #FFB7C5; /* Cherry Blossom Pink */
        color: white;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 5px 0;
        text-align: right;
        font-weight: bold;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .chat-bubble-bot {
        background-color: #FFFFFF;
        color: #D63384; /* Pink text */
        padding: 10px 15px;
        border-radius: 15px;
        margin: 5px 0;
        text-align: left;
        border: 2px solid #FFB7C5;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #D63384 !important;
        font-family: 'Comic Sans MS', cursive, sans-serif;
    }
    .stTextInput > div > div > input {
        border-radius: 20px;
        border: 2px solid #FF69B4;
    }
    .stButton > button {
        background-color: #FF69B4;
        color: white;
        border-radius: 20px;
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Inicializar Inventario en Session State
if "inventario" not in st.session_state:
    st.session_state.inventario = {
        "eclairs": {"nombre": "Eclairs de Fresa", "precio": 3.50, "cantidad": 12, "img": "assets/food/01_eclairs_fresa_crema.png"},
        "brownies": {"nombre": "Brownies de Fresa", "precio": 4.00, "cantidad": 8, "img": "assets/food/02_brownies_de_fresa.png"},
        "macarons": {"nombre": "Macarons Love", "precio": 2.50, "cantidad": 20, "img": "assets/food/03_macarons_corazon_fresa_vainilla.png"},
        "donas_rellenas": {"nombre": "Donas Rellenas", "precio": 2.50, "cantidad": 15, "img": "assets/food/04_donas_rellenas_crema_fresa.png"},
        "donas_glaseadas": {"nombre": "Donas Glaseadas", "precio": 2.00, "cantidad": 10, "img": "assets/food/06_donas_glaseado_fresa.png"},
        "donas_fresa": {"nombre": "Donas Fresa Entera", "precio": 3.00, "cantidad": 5, "img": "assets/food/11_donas_glaseado_fresa_fresa_entera.jpg"},
        "bomboloni": {"nombre": "Bomboloni de Fresa", "precio": 3.50, "cantidad": 10, "img": "assets/food/05_bomboloni_fresa_crema.png"},
        "postre_capas": {"nombre": "Postre en Capas", "precio": 5.00, "cantidad": 6, "img": "assets/food/07_postre_frio_capas_fresa.png"},
        "flan": {"nombre": "Flan Vainilla", "precio": 4.00, "cantidad": 8, "img": "assets/food/08_flan_vainilla_caramelo.png"},
        "mousse": {"nombre": "Mousse Espejo", "precio": 6.50, "cantidad": 4, "img": "assets/food/18_mousse_individual_cereza_glaseado_espejo.png"},
        "croissant": {"nombre": "Croissant Glaseado", "precio": 3.00, "cantidad": 10, "img": "assets/food/09_croissant_glaseado_fresa.png"},
        "hojaldre": {"nombre": "Hojaldre Corazón", "precio": 3.50, "cantidad": 8, "img": "assets/food/10_hojaldre_corazon_crema_fresa.jpg"},
        "choux": {"nombre": "Choux Craquelin", "precio": 4.50, "cantidad": 12, "img": "assets/food/12_choux_craquelin_fresa_flor_crema.png"},
        "rollo_canela": {"nombre": "Rollo Canela Rosa", "precio": 3.50, "cantidad": 15, "img": "assets/food/13_rollo_canela_glaseado_rosa.png"},
        "sandwich": {"nombre": "Sándwich Japonés", "precio": 5.00, "cantidad": 7, "img": "assets/food/14_sandwich_japones_de_fresa.png"},
        "galletas": {"nombre": "Galletas Suaves", "precio": 1.50, "cantidad": 30, "img": "assets/food/15_galletas_suaves_frambuesa.png"},
        "mini_tartas": {"nombre": "Mini Tartas", "precio": 4.00, "cantidad": 12, "img": "assets/food/16_mini_tartas_galleta_rosa_crema.png"},
        "pastel_mousse": {"nombre": "Pastel Mousse Cereza", "precio": 28.00, "cantidad": 3, "img": "assets/food/17_pastel_mousse_cereza_chocolate.jpg"},
        "tarta_frambuesa": {"nombre": "Tarta Frambuesa", "precio": 25.00, "cantidad": 2, "img": "assets/food/19_tarta_frambuesa_flores_comestibles.png"},
        "shortcake": {"nombre": "Shortcake Corazón", "precio": 30.00, "cantidad": 1, "img": "assets/food/20_pastel_corazon_shortcake_fresa.jpg"},
    }

if "ultimo_producto_visto" not in st.session_state:
    st.session_state.ultimo_producto_visto = None

def cargar_txt(ruta):
    preguntas = []
    respuestas = []

    if not os.path.exists(ruta):
        return pd.DataFrame({"Preguntas": [], "Respuestas": []})

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
def preparar_embeddings(df, _modelo):
    if df.empty:
        return None
    embeddings = _modelo.encode(df["Preguntas"].tolist(), convert_to_tensor=True)
    return embeddings

def consultar_gemini(pregunta, contexto_txt, historial=[]):
    try:
        model = genai.GenerativeModel('gemini-flash-latest')
        
        # Formatear historial para contexto
        historial_txt = ""
        for rol, texto in historial[-5:]: # Últimos 5 mensajes para no saturar
            nombre = "Cliente" if rol == "user" else "Mili"
            historial_txt += f"{nombre}: {texto}\n"

        prompt = f"""
        Eres Mili, la asistente virtual de una cafetería llamada 'Cafetería Mili'.
        Tu personalidad es muy tierna, amable, usas emojis como 🌸, 💖, 🍰, y hablas de forma "kawaii".
        Todo en la cafetería es de color rosa y temática Hello Kitty.
        
        Usa la siguiente información de contexto sobre la cafetería para responder.
        También tienes el historial de la conversación reciente para entender el contexto.
        
        Contexto de la Cafetería:
        {contexto_txt}
        
        Historial de Conversación:
        {historial_txt}
        
        Pregunta actual del cliente: {pregunta}
        Respuesta:
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error Gemini: {e}") # Log error to console
        return "¡Ups! Mi cerebro rosado está pensando demasiado. Intenta de nuevo. 🌸"

# Cargar datos y modelo
df = cargar_txt("cafeteria.txt")
contexto_completo = open("cafeteria.txt", "r", encoding="utf-8").read()
modelo = cargar_modelo()
embeddings_preguntas = preparar_embeddings(df, modelo)

# Sidebar
with st.sidebar:
    st.image("assets/mili_logo.png", use_container_width=True)
    st.title("Cafetería Mili 🌸")
    st.markdown("¡Tu lugar feliz y rosado! 💖")
    
    if "historial" not in st.session_state:
        st.session_state.historial = []

    pregunta = st.text_input("¿Qué deseas saber? 💕")
    
    if pregunta:
        if pregunta.lower() in ["salir", "exit", "quit", "chao"]:
            st.info("Chat: ¡Bye bye! Vuelve pronto 🌸")
        else:
            respuesta = ""
            
            # Lógica de Compra
            frases_compra = ["quiero una", "dame una", "comprar", "me llevo una", "quiero uno", "dame uno"]
            es_compra = any(frase in pregunta.lower() for frase in frases_compra)
            es_receta = "receta" in pregunta.lower()
            
            if es_compra and not es_receta:
                prod_key = st.session_state.ultimo_producto_visto
                if prod_key and prod_key in st.session_state.inventario:
                    producto = st.session_state.inventario[prod_key]
                    if producto["cantidad"] > 0:
                        st.session_state.inventario[prod_key]["cantidad"] -= 1
                        respuesta = f"¡Pedido confirmado! 🛵💨\nTu **{producto['nombre']}** está en camino.\nQuedan {st.session_state.inventario[prod_key]['cantidad']} disponibles."
                    else:
                        respuesta = f"¡Oh no! 😿 Se nos acabaron los **{producto['nombre']}**. ¿Quieres probar otra cosa?"
                else:
                    respuesta = "¿Qué te gustaría pedir? Primero pregúntame por algún postre para mostrártelo. 🍰"
            
            # Búsqueda Semántica Local
            elif embeddings_preguntas is not None:
                embedding_pregunta = modelo.encode(pregunta, convert_to_tensor=True)
                similitudes = util.semantic_search(embedding_pregunta, embeddings_preguntas, top_k=1)
                
                if similitudes[0][0]['score'] > 0.5:
                    idx = similitudes[0][0]['corpus_id']
                    respuesta = df["Respuestas"].iloc[int(idx)]
                else:
                    # Fallback a Gemini con Historial
                    respuesta = consultar_gemini(pregunta, contexto_completo, st.session_state.historial)
            else:
                respuesta = consultar_gemini(pregunta, contexto_completo, st.session_state.historial)

            st.session_state.historial.append(("user", pregunta))
            st.session_state.historial.append(("bot", respuesta))

    # Mostrar historial (más recientes arriba)
    for rol, texto in reversed(st.session_state.historial):
        if rol == "user":
            st.markdown(f"<div class='chat-bubble-user'><b>Tú:</b> {texto}</div>", unsafe_allow_html=True)
        else:
            texto_limpio = texto.split("[")[0]
            st.markdown(f"<div class='chat-bubble-bot'><b>Mili:</b> {texto_limpio}</div>", unsafe_allow_html=True)

# Área Principal (Manejo de Triggers y Visualización)
st.title("Bienvenid@ a Cafetería Mili 🍰")
st.image("assets/cupcake.png", width=100)

# Helper para mostrar producto individual y actualizar estado
def mostrar_producto(key):
    st.session_state.ultimo_producto_visto = key
    p = st.session_state.inventario[key]
    st.subheader(f"✨ {p['nombre']}")
    st.image(p['img'], width=400)
    st.markdown(f"**Precio:** ${p['precio']:.2f} | **Disponibles:** {p['cantidad']}")

# Helper para mostrar lista
def mostrar_lista(keys):
    cols = st.columns(len(keys))
    for i, col in enumerate(cols):
        key = keys[i]
        p = st.session_state.inventario[key]
        with col:
            st.image(p['img'], use_container_width=True)
            st.markdown(f"**{p['nombre']}**")
            st.caption(f"💰 ${p['precio']:.2f} | 📦 {p['cantidad']}")

if st.session_state.historial:
    ultima_respuesta = st.session_state.historial[-1][1]
    
    if "[VER_MENU]" in ultima_respuesta or "[VER_CAFES]" in ultima_respuesta:
        st.subheader("☕ Nuestro Menú de Cafés")
        menu_cafes = pd.DataFrame({
            "Café": ["Cappuccino Rosa", "Latte de Fresa", "Mochaccino Love", "Espresso Cute"],
            "Descripción": ["Con espuma rosa y chispas", "Sabor a fresa natural", "Chocolate blanco y frambuesa", "Intenso pero dulce"],
            "Precio": ["$4.50", "$5.00", "$5.50", "$3.00"]
        })
        st.table(menu_cafes)

    if "[VER_POSTRES]" in ultima_respuesta:
        st.subheader("🍰 Nuestros Postres Favoritos")
        mostrar_lista(["eclairs", "brownies", "macarons"])

    if "[VER_ECLAIRS]" in ultima_respuesta:
        mostrar_producto("eclairs")

    if "[VER_BROWNIES]" in ultima_respuesta:
        mostrar_producto("brownies")

    if "[VER_MACARONS]" in ultima_respuesta:
        mostrar_producto("macarons")

    if "[VER_DONAS]" in ultima_respuesta:
        st.subheader("🍩 Nuestras Donas")
        mostrar_lista(["donas_rellenas", "donas_glaseadas", "donas_fresa"])
        # Actualizamos el último visto al primero de la lista por defecto, o dejamos null si queremos ser específicos
        st.session_state.ultimo_producto_visto = "donas_rellenas" 

    if "[VER_FRIOS]" in ultima_respuesta:
        st.subheader("🍨 Postres Fríos")
        mostrar_lista(["postre_capas", "mousse"])
        st.session_state.ultimo_producto_visto = "postre_capas"

    if "[VER_CROISSANT]" in ultima_respuesta:
        st.subheader("🥐 Bollería")
        mostrar_lista(["croissant", "hojaldre"])
        st.session_state.ultimo_producto_visto = "croissant"

    if "[VER_GALLETAS]" in ultima_respuesta:
        st.subheader("🍪 Galletas y Tartaletas")
        mostrar_lista(["galletas", "mini_tartas"])
        st.session_state.ultimo_producto_visto = "galletas"

    if "[VER_PASTELES]" in ultima_respuesta:
        st.subheader("🎂 Pasteles Especiales")
        mostrar_lista(["tarta_frambuesa", "shortcake"])
        st.session_state.ultimo_producto_visto = "tarta_frambuesa"

    if "[VER_PRECIOS]" in ultima_respuesta:
        st.subheader("💲 Lista de Precios")
        st.info("Todos nuestros precios incluyen impuestos y mucho amor. 💖")
        # Generar tabla dinámica desde el inventario
        data = []
        for k, v in st.session_state.inventario.items():
            data.append({"Producto": v["nombre"], "Precio": f"${v['precio']:.2f}", "Stock": v["cantidad"]})
        st.dataframe(pd.DataFrame(data), use_container_width=True)

    if "[VER_UBICACION]" in ultima_respuesta:
        st.subheader("📍 Nuestra Ubicación")
        st.markdown("Calle de las Flores #88, Ciudad Rosa.")
        map_data = pd.DataFrame({'lat': [4.6097], 'lon': [-74.0817]})
        st.map(map_data)

    if "[VER_TODO]" in ultima_respuesta:
        st.subheader("💖 Menú Completo")
        # Mostrar todo el inventario en grid
        keys = list(st.session_state.inventario.keys())
        mostrar_lista(keys)
