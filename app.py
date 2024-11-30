import streamlit as st
import requests
import pandas as pd
from iteration_utilities import unique_everseen

# Configuración de la página con todos los parámetros posibles
st.set_page_config(
    page_title="Uso de modelos de Hugging Face",
    page_icon="🤪",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.streamlit.io',
        'Report a bug': 'https://www.streamlit.io/contact',
        'About': 'Esta es una aplicación de prueba creada con Streamlit'
    }
)

HUGGINGFACE_API_KEY = st.secrets["HUGGINGFACE_API_KEY"]

# Lista de modelos y sus descripciones
modelos = {

    "FacebookAI/xlm-roberta-large-finetuned-conll03-english": "Este modelo está basado en XLM-RoBERTa, una variante de BERT multilingüe, y se ha afinado para tareas de reconocimiento de entidades en inglés. \n\n\n\n\n**RoBERTa** (Robustly Optimized BERT Approach) es una versión mejorada de BERT desarrollada por Facebook AI. Fue entrenado con más datos y durante más tiempo, eliminando algunas restricciones, lo que resulta en un modelo más robusto y eficiente para comprender patrones complejos en el lenguaje.",

    "dslim/bert-base-NER": "Este modelo utiliza BERT para realizar tareas de reconocimiento de entidades, optimizado para detectar nombres, lugares y organizaciones. \n\n\n\n **BERT** (Bidirectional Encoder Representations from Transformers) es un modelo de lenguaje natural desarrollado por Google en 2018. Su principal innovación es la comprensión bidireccional del contexto, lo que le permite analizar palabras dentro de una oración considerando todo el entorno, mejorando la precisión en tareas como clasificación de texto y respuesta a preguntas."
}

# Función para hacer la consulta al modelo
def query(payload, API_URL, headers):
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Lanza un error si la respuesta es un error HTTP
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error en la solicitud: {e}")
        return {}

# Contenido de la barra lateral
st.sidebar.title("Opciones")
modelo = st.sidebar.selectbox("Selecciona una opción", list(modelos.keys()))

# Mostrar la descripción del modelo seleccionado en la barra lateral
st.sidebar.markdown(f"**Descripción del modelo:**\n\n{modelos[modelo]}")

# Columnas principales
columna1, columna2 = st.columns(2)

with columna1:
    API_URL = f"https://api-inference.huggingface.co/models/{modelo}"
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
        "Content-Type": "application/json"
    }

    if "texto_ejemplo" not in st.session_state:
        st.session_state.texto_ejemplo = ""

    # Botones para los textos de muestra
    colboton1, colboton2, colboton3 = st.columns(3)

    with colboton1:
        if st.button("Texto de muestra 1"):
            st.session_state.texto_ejemplo = "El Hormiguero se presentó en televisión como un programa con invitados de Hollywood, comedia, experimentos científicos y todo tipo de artificios como construcciones mastodónticas o acrobacias espectaculares. Un espacio para toda la familia, liderado por Pablo Motos, que pronto se acostumbró a ser líder de audiencias. Un programa de televisión con entrevistas a estrellas internacionales, dos marionetas, ciencia y magia, entre otros menesteres, lideró con contundencia la franja del prime time en España durante años. Con el paso del tiempo, la ciencia y la magia fueron dejando paso a polémicas, una suerte de reflexiones políticas y tertulias entre personas que están de acuerdo en todo. Eso sí, las marionetas continuaban."

    with colboton2:
        if st.button("Texto de muestra 2"):
            st.session_state.texto_ejemplo = "María Fernández, una ingeniera de Google en Barcelona, viajó a Nueva York para una importante conferencia tecnológica. Durante su estancia, se reunió con Carlos Rodríguez, director de innovación de Apple, para discutir potenciales colaboraciones entre sus equipos. Juntos exploraron ideas sobre inteligencia artificial y su impacto en las ciudades inteligentes. El encuentro tuvo lugar en un elegante restaurante cerca de Wall Street, donde compartieron sus visiones sobre el futuro tecnológico. La conversación abordó proyectos innovadores que podrían revolucionar la forma en que interactuamos con la tecnología en el siglo XXI."

    with colboton3:
        if st.button("Texto de muestra 3"):
            st.session_state.texto_ejemplo = "El Dr. John Smith, un renombrado científico, ha sido galardonado con el Premio Nobel en Medicina por sus contribuciones en la investigación del cáncer. Nacido en Nueva York en 1965, Smith ha trabajado en el Instituto de Tecnología de Massachusetts (MIT) durante más de dos décadas. Su equipo ha desarrollado un nuevo fármaco que ha demostrado ser eficaz en el tratamiento de varios tipos de cáncer. La compañía farmacéutica GlobalHealth, con sede en San Francisco, ha adquirido los derechos de comercialización del fármaco. Este logro ha sido ampliamente reconocido en todo el mundo, y se espera que tenga un impacto significativo en la lucha contra el cáncer."

    # Área de texto
    texto_area = st.text_area("Texto a analizar", value=f"{st.session_state.texto_ejemplo}", height=600)

with columna2:
    if texto_area:
        resultado = query({"inputs": texto_area}, API_URL, headers)

        tabJSON, tabEntidades = st.tabs(["Resultado del modelo", "Entidades"])

        with tabJSON:
            st.write(resultado)

        with tabEntidades:
            if resultado:
                # Crea una lista [] de diccionarios de la forma {"Entidad":xxx, "Tipo":xxxx}
                entidades = [{"Entidad": x["word"], "Tipo": x["entity_group"]} for x in resultado]

                dfEntidades = pd.DataFrame.from_dict(entidades)
                st.table(dfEntidades.drop_duplicates())

'''
Otros modelos en app aparte o mejor en esta misma

https://huggingface.co/facebook/musicgen-small

https://huggingface.co/facebook/detr-resnet-50

https://huggingface.co/Yntec/IsThisDisney?text=a+modern+disney++lion+king

((mirar hg hub code que parece mas moderno))

connersdavis/storyboard-v2

https://huggingface.co/jordiclive/flan-t5-3b-summarizer  (este mola, hace resumenes)
https://huggingface.co/google/pegasus-xsum  otro que resume


añadir spinners

Como se puede usar los modelos desde huggingFace ?
Como se pueden usar los modelos en local, decargandolos ??
Como se pueden usar los modelo desde un colab ?
'''