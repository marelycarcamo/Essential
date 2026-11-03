"""Chain de tres pasos con LangChain — Bloque 18.

Caso: ALESSIA (proyecto previo de Marely) — transformar su documentación
de pipeline RAG en material de capacitación para un curso.

Pasos (responsabilidad clara, separada, encadenada):
1. Traducir / extraer puntos clave (de inglés técnico a español claro).
2. Resumir a las 3 ideas principales.
3. Formato como lista de viñetas Markdown, lista para curso.

No es una copia del ejemplo del material: el texto de entrada es real de
ALESSIA (pipeline RAG: ChromaDB + embeddings + reranking) y el propósito
es preparar material de capacitación, no solo formatear.
"""

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# --- Modelo ---
# Temperature=0 para consistencia en pasos de extracción/resumen;
# si quisiéramos creatividad en el formato, podría subirse.
modelo = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)

# --- Entrada real de ALESSIA (texto largo, no trivial) ---
# Esto describe el pipeline RAG que ALESSIA construyó; se toma como caso
# de entrada para preparar material de capacitación.
texto_entrada = (
    "ALESSIA is a RAG-based chatbot built with Python, LangGraph, and ChromaDB. "
    "It ingests PDFs, splits them into chunks with overlap, embeds them using "
    "a multilingual model, stores them in a vector index, retrieves top-k fragments, "
    "reranks them with a CrossEncoder, and verifies fidelity before generating answers. "
    "The pipeline uses structured output (Pydantic) and persists state in OCI."
)

# --- Paso 1: traducir y extraer puntos clave ---
# Responsabilidad: tomar texto técnico en inglés y devolver los puntos
# centrales, ya traducidos al español, sin desarrollarlos.
plantilla_extraccion = PromptTemplate.from_template(
    "Lee el siguiente texto técnico sobre ALESSIA e indica sus 3 puntos "
    "clave en español, uno por línea, sin desarrollarlos:\n\n{texto}"
)

paso_extraccion = plantilla_extraccion | modelo | StrOutputParser()

# --- Paso 2: resumir en lenguaje simple ---
# Responsabilidad: recibir los puntos clave y reformularlos para alguien
# que no conoce RAG, en un tono cercano y sin jerga técnica.
plantilla_resumen = PromptTemplate.from_template(
    "Reescribe estas ideas principales en lenguaje simple, sin jerga técnica, "
    "como si explicaras a un alumno de capacitación:\n\n{puntos}"
)

paso_resumen = plantilla_resumen | modelo | StrOutputParser()

# --- Paso 3: formato como lista de viñetas ---
# Responsabilidad: tomar el resumen simple y convertirlo en lista
# markdown lista para pegar en un documento de curso.
plantilla_formato = PromptTemplate.from_template(
    "Convierte este texto en una lista de viñetas markdown, de máximo 3 líneas, "
    "sin introducción, para usar en un módulo de capacitación:\n\n{resumen}"
)

paso_formato = plantilla_formato | modelo | StrOutputParser()

# --- Función que arma la cadena (para poder inyectar mock en tests) ---
def construir_cadena(modelo):
    """Construye la chain de 3 pasos con funciones intermedias que
    convierten string -> dict para que el siguiente PromptTemplate reciba
    la clave correcta (puntos / resumen)."""
    paso_1 = plantilla_extraccion | modelo | StrOutputParser()
    paso_2 = plantilla_resumen | modelo | StrOutputParser()
    paso_3 = plantilla_formato | modelo | StrOutputParser()

    return (
        paso_1
        | (lambda puntos: {"puntos": puntos})
        | paso_2
        | (lambda resumen: {"resumen": resumen})
        | paso_3
    )


def invocar_con_conmutador(modelo_principal, modelos_alternativos, inputs):
    """Intenta invocar la chain primero con ``modelo_principal``, y si falla
    (lanza cualquier excepción), intenta automáticamente con cada modelo de
    ``modelos_alternativos`` en el orden que se proporcionen.

    Útil cuando un modelo está saturado (503, 429) y se quiere probar con
    otro proveedor antes de rendirse.

    Retorna el resultado de la primera invocación exitosa, o lanza
    ``RuntimeError`` si ningún modelo pudo responder.
    """
    candidatos = [modelo_principal] + list(modelos_alternativos or [])
    for modelo in candidatos:
        try:
            cadena = construir_cadena(modelo)
            return cadena.invoke(inputs)
        except Exception:
            continue
    raise RuntimeError("Ningún modelo pudo responder la cadena")


# --- Ejecución ---
if __name__ == "__main__":
    cadena = construir_cadena(modelo)
    resultado = cadena.invoke({"texto": texto_entrada})
    print("=== Resultado final de la chain (3 pasos) ===")
    print(resultado)
