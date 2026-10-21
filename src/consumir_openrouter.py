# consumir_openrouter.py

import os
import requests
from dotenv import load_dotenv

# Cargar las variables de entorno
load_dotenv()

def obtener_dato_de_prueba(pregunta):
    # Obtener la API key
    api_key = os.getenv("OPENROUTER_API_KEY")

    # Configurar el modelo y la URL
    modelo = "openai/gpt-4o-mini"
    url = "https://openrouter.ai/api/v1/chat/completions"

    # Configurar las cabeceras
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Preparar el cuerpo de la solicitud
    body = {
        "model": modelo,
        "messages": [
            {
                "role": "user",
                "content": pregunta
            }
        ]
    }

    # Enviar la solicitud a OpenRouter
    respuesta = requests.post(
        url,
        headers=headers,
        json=body,
        timeout=60
    )

    # Comprobar si hubo un error HTTP
    respuesta.raise_for_status()

    # Convertir la respuesta a un diccionario Python
    datos = respuesta.json()

    # Extraer el texto generado por el modelo
    return datos["choices"][0]["message"]["content"]


if __name__ == "__main__":
    pregunta = "¿Qué es la inteligencia artificial? Explícalo brevemente."
    resultado = obtener_dato_de_prueba(pregunta)
    print(resultado)