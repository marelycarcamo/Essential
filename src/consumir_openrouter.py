import os
import requests
from dotenv import load_dotenv

load_dotenv()


def armar_peticion(pregunta):
    """Prepara la solicitud para la API de OpenRouter."""

    api_key = os.getenv("OPENROUTER_API_KEY")
    modelo = "openai/gpt-4o-mini"
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    body = {
        "model": modelo,
        "messages": [
            {
                "role": "user",
                "content": pregunta
            }
        ]
    }

    return url, headers, body


def enviar_peticion(url, headers, body):
    """Envía la solicitud a OpenRouter y devuelve la respuesta JSON."""

    respuesta = requests.post(
        url,
        headers=headers,
        json=body,
        timeout=60
    )

    respuesta.raise_for_status()

    return respuesta.json()


def extraer_texto(datos):
    """Extrae el texto generado por el modelo."""

    return datos["choices"][0]["message"]["content"]


def obtener_dato_de_prueba(pregunta):
    """Coordina el proceso de consulta a OpenRouter."""

    url, headers, body = armar_peticion(pregunta)
    datos = enviar_peticion(url, headers, body)

    return extraer_texto(datos)


if __name__ == "__main__":
    pregunta = "¿Qué es la inteligencia artificial? Explícalo brevemente."

    resultado = obtener_dato_de_prueba(pregunta)

    print(resultado)