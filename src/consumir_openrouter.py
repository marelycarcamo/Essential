import os
import requests
from dotenv import load_dotenv

load_dotenv()


def armar_peticion(pregunta):
    """Valida la pregunta y prepara la solicitud a OpenRouter."""

    if not pregunta or not pregunta.strip():
        raise ValueError("La pregunta no puede estar vacía.")

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError("No se encontró la API key de OpenRouter.")

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
    """Envía la solicitud y devuelve la respuesta en formato JSON."""

    try:
        respuesta = requests.post(
            url,
            headers=headers,
            json=body,
            timeout=60
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "No se pudo conectar con OpenRouter. "
            "Intenta de nuevo en unos segundos."
        ) from None
    except requests.exceptions.Timeout:
        raise RuntimeError(
            "OpenRouter tardó demasiado en responder."
        ) from None

    if not respuesta.ok:
        raise RuntimeError(
            f"OpenRouter devolvió el error HTTP {respuesta.status_code}."
        )

    try:
        return respuesta.json()
    except requests.exceptions.JSONDecodeError:
        raise RuntimeError(
            "La respuesta del servicio no llegó en el formato esperado."
        ) from None


def extraer_texto(datos):
    """Extrae el texto generado por el modelo."""

    try:
        texto = datos["choices"][0]["message"]["content"]

        if not isinstance(texto, str) or not texto.strip():
            raise RuntimeError(
                "La respuesta del servicio no contiene texto generado."
            )

        return texto

    except (KeyError, IndexError, TypeError):
        raise RuntimeError(
            "La respuesta no trae el texto generado "
            "en el lugar esperado."
        ) from None


def obtener_dato_de_prueba(pregunta):
    """Coordina la consulta a OpenRouter."""

    url, headers, body = armar_peticion(pregunta)
    datos = enviar_peticion(url, headers, body)

    return extraer_texto(datos)


if __name__ == "__main__":
    pregunta = "¿Qué es la inteligencia artificial?"

    try:
        resultado = obtener_dato_de_prueba(pregunta)
        print(resultado)

    except (ValueError, RuntimeError) as error:
        print("No se pudo completar la operación:", error)