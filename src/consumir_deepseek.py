
import os
import requests
from dotenv import load_dotenv

load_dotenv()


def armar_peticion(pregunta):
    """Valida la pregunta y prepara la solicitud a DeepSeek."""

    if not pregunta or not pregunta.strip():
        raise ValueError("La pregunta no puede estar vacía.")

    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise RuntimeError("No se encontró la API key de DeepSeek.")

    base_url = os.getenv(
        "DEEPSEEK_BASE_URL",
        "https://api.deepseek.com"
    ).rstrip("/")

    modelo = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
    url = f"{base_url}/chat/completions"

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
        ],
        "stream": False
    }

    return url, headers, body


def enviar_peticion(url, headers, body):
    """Envía la solicitud y devuelve la respuesta JSON."""

    try:
        respuesta = requests.post(
            url,
            headers=headers,
            json=body,
            timeout=60
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "No se pudo conectar con DeepSeek."
        ) from None
    except requests.exceptions.Timeout:
        raise RuntimeError(
            "DeepSeek tardó demasiado en responder."
        ) from None

    if not respuesta.ok:
        raise RuntimeError(
            f"DeepSeek devolvió HTTP {respuesta.status_code}: "
            f"{respuesta.text[:500]}"
        )

    try:
        return respuesta.json()
    except requests.exceptions.JSONDecodeError:
        raise RuntimeError(
            "La respuesta de DeepSeek no llegó en formato JSON."
        ) from None


def extraer_texto(datos):
    """Extrae el texto generado por DeepSeek."""

    try:
        texto = datos["choices"][0]["message"]["content"]

        if not isinstance(texto, str) or not texto.strip():
            raise RuntimeError(
                "La respuesta no contiene texto generado."
            )

        return texto

    except (KeyError, IndexError, TypeError):
        raise RuntimeError(
            "La respuesta no contiene el texto esperado."
        ) from None


if __name__ == "__main__":
    pregunta = "¿Qué es la inteligencia artificial?"

    try:
        url, headers, body = armar_peticion(pregunta)
        datos = enviar_peticion(url, headers, body)
        resultado = extraer_texto(datos)

        print(f"Proveedor: DeepSeek")
        print(f"Modelo solicitado: {body['model']}")
        print(f"Modelo informado: {datos.get('model', 'No informado')}")
        print("\nRespuesta:")
        print(resultado)

        print("\nUso informado por DeepSeek:")
        print(datos.get("usage", "No informado"))

    except (ValueError, RuntimeError) as error:
        print("No se pudo completar la operación:", error)
