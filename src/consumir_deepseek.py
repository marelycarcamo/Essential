
import os
import requests
from dotenv import load_dotenv

from errores import ErrorAutenticacion, ErrorTemporal

load_dotenv()


def armar_peticion(pregunta, system_instruction=None):
    """Valida la pregunta y prepara la solicitud a DeepSeek."""

    if not isinstance(pregunta, str):
        raise ValueError(
            f"La pregunta debe ser texto; se recibió {type(pregunta).__name__}."
        )

    if not pregunta.strip():
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
        "messages": _armar_mensajes(pregunta, system_instruction),
        "stream": False
    }

    return url, headers, body


def _armar_mensajes(pregunta, system_instruction=None):
    """Construye la lista de mensajes de la conversación.

    La instrucción de sistema, si existe, va primero: es el contexto permanente
    que el modelo debe respetar durante todo el intercambio.
    """

    messages = []

    if system_instruction:
        messages.append({
            "role": "system",
            "content": system_instruction
        })

    messages.append({
        "role": "user",
        "content": pregunta
    })

    return messages


def enviar_peticion(url, headers, body):
    """Envía la solicitud y devuelve la respuesta JSON.

    Clasifica el fallo según lo que corresponda hacer con él:

    - 401/403 -> ``ErrorAutenticacion``: no se reintenta, hay que corregir la clave.
    - 429/5xx, timeout, error de conexión -> ``ErrorTemporal``: se reintenta.
    - Cualquier otro error HTTP o JSON inválido -> ``RuntimeError``: no es
      reintentable y no es de credenciales.
    """

    try:
        respuesta = requests.post(
            url,
            headers=headers,
            json=body,
            timeout=60
        )
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        raise ErrorTemporal(
            "No se pudo contactar con DeepSeek. Puede ser momentáneo."
        ) from None

    if respuesta.status_code in (401, 403):
        raise ErrorAutenticacion(
            "Credencial inválida o sin permiso. Revisa tu DEEPSEEK_API_KEY."
        )

    if respuesta.status_code == 429 or respuesta.status_code >= 500:
        raise ErrorTemporal(
            "DeepSeek no pudo procesar la petición. Hay que esperar y reintentar."
        )

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
