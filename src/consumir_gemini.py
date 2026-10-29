"""Cliente de consumo de Gemini (proveedor de respaldo).

Por qué existe
--------------
DeepSeek es el proveedor principal (``docs/decision-arquitectura.md``). Este
módulo se conserva como **respaldo** para cuando el saldo o la cuota del
principal se agoten.

Para que sirva de respaldo de verdad, no puede ser más frágil que el principal:
tiene el mismo ``timeout``, la misma clasificación de errores y la misma
validación de entrada que ``consumir_deepseek.py``. Un respaldo que se cuelga
no es un respaldo.

Nota: este módulo **no** lo usa todavía ``cliente.py``. Para que el cliente
pueda cambiar de proveedor sin tocar código hace falta un proveedor inyectable
(ver "Limitaciones y mejoras futuras" en el README).
"""

import os

import requests
from dotenv import load_dotenv

from errores import ErrorAutenticacion, ErrorTemporal

load_dotenv()

# La URL base es infraestructura estable; el modelo, en cambio, cambia cada
# pocos meses. Por eso la URL tiene valor por defecto y el modelo no: si no
# está en la configuración, el módulo falla en vez de adivinar una versión.
BASE_URL_POR_DEFECTO = "https://generativelanguage.googleapis.com/v1beta"
TIMEOUT_SEGUNDOS = 60


def armar_peticion(pregunta, system_instruction=None):
    """Valida la pregunta y prepara la solicitud a Gemini.

    Args:
        pregunta: texto del usuario. No puede estar vacío.
        system_instruction: contexto permanente para el modelo. Opcional.

    Returns:
        Tupla ``(url, body)``. Gemini no usa cabecera de autenticación: la
        clave viaja en la URL.

    Raises:
        ValueError: si la pregunta viene vacía o es solo espacios.
        RuntimeError: si falta la clave o el modelo en la configuración.
    """

    if not isinstance(pregunta, str):
        raise ValueError(
            f"La pregunta debe ser texto; se recibió {type(pregunta).__name__}."
        )

    if not pregunta.strip():
        raise ValueError("La pregunta no puede estar vacía.")

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("No se encontró la API key de Gemini.")

    modelo = os.getenv("GEMINI_MODEL")

    if not modelo:
        raise RuntimeError(
            "No se encontró el modelo de Gemini. Define GEMINI_MODEL en el .env."
        )

    base_url = os.getenv("GEMINI_BASE_URL", BASE_URL_POR_DEFECTO).rstrip("/")
    url = f"{base_url}/models/{modelo}:generateContent?key={api_key}"

    body = {
        "contents": [{"parts": [{"text": pregunta}]}]
    }

    if system_instruction:
        body["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    return url, body


def _es_error_de_credencial(respuesta):
    """Detecta si un HTTP 400 de Gemini es en realidad un problema de clave.

    Google no sigue la convención de usar 401 para credenciales inválidas:
    devuelve 400 con ``"reason": "API_KEY_INVALID"`` dentro de ``error.details``.

    Se lee el campo estructurado, no el texto del mensaje: el texto puede
    cambiar de idioma o de redacción, el campo ``reason`` es parte del contrato.

    Verificado contra la API real el 2026-10-29.
    """

    try:
        detalles = respuesta.json().get("error", {}).get("details", [])
    except ValueError:
        # El cuerpo no era JSON: no podemos afirmar nada, así que no lo
        # tratamos como problema de credencial.
        return False

    return any(detalle.get("reason") == "API_KEY_INVALID" for detalle in detalles)


def enviar_peticion(url, body):
    """Envía la solicitud y devuelve la respuesta JSON.

    Clasifica el fallo según lo que corresponda hacer con él, igual que
    ``consumir_deepseek.enviar_peticion``:

    - 401/403, o 400 con ``API_KEY_INVALID`` -> ``ErrorAutenticacion``:
      no se reintenta, hay que corregir la clave.
    - 429/5xx, timeout, error de conexión -> ``ErrorTemporal``: se reintenta.
    - Cualquier otro error HTTP o JSON inválido -> ``RuntimeError``.
    """

    try:
        respuesta = requests.post(url, json=body, timeout=TIMEOUT_SEGUNDOS)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        raise ErrorTemporal(
            "No se pudo contactar con Gemini. Puede ser momentáneo."
        ) from None

    if respuesta.status_code in (401, 403) or (
        respuesta.status_code == 400 and _es_error_de_credencial(respuesta)
    ):
        raise ErrorAutenticacion(
            "Credencial inválida o sin permiso. Revisa tu GEMINI_API_KEY."
        )

    if respuesta.status_code == 429 or respuesta.status_code >= 500:
        raise ErrorTemporal(
            "Gemini no pudo procesar la petición. Hay que esperar y reintentar."
        )

    if not respuesta.ok:
        raise RuntimeError(
            f"Gemini devolvió HTTP {respuesta.status_code}: "
            f"{respuesta.text[:500]}"
        )

    try:
        return respuesta.json()
    except requests.exceptions.JSONDecodeError:
        raise RuntimeError(
            "La respuesta de Gemini no llegó en formato JSON."
        ) from None


def extraer_texto(datos):
    """Extrae el texto generado por el modelo."""

    try:
        texto = datos["candidates"][0]["content"]["parts"][0]["text"]

        if not isinstance(texto, str) or not texto.strip():
            raise RuntimeError("La respuesta no contiene texto generado.")

        return texto

    except (KeyError, IndexError, TypeError):
        raise RuntimeError(
            "La respuesta no contiene el texto esperado."
        ) from None


def obtener_dato_de_prueba(pregunta, system_instruction=None):
    """Coordina una consulta completa a Gemini."""

    url, body = armar_peticion(pregunta, system_instruction)
    datos = enviar_peticion(url, body)

    return extraer_texto(datos)
