import json
import re


def parsear_respuesta_json(texto, claves_esperadas=None):
    """
    Convierte una respuesta de texto en un objeto JSON de Python.

    Parámetros:
        texto (str): respuesta generada por el modelo.
        claves_esperadas (list, opcional): claves obligatorias
            que debe contener el objeto JSON.

    Retorna:
        dict: objeto JSON convertido en un diccionario de Python.

    Raises:
        ValueError: si el texto está vacío, el JSON es inválido,
                    el resultado no es un objeto o faltan claves.
    """

    # 1. Comprobar que la entrada sea texto válido
    if not isinstance(texto, str) or not texto.strip():
        raise ValueError("La respuesta está vacía o no es texto.")

    # 2. Limpiar espacios y posibles bloques Markdown
    texto_limpio = texto.strip()

    texto_limpio = re.sub(
        r"^```(?:json)?\s*",
        "",
        texto_limpio,
        flags=re.IGNORECASE
    )

    texto_limpio = re.sub(r"\s*```$", "", texto_limpio)

    # 3. Convertir el texto a un objeto Python
    try:
        datos = json.loads(texto_limpio)
    except json.JSONDecodeError:
        raise ValueError(
            "La respuesta del modelo no es JSON válido."
        ) from None

    # 4. Comprobar que la respuesta sea un objeto JSON
    if not isinstance(datos, dict):
        raise ValueError(
            "La respuesta JSON debe ser un objeto."
        )

    # 5. Comprobar las claves obligatorias, si se especificaron
    if claves_esperadas is not None:
        faltantes = [
            clave
            for clave in claves_esperadas
            if clave not in datos
        ]

        if faltantes:
            raise ValueError(
                f"Faltan las claves esperadas: {faltantes}"
            )

    return datos