
"""Pruebas de consistencia del banco de prompts."""

import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------
# 1. Configuración e importaciones
# --------------------------------------------------

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
SRC = RAIZ_PROYECTO / "src"

load_dotenv(RAIZ_PROYECTO / ".env")

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from banco import BANCO_DE_PROMPTS
from consumir_openrouter import (
    armar_peticion,
    enviar_peticion,
    extraer_texto,
)
from parsear_json import parsear_respuesta_json


# --------------------------------------------------
# 2. Validación de la respuesta
# --------------------------------------------------

def validar_estructura(datos, esquema):
    """Comprueba la estructura básica de una respuesta según su esquema."""

    if esquema.get("type") == "object":
        if not isinstance(datos, dict):
            raise ValueError("La respuesta debe ser un objeto JSON.")

        propiedades = esquema.get("properties", {})
        obligatorias = esquema.get("required", [])

        for clave in obligatorias:
            if clave not in datos:
                raise ValueError(f"Falta la clave obligatoria: {clave}")

        if esquema.get("additionalProperties") is False:
            extras = set(datos) - set(propiedades)
            if extras:
                raise ValueError(f"Claves no permitidas: {sorted(extras)}")

        for clave, valor in datos.items():
            if clave in propiedades:
                validar_estructura(valor, propiedades[clave])

    elif esquema.get("type") == "array":
        if not isinstance(datos, list):
            raise ValueError("El valor debe ser una lista.")

        for elemento in datos:
            validar_estructura(elemento, esquema["items"])

    elif esquema.get("type") == "string":
        if not isinstance(datos, str):
            raise ValueError("El valor debe ser un texto.")

        if "enum" in esquema and datos not in esquema["enum"]:
            raise ValueError(
                f"Valor no permitido: {datos!r}. "
                f"Valores aceptados: {esquema['enum']}"
            )


# --------------------------------------------------
# 3. Ejecución de un intento
# --------------------------------------------------

def ejecutar_intento(prompt):
    """Ejecuta una consulta y valida su respuesta."""

    url, headers, body = armar_peticion(
        pregunta=prompt["ejemplo_usuario"],
        system_instruction=prompt["system_instruction"],
        response_schema=prompt.get("response_schema"),
    )

    respuesta = enviar_peticion(url, headers, body)
    texto = extraer_texto(respuesta)

    datos = parsear_respuesta_json(
        texto,
        claves_esperadas=prompt["claves_esperadas"],
    )

    validar_estructura(datos, prompt["response_schema"])

    return datos


# --------------------------------------------------
# 4. Pruebas de consistencia
# --------------------------------------------------

def ejecutar_pruebas(intentos=3):
    """Ejecuta todos los prompts y genera el registro de resultados."""

    registro = {}

    for prompt in BANCO_DE_PROMPTS:
        nombre = prompt["nombre"]
        exitos = 0
        errores = []

        print(f"\nProbando: {nombre}")

        for intento in range(intentos):
            numero_intento = intento + 1

            try:
                ejecutar_intento(prompt)
                exitos += 1

            except Exception as error:
                detalle = str(error) or type(error).__name__

                errores.append({
                    "intento": numero_intento,
                    "error": detalle,
                })

                print(
                    f"  intento {numero_intento} falló: {detalle}"
                )

            if numero_intento < intentos:
                time.sleep(1)

        print(f"{nombre}: {exitos}/{intentos} respuestas válidas")

        registro[nombre] = {
            "exitos": exitos,
            "intentos": intentos,
            "fallos": errores,
        }

    # --------------------------------------------------
    # 5. Guardar resultados
    # --------------------------------------------------

    ruta_registro = (
        RAIZ_PROYECTO / "prompts" / "registro_resultados.json"
    )

    with ruta_registro.open("w", encoding="utf-8") as archivo:
        json.dump(
            registro,
            archivo,
            indent=2,
            ensure_ascii=False,
        )

    total_intentos = sum(
        resultado["intentos"] for resultado in registro.values()
    )

    total_exitos = sum(
        resultado["exitos"] for resultado in registro.values()
    )

    porcentaje = (
        total_exitos / total_intentos * 100
        if total_intentos
        else 0
    )

    print("\n" + "=" * 50)
    print("RESUMEN GENERAL")
    print("=" * 50)
    print(f"Total de respuestas válidas: {total_exitos}/{total_intentos}")
    print(f"Porcentaje de éxito: {porcentaje:.1f}%")
    print(f"Registro guardado en: {ruta_registro.relative_to(RAIZ_PROYECTO)}")


if __name__ == "__main__":
    ejecutar_pruebas()
