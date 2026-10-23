"""Punto de entrada para ejecutar el banco de prompts."""

import sys
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------
# 1. Configuración del proyecto
# --------------------------------------------------

RAIZ_PROYECTO = Path(__file__).resolve().parent
SRC = RAIZ_PROYECTO / "src"

# Cargar las variables de entorno desde la raíz.
load_dotenv(RAIZ_PROYECTO / ".env")

# Permitir importar los módulos de src.
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from banco_prompts import BANCO_DE_PROMPTS
from consumir_openrouter import (
    armar_peticion,
    enviar_peticion,
    extraer_texto,
)
from parsear_json import parsear_respuesta_json


# --------------------------------------------------
# 2. Ejecutar un prompt
# --------------------------------------------------

def ejecutar_prompt(prompt):
    """Ejecuta un prompt, procesa la respuesta y valida sus claves."""

    nombre = prompt["nombre"]

    print(f"\n{'=' * 60}")
    print(f"PROMPT: {nombre}")
    print("=" * 60)

    try:
        # Preparar la solicitud con las instrucciones y el esquema.
        url, headers, body = armar_peticion(
            pregunta=prompt["ejemplo_usuario"],
            system_instruction=prompt["system_instruction"],
            response_schema=prompt["response_schema"],
        )

        # Consultar el modelo mediante OpenRouter.
        datos = enviar_peticion(url, headers, body)

        # Extraer y procesar la respuesta JSON.
        texto = extraer_texto(datos)
        resultado = parsear_respuesta_json(
            texto,
            claves_esperadas=prompt["claves_esperadas"],
        )

        print("Estado: CORRECTO")
        print("\nRespuesta:")
        print(texto)

        return {
            "nombre": nombre,
            "correcto": True,
            "resultado": resultado,
            "error": None,
        }

    except (ValueError, RuntimeError, KeyError, TypeError) as error:
        print(f"Estado: ERROR")
        print(f"Detalle: {error}")

        return {
            "nombre": nombre,
            "correcto": False,
            "resultado": None,
            "error": str(error),
        }


# --------------------------------------------------
# 3. Ejecutar el banco completo
# --------------------------------------------------

def main():
    """Ejecuta todos los prompts y presenta un resumen."""

    print("=" * 60)
    print("BANCO DE PROMPTS: ANÁLISIS DE ENCUESTAS")
    print("=" * 60)

    if not BANCO_DE_PROMPTS:
        print("El banco de prompts está vacío.")
        return

    resultados = []

    for prompt in BANCO_DE_PROMPTS:
        resultado = ejecutar_prompt(prompt)
        resultados.append(resultado)

    # --------------------------------------------------
    # 4. Resumen de ejecución
    # --------------------------------------------------

    exitosos = sum(
        1 for resultado in resultados
        if resultado["correcto"]
    )

    fallidos = len(resultados) - exitosos
    porcentaje = exitosos / len(resultados) * 100

    print("\n" + "=" * 60)
    print("RESUMEN DE EJECUCIÓN")
    print("=" * 60)

    print(f"Total de prompts: {len(resultados)}")
    print(f"Ejecuciones correctas: {exitosos}")
    print(f"Ejecuciones fallidas: {fallidos}")
    print(f"Porcentaje de éxito: {porcentaje:.1f}%")

    if fallidos:
        print("\nRevisa los errores indicados en cada prompt.")
    else:
        print("\nTodos los prompts se ejecutaron correctamente.")


# --------------------------------------------------
# 5. Punto de entrada
# --------------------------------------------------

if __name__ == "__main__":
    main()
