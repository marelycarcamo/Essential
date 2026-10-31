"""Mide una respuesta real antes y después del pipeline (Bloque 15).

Qué hace
--------
Genera tres respuestas con el cliente del proyecto, las compara contra una
referencia escrita a mano, y muestra la misma métrica calculada dos veces: sobre
el texto crudo y sobre el texto procesado. La diferencia entre esos dos números
es la evidencia de que el pipeline cambia el resultado.

Además reproduce, sin llamar a la API, los dos límites conocidos de la métrica:
el orden de las palabras y las repeticiones.

Cómo se corre
-------------
    .\\.venv\\Scripts\\python.exe experimentos\\medir_pipeline.py

Usa el proveedor principal (DeepSeek): consume tres llamadas. El registro
completo queda en ``docs/registro_pipeline_texto.txt``.
"""

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from cliente import consultar_modelo  # noqa: E402
from costos import costo_acumulado  # noqa: E402
from errores import ErrorAutenticacion, ErrorReintentosAgotados  # noqa: E402
from pipeline_texto import procesar_texto  # noqa: E402

RUTA_REGISTRO = RAIZ / "docs" / "registro_pipeline_texto.txt"

# Cada caso tiene una consulta y una referencia escrita a mano con lo que una
# respuesta correcta debería contener. La referencia es el patrón contra el que
# se mide: si está mal escrita, la métrica mide otra cosa.
CASOS = [
    {
        "consulta": "¿Qué es Python? Responde en una sola oración.",
        "referencia": "Python es un lenguaje de programación interpretado.",
    },
    {
        "consulta": "¿Qué es una lista en Python? Responde en una sola oración.",
        "referencia": "Una lista es una colección ordenada y modificable de elementos.",
    },
    {
        "consulta": "¿Para qué sirve una función? Responde en una sola oración.",
        "referencia": "Una función agrupa instrucciones y puede recibir parámetros y devolver un resultado.",
    },
]


def coincidencia(tokens_referencia: list[str], tokens_respuesta: list[str]) -> float:
    """Proporción de palabras de la referencia que aparecen en la respuesta.

    Es la métrica del material: conjunto de palabras, sin orden ni frecuencia.
    Se divide por la cantidad de palabras **distintas** de la referencia, así
    que una respuesta que repite todo no sube de 1.0.

    Args:
        tokens_referencia: referencia ya tokenizada.
        tokens_respuesta: respuesta ya tokenizada.

    Returns:
        Proporción entre 0 y 1. Es 0.0 si la referencia no tiene tokens, para no
        dividir por cero.
    """

    if not tokens_referencia:
        return 0.0

    comunes = set(tokens_referencia) & set(tokens_respuesta)

    return len(comunes) / len(set(tokens_referencia))


def _linea(texto: str) -> str:
    """Envuelve un texto largo para que el registro se lea en una consola."""

    return " ".join(texto.split())


def main() -> int:
    lineas = ["Registro del pipeline de texto (Bloque 15)", "=" * 70, ""]
    tabla = []

    for numero, caso in enumerate(CASOS, start=1):
        try:
            respuesta = consultar_modelo(caso["consulta"])
        except ErrorAutenticacion as error:
            print(f"\nProblema de credenciales: {error}")
            return 1
        except ErrorReintentosAgotados as error:
            print(f"\nNo se pudo obtener respuesta: {error}")
            return 1

        crudo_ref = caso["referencia"].split()
        crudo_resp = respuesta.split()

        procesado_ref = procesar_texto(caso["referencia"])
        procesado_resp = procesar_texto(respuesta)

        pct_crudo = coincidencia(crudo_ref, crudo_resp)
        pct_procesado = coincidencia(procesado_ref, procesado_resp)

        tabla.append((numero, pct_crudo, pct_procesado))

        lineas += [
            f"[Caso {numero}]",
            f"Consulta:   {caso['consulta']}",
            f"Referencia: {caso['referencia']}",
            f"Respuesta:  {_linea(respuesta)}",
            f"Coincidencia con texto crudo:     {pct_crudo:.1%}",
            f"Coincidencia con texto procesado: {pct_procesado:.1%}",
            "",
        ]

    # Los dos límites de la métrica, calculados sin llamar a la API.
    limite_orden_ref = procesar_texto("El perro persiguió al gato.")
    limite_orden_resp = procesar_texto("El gato persiguió al perro.")
    limite_repeticion_ref = procesar_texto("el gato")
    limite_repeticion_resp = procesar_texto("el gato el gato el gato")

    lineas += [
        "Límites de la métrica (sin llamadas a la API)",
        "-" * 70,
        "Orden: mismas palabras, significado opuesto.",
        f"  Referencia: El perro persiguió al gato.",
        f"  Respuesta:  El gato persiguió al perro.",
        f"  Coincidencia: {coincidencia(limite_orden_ref, limite_orden_resp):.1%}",
        "",
        "Repeticiones: el conjunto no distingue una aparición de tres.",
        f"  Referencia: el gato",
        f"  Respuesta:  el gato el gato el gato",
        f"  Coincidencia: {coincidencia(limite_repeticion_ref, limite_repeticion_resp):.1%}",
        "",
        f"Costo acumulado de esta corrida: ${costo_acumulado():.8f} USD",
    ]

    texto = "\n".join(lineas)

    print(texto)

    RUTA_REGISTRO.parent.mkdir(parents=True, exist_ok=True)
    RUTA_REGISTRO.write_text(texto + "\n", encoding="utf-8")
    print(f"\nRegistro guardado en {RUTA_REGISTRO.relative_to(RAIZ)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
