"""Mide varias respuestas reales con precisión, cobertura y F1 (Bloque 16).

Qué hace
--------
Genera una respuesta por consulta con el cliente del proyecto, la compara contra
una referencia escrita a mano y calcula las tres métricas de punta a punta.
Guarda el resultado en ``docs/resultados_metricas.json`` —con la respuesta
incluida, no solo los números— para poder volver a leer los casos bajos sin
regenerarlos.

Al final imprime el resumen agregado y los tres peores casos, que es por donde
conviene empezar a leer.

Cómo se corre
-------------
    .\\.venv\\Scripts\\python.exe experimentos\\medir_metricas.py

Consume una llamada al proveedor principal por cada caso.
"""

import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from cliente import consultar_modelo  # noqa: E402
from costos import costo_acumulado  # noqa: E402
from errores import ErrorAutenticacion, ErrorReintentosAgotados  # noqa: E402
from metricas import (  # noqa: E402
    calcular_metricas,
    ordenar_por_f1,
    resumir_metricas,
)

RUTA_RESULTADOS = RAIZ / "docs" / "resultados_metricas.json"

# Cada caso lleva la referencia escrita a mano con lo que una respuesta correcta
# debería contener. La referencia es el patrón: si está mal escrita, la métrica
# mide otra cosa, por muy bien calculada que esté.
CASOS = [
    (
        "¿Qué es Python? Responde en una sola oración.",
        "Python es un lenguaje de programación interpretado de alto nivel y propósito general.",
    ),
    (
        "¿Quién creó Python? Responde en una sola oración.",
        "Python fue creado por Guido van Rossum.",
    ),
    (
        "¿En qué año se publicó la primera versión de Python? Responde solo con el año.",
        "1991",
    ),
    (
        "¿Qué es una lista en Python? Responde en una sola oración.",
        "Una lista es una colección ordenada y modificable de elementos.",
    ),
    (
        "¿Qué es una tupla en Python? Responde en una sola oración.",
        "Una tupla es una colección ordenada e inmutable de elementos.",
    ),
    (
        "¿Qué diferencia hay entre una lista y una tupla? Responde en una sola oración.",
        "Una lista es modificable y una tupla es inmutable.",
    ),
    (
        "¿Qué es un diccionario en Python? Responde en una sola oración.",
        "Un diccionario es una colección de pares de clave y valor.",
    ),
    (
        "¿Qué es una variable? Responde en una sola oración.",
        "Una variable es un nombre que guarda un valor en memoria.",
    ),
    (
        "¿Para qué sirve una función? Responde en una sola oración.",
        "Una función agrupa instrucciones, puede recibir parámetros y puede devolver un resultado.",
    ),
    (
        "¿Qué es un bucle for? Responde en una sola oración.",
        "Un bucle for recorre los elementos de una secuencia.",
    ),
    (
        "¿Qué es una excepción en Python? Responde en una sola oración.",
        "Una excepción es un error que interrumpe la ejecución normal del programa.",
    ),
    (
        "¿Qué hace la función len en Python? Responde en una sola oración.",
        "len devuelve la cantidad de elementos de una secuencia.",
    ),
    (
        "¿Cómo se declara una función en Python? Responde en una sola oración.",
        "Se declara con la palabra def, seguida del nombre, los parámetros entre paréntesis y dos puntos.",
    ),
    (
        "¿Qué es PEP 8? Responde en una sola oración.",
        "PEP 8 es la guía de estilo para escribir código Python.",
    ),
    (
        "¿Qué significa la sigla PEP? Responde en una sola oración.",
        "PEP significa Python Enhancement Proposal.",
    ),
    (
        "¿Qué es un módulo en Python? Responde en una sola oración.",
        "Un módulo es un archivo que contiene código Python reutilizable.",
    ),
    (
        "¿Qué es una clase en Python? Responde en una sola oración.",
        "Una clase es una plantilla para crear objetos con atributos y métodos.",
    ),
    (
        "¿Cómo se abre un archivo en Python? Responde en una sola oración.",
        "Se abre con la función open, indicando la ruta y el modo.",
    ),
    (
        "¿Qué es un comentario en Python? Responde en una sola oración.",
        "Un comentario es texto que el intérprete ignora y sirve para explicar el código.",
    ),
    (
        "¿Qué es la indentación en Python? Responde en una sola oración.",
        "La indentación delimita los bloques de código en Python.",
    ),
]


def guardar(resultados: list[dict]) -> None:
    """Escribe el JSON completo. Se llama después de cada caso.

    Guardar solo al final significaría perder todo lo medido si la conexión o la
    cuota fallan a mitad del ciclo.
    """

    RUTA_RESULTADOS.parent.mkdir(parents=True, exist_ok=True)
    RUTA_RESULTADOS.write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    resultados: list[dict] = []

    for numero, (consulta, referencia) in enumerate(CASOS, start=1):
        try:
            respuesta = consultar_modelo(consulta)
        except ErrorAutenticacion as error:
            print(f"\nProblema de credenciales en el caso {numero}: {error}")
            return 1
        except ErrorReintentosAgotados as error:
            print(f"\nNo se pudo obtener respuesta en el caso {numero}: {error}")
            return 1

        metricas = calcular_metricas(referencia, respuesta)

        resultados.append(
            {"consulta": consulta, "respuesta": respuesta, **metricas}
        )

        guardar(resultados)

        print(
            f"[Caso {numero}] F1 {metricas['f1']:.2f}  "
            f"(precisión {metricas['precision']:.2f} · "
            f"cobertura {metricas['cobertura']:.2f})"
        )

    print()
    print(f"{'#':>2}  {'F1':>5}  {'Prec.':>6}  {'Cob.':>6}  Consulta")
    print("-" * 78)

    for numero, resultado in enumerate(resultados, start=1):
        print(
            f"{numero:>2}  {resultado['f1']:>5.2f}  {resultado['precision']:>6.2f}  "
            f"{resultado['cobertura']:>6.2f}  {resultado['consulta']}"
        )

    resumen = resumir_metricas(resultados)
    print("-" * 78)
    print(
        f"Promedio: F1 {resumen['f1_promedio']:.2f} · "
        f"precisión {resumen['precision_promedio']:.2f} · "
        f"cobertura {resumen['cobertura_promedio']:.2f}"
    )

    print("\nLos tres peores casos, que es por donde hay que empezar a leer:")
    for resultado in ordenar_por_f1(resultados)[:3]:
        print(f"  F1 {resultado['f1']:.2f} — {resultado['consulta']}")
        print(f"     respuesta: {resultado['respuesta'][:160]}")

    print(f"\nCosto acumulado de esta corrida: ${costo_acumulado():.8f} USD")
    print(f"Resultados guardados en {RUTA_RESULTADOS.relative_to(RAIZ)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
