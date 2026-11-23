"""Dataset validado para personalización con PEFT (Bloque 29).

Qué se elige y por qué
----------------------
La actividad pide una tarea donde el problema sea de **comportamiento
consistente** (formato, estilo, clasificación fija) y no de información
faltante. Si el modelo "no sabe X", eso se resuelve con RAG; si sabe pero
responde con formatos distintos cada vez, eso sí se puede corregir con
ejemplos.

La tarea elegida es la del banco de prompts del proyecto: convertir un
comentario libre de una encuesta de satisfacción en un JSON con las dimensiones
mencionadas y la valoración. El modelo ya entiende los comentarios —no le falta
información—, lo que falla es que la salida no siempre respeta las categorías ni
el formato.

Qué contiene el dataset
-----------------------
13 pares comentario → JSON. Cada salida usa exactamente las categorías del
catálogo del proyecto, en minúsculas y sin tildes, porque el modelo aprende a
imitar los ejemplos tal cual están escritos: un error de formato en un ejemplo
se aprende igual que un acierto.

Cómo se valida antes de gastar
------------------------------
Tres comprobaciones, todas locales y sin costo:

1. **Balance**: ninguna categoría puede dominar. Un dataset desbalanceado
   entrena un modelo sesgado hacia la categoría más frecuente.
2. **Formato**: el JSON es válido, tiene las claves esperadas y ninguna extra.
   Una clave extra también se aprende.
3. **Catálogo**: las dimensiones y la valoración pertenecen a la lista cerrada
   del proyecto. Es la comprobación que evita el modo de falla más caro:
   etiquetas inventadas que el modelo reproduce con seguridad.

El envío del trabajo de tuning **no** se ejecuta desde este script: requiere una
cuenta con tuning habilitado y consume cuota. Lo que se entrega es el dataset
preparado y validado, que es lo que la actividad pide.

Uso
---
    ./.venv/Scripts/python.exe dataset_tuning.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter

from google.genai import types

# Catálogo cerrado, tomado de src/banco_prompts.py y del corpus del proyecto.
# Cualquier etiqueta fuera de estas listas es un error que el modelo aprendería.
DIMENSIONES_VALIDAS = (
    "facilitacion",
    "contenidos",
    "materiales",
    "infraestructura",
    "coordinacion",
    "tiempo_practico",
    "alimentacion",
    "valoracion_general",
)

VALORACIONES_VALIDAS = ("positiva", "negativa", "mixta", "neutral")

CLAVES_ESPERADAS = ("dimensiones", "valoracion")

# Umbral de desbalance: si una categoría concentra más de la mitad de los
# ejemplos, el modelo tiende a predecirla por defecto. No es una regla de la
# industria, es un control razonable para un dataset de este tamaño, y queda
# declarado como criterio propio.
PROPORCION_MAXIMA = 0.5

# Cada tupla es (comentario libre, salida esperada).
EJEMPLOS_RAW: list[tuple[str, dict]] = [
    (
        "El relator explicó con mucha paciencia y respondió todas las dudas del grupo.",
        {"dimensiones": ["facilitacion"], "valoracion": "positiva"},
    ),
    (
        "El material entregado estaba desactualizado y varias láminas tenían errores.",
        {"dimensiones": ["materiales"], "valoracion": "negativa"},
    ),
    (
        "Los contenidos fueron útiles, pero faltó práctica con ejercicios reales.",
        {"dimensiones": ["contenidos", "tiempo_practico"], "valoracion": "mixta"},
    ),
    (
        "La sala no tenía enchufes suficientes y el proyector se veía borroso.",
        {"dimensiones": ["infraestructura"], "valoracion": "negativa"},
    ),
    (
        "Todo bien, sin comentarios.",
        {"dimensiones": ["valoracion_general"], "valoracion": "neutral"},
    ),
    (
        "Excelente organización: llegaron los horarios a tiempo y el coffee estaba listo.",
        {"dimensiones": ["coordinacion", "alimentacion"], "valoracion": "positiva"},
    ),
    (
        "El curso estuvo bien en general, aunque el ritmo fue muy rápido para quienes no teníamos base.",
        {"dimensiones": ["contenidos", "facilitacion"], "valoracion": "mixta"},
    ),
    (
        "No se entregó el certificado y nadie respondió mis correos después del curso.",
        {"dimensiones": ["coordinacion"], "valoracion": "negativa"},
    ),
    (
        "La relatora domina el tema y usa ejemplos del rubro.",
        {"dimensiones": ["facilitacion"], "valoracion": "positiva"},
    ),
    (
        "Los ejercicios eran muy largos y no alcanzamos a terminarlos.",
        {"dimensiones": ["tiempo_practico"], "valoracion": "negativa"},
    ),
    (
        "Me pareció un curso correcto: ni bueno ni malo, cumple.",
        {"dimensiones": ["valoracion_general"], "valoracion": "neutral"},
    ),
    (
        "Buen contenido y buenos materiales, pero la sala era incómoda.",
        {"dimensiones": ["contenidos", "materiales", "infraestructura"], "valoracion": "mixta"},
    ),
    (
        "El almuerzo fue variado y se sirvió a tiempo todos los días.",
        {"dimensiones": ["alimentacion"], "valoracion": "positiva"},
    ),
]


def construir_ejemplos(ejemplos_raw: list[tuple[str, dict]]) -> list[types.TuningExample]:
    """Convierte los pares del proyecto en ejemplos del formato de tuning.

    La salida se serializa con ``ensure_ascii=False`` para que quede texto legible
    pero idéntico en estructura: el modelo aprende exactamente la cadena que
    recibe, espacios incluidos.
    """

    return [
        types.TuningExample(
            text_input=comentario, output=json.dumps(salida, ensure_ascii=False)
        )
        for comentario, salida in ejemplos_raw
    ]


def chequear_balance(ejemplos_raw: list[tuple[str, dict]], clave: str) -> Counter:
    """Cuenta cuántos ejemplos hay por categoría de la clave indicada.

    Para ``valoracion`` cuenta un ejemplo por valor. Para ``dimensiones`` cuenta
    una mención por dimensión, así que la suma puede superar el número de
    ejemplos: un comentario con dos dimensiones aporta a las dos.
    """

    if clave == "dimensiones":
        return Counter(
            dimension
            for _, salida in ejemplos_raw
            for dimension in salida.get("dimensiones", [])
        )

    return Counter(salida.get(clave, "sin_clave") for _, salida in ejemplos_raw)


def validar_dataset(
    ejemplos: list[types.TuningExample],
    claves_esperadas: tuple[str, ...] = CLAVES_ESPERADAS,
    dimensiones_validas: tuple[str, ...] = DIMENSIONES_VALIDAS,
    valoraciones_validas: tuple[str, ...] = VALORACIONES_VALIDAS,
) -> list[str]:
    """Revisa el dataset completo y devuelve la lista de problemas encontrados.

    Comprobar que el validador corre sin error no dice nada: la prueba de que
    sirve es que **encuentre** los problemas cuando se los pone a propósito. Por
    eso los casos rotos están entre los tests.

    Returns:
        Lista de problemas. Vacía si el dataset está listo para enviarse.
    """

    problemas: list[str] = []
    vistos: set[str] = set()

    for posicion, ejemplo in enumerate(ejemplos):
        texto = (ejemplo.text_input or "").strip()
        salida_cruda = (ejemplo.output or "").strip()

        if not texto:
            problemas.append(f"ejemplo {posicion}: text_input vacío")
        elif texto in vistos:
            problemas.append(f"ejemplo {posicion}: text_input duplicado")
        else:
            vistos.add(texto)

        if not salida_cruda:
            problemas.append(f"ejemplo {posicion}: output vacío")
            continue

        try:
            salida = json.loads(salida_cruda)
        except json.JSONDecodeError:
            problemas.append(f"ejemplo {posicion}: output no es JSON válido")
            continue

        if not isinstance(salida, dict):
            problemas.append(f"ejemplo {posicion}: output no es un objeto JSON")
            continue

        for clave in claves_esperadas:
            if clave not in salida:
                problemas.append(f'ejemplo {posicion}: falta la clave "{clave}"')

        extras = set(salida) - set(claves_esperadas)
        for extra in sorted(extras):
            problemas.append(f'ejemplo {posicion}: clave inesperada "{extra}"')

        dimensiones = salida.get("dimensiones")
        if "dimensiones" in salida:
            if not isinstance(dimensiones, list) or not dimensiones:
                problemas.append(f"ejemplo {posicion}: 'dimensiones' debe ser una lista no vacía")
            else:
                fuera = [d for d in dimensiones if d not in dimensiones_validas]
                for dimension in fuera:
                    problemas.append(
                        f'ejemplo {posicion}: dimensión fuera del catálogo "{dimension}"'
                    )
                if len(set(dimensiones)) != len(dimensiones):
                    problemas.append(f"ejemplo {posicion}: dimensión repetida")

        if "valoracion" in salida and salida["valoracion"] not in valoraciones_validas:
            problemas.append(
                f'ejemplo {posicion}: valoración fuera del catálogo "{salida["valoracion"]}"'
            )

    return problemas


def revisar_balance(
    ejemplos_raw: list[tuple[str, dict]],
    clave: str,
    proporcion_maxima: float = PROPORCION_MAXIMA,
) -> list[str]:
    """Señala las categorías que concentran demasiados ejemplos.

    Devuelve avisos, no errores: un dataset desbalanceado se puede enviar, solo
    que el modelo va a tender a esa categoría. El criterio queda a la vista para
    que la decisión sea explícita.
    """

    conteo = chequear_balance(ejemplos_raw, clave)
    total = sum(conteo.values())
    if total == 0:
        return [f"'{clave}': no hay ejemplos que contar"]

    return [
        f"'{clave}': {categoria} concentra {cuenta}/{total} "
        f"({cuenta / total:.0%}), sobre el máximo de {proporcion_maxima:.0%}"
        for categoria, cuenta in conteo.most_common()
        if cuenta / total > proporcion_maxima
    ]


def main() -> int:
    ejemplos = construir_ejemplos(EJEMPLOS_RAW)
    print(f"{len(ejemplos)} ejemplos construidos\n")

    print("Balance por valoración:")
    for categoria, cuenta in chequear_balance(EJEMPLOS_RAW, "valoracion").most_common():
        print(f"  {categoria}: {cuenta}")

    print("\nMenciones por dimensión:")
    for dimension, cuenta in chequear_balance(EJEMPLOS_RAW, "dimensiones").most_common():
        print(f"  {dimension}: {cuenta}")

    problemas = validar_dataset(ejemplos)
    avisos = revisar_balance(EJEMPLOS_RAW, "valoracion") + revisar_balance(
        EJEMPLOS_RAW, "dimensiones"
    )

    print(f"\n{len(ejemplos)} ejemplos, {len(problemas)} problemas encontrados")
    for problema in problemas:
        print(f"  - {problema}")

    if avisos:
        print("\nAvisos de balance:")
        for aviso in avisos:
            print(f"  - {aviso}")

    return 1 if problemas else 0


if __name__ == "__main__":
    sys.exit(main())
