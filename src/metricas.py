"""Métricas de coincidencia léxica sobre texto generado (Bloque 16).

Por qué existen
---------------
En el Bloque 15 la comparación se hacía con ``set()``, que solo sabe si una
palabra aparece o no. Eso deja dos problemas: ``"el gato el gato"`` y
``"el gato"`` daban 100 %, y no había forma de distinguir una respuesta que se
pasa de larga de una que se queda corta.

Precisión y cobertura separan esas dos preguntas:

- **Precisión**: de todo lo que generó el modelo, cuánto aparece en la
  referencia.
- **Cobertura** (recall): de todo lo que debería haber dicho, cuánto dijo.
- **F1**: la media armónica de las dos, para comparar en volumen sin mirar dos
  columnas a la vez.

Por qué ``Counter`` y no ``set``
--------------------------------
``Counter`` cuenta apariciones, y el operador ``&`` entre dos contadores toma el
**mínimo** de cada palabra. Así, una respuesta que repite "el" cinco veces
donde la referencia lo usa dos no infla el puntaje.

Qué NO resuelve
---------------
La métrica mide parecido léxico, no corrección. Los tres casos están en la
suite como pruebas, porque son el argumento de por qué un número no decide solo:

1. **Sinónimos**: "El auto es veloz" frente a "El vehículo es rápido" da 0.5.
2. **Datos equivocados**: "creado en 1991" frente a "creado en 1989" da 0.8.
3. **Negaciones**: "Python no es compilado" frente a "Python es compilado" da
   0.909, con significados opuestos.

Sirve para revisar mucho texto rápido y para decidir **qué leer primero**. No
sirve para aprobar una respuesta sin mirarla.
"""

from collections import Counter

from pipeline_texto import procesar_texto


def _validar_tokens(tokens, nombre: str) -> None:
    """Rechaza texto crudo donde se espera una lista de tokens.

    Si se pasa un ``str``, ``Counter`` cuenta cada carácter como si fuera una
    palabra y el resultado no tiene sentido, pero tampoco falla: da un número
    equivocado en silencio. Mejor detenerlo con un mensaje que diga qué hacer.
    """

    if isinstance(tokens, str):
        raise TypeError(
            f"{nombre} debe ser una lista de tokens, no un texto. "
            "Pásalo por procesar_texto() antes de comparar."
        )


def contar_coincidencias(
    tokens_referencia: list[str], tokens_respuesta: list[str]
) -> int:
    """Cuenta cuántos tokens coinciden, respetando las repeticiones.

    Args:
        tokens_referencia: tokens de la referencia, ya normalizados.
        tokens_respuesta: tokens de la respuesta, ya normalizados.

    Returns:
        Cantidad de coincidencias. Para cada palabra se toma el mínimo entre lo
        que aparece en la referencia y lo que aparece en la respuesta.

    Raises:
        TypeError: si alguno de los argumentos es un texto en vez de una lista
            de tokens.
    """

    _validar_tokens(tokens_referencia, "tokens_referencia")
    _validar_tokens(tokens_respuesta, "tokens_respuesta")

    conteo_referencia = Counter(tokens_referencia)
    conteo_respuesta = Counter(tokens_respuesta)
    interseccion = conteo_referencia & conteo_respuesta

    return sum(interseccion.values())


def calcular_precision_cobertura_f1(
    tokens_referencia: list[str], tokens_respuesta: list[str]
) -> dict[str, float]:
    """Calcula precisión, cobertura y F1 de dos listas de tokens.

    Args:
        tokens_referencia: tokens de la referencia, ya normalizados.
        tokens_respuesta: tokens de la respuesta, ya normalizados.

    Returns:
        Diccionario con las claves ``precision``, ``cobertura`` y ``f1``, todas
        entre 0 y 1. Si alguno de los dos lados está vacío, las tres valen
        ``0.0`` en lugar de lanzar una división por cero.

    Raises:
        TypeError: si alguno de los argumentos es un texto en vez de una lista
            de tokens.
    """

    coincidencias = contar_coincidencias(tokens_referencia, tokens_respuesta)

    # Una respuesta vacía o cortada llega hasta aquí como lista vacía. Sin este
    # corte, la primera comparación contra ese caso lanzaría ZeroDivisionError.
    if not tokens_respuesta or not tokens_referencia:
        return {"precision": 0.0, "cobertura": 0.0, "f1": 0.0}

    precision = coincidencias / len(tokens_respuesta)
    cobertura = coincidencias / len(tokens_referencia)

    if precision + cobertura == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * cobertura / (precision + cobertura)

    return {"precision": precision, "cobertura": cobertura, "f1": f1}


def calcular_metricas(texto_referencia: str, texto_respuesta: str) -> dict[str, float]:
    """Mide dos textos crudos, sin que quien llama tenga que acordarse del pipeline.

    Es la función que conviene usar al medir: llamar a ``procesar_texto`` y a
    ``calcular_precision_cobertura_f1`` por separado deja abierta la
    posibilidad de olvidar la normalización en uno de los dos lados.

    Args:
        texto_referencia: referencia tal cual, sin limpiar.
        texto_respuesta: respuesta del modelo tal cual, sin limpiar.

    Returns:
        Diccionario con ``precision``, ``cobertura`` y ``f1``.
    """

    return calcular_precision_cobertura_f1(
        procesar_texto(texto_referencia), procesar_texto(texto_respuesta)
    )


def resumir_metricas(resultados: list[dict]) -> dict[str, float]:
    """Promedia las métricas de varios resultados.

    Args:
        resultados: lista de diccionarios con al menos ``precision``,
            ``cobertura`` y ``f1``.

    Returns:
        Diccionario con ``precision_promedio``, ``cobertura_promedio`` y
        ``f1_promedio``. Si la lista está vacía, los tres valen ``0.0``.

    Nota:
        Un promedio esconde si hay tres respuestas mediocres o dos buenas y una
        mala. Sirve como primera lectura, no como veredicto.
    """

    if not resultados:
        return {
            "precision_promedio": 0.0,
            "cobertura_promedio": 0.0,
            "f1_promedio": 0.0,
        }

    total = len(resultados)

    return {
        "precision_promedio": sum(r["precision"] for r in resultados) / total,
        "cobertura_promedio": sum(r["cobertura"] for r in resultados) / total,
        "f1_promedio": sum(r["f1"] for r in resultados) / total,
    }


def ordenar_por_f1(resultados: list[dict]) -> list[dict]:
    """Devuelve los resultados del peor F1 al mejor.

    Revisar en el orden en que se generaron es una forma segura de gastar el
    tiempo en los casos que no lo necesitan.

    Args:
        resultados: lista de diccionarios con la clave ``f1``.

    Returns:
        Una lista nueva, ordenada de menor a mayor F1. No modifica la original.
    """

    return sorted(resultados, key=lambda resultado: resultado["f1"])
