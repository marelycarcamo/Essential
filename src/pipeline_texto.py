"""Pipeline de limpieza y tokenización de texto (Bloque 15).

Por qué existe
--------------
Antes de medir qué tan parecida es una respuesta generada a una referencia, el
texto tiene que estar en un formato comparable. ``"El cielo es azul."`` y
``"el cielo es azul"`` dicen exactamente lo mismo, pero para una comparación
ingenua son cadenas distintas: la mayúscula inicial y el punto final cuentan
como diferencias de contenido cuando solo son diferencias de formato.

Centralizar la limpieza y la tokenización en una sola función
(``procesar_texto``) evita el error más común: limpiar un lado de la
comparación y olvidar el otro, o aplicar un paso y saltarse el siguiente.

Sin dependencias externas: ``re`` y ``collections`` son de la biblioteca
estándar, así que no se agrega nada a ``requirements.txt``.

Qué NO resuelve
---------------
El orden ni el significado. ``"El perro persiguió al gato"`` y ``"El gato
persiguió al perro"`` comparten todas sus palabras, y una comparación por
conjuntos las da por idénticas. Ese límite es de la métrica, no del pipeline.

Decisiones de diseño que este módulo ya toma, aunque no se vean
---------------------------------------------------------------
- Los **números sobreviven** a la limpieza, porque ``\\w`` incluye dígitos.
- Los **emojis y símbolos** se descartan igual que la puntuación.
- **La puntuación se elimina, no se reemplaza por espacio**: ``"hola,mundo"``
  queda como ``"holamundo"``. Para este tipo de comparación es aceptable, pero
  conviene tenerlo presente.
- Un texto que era **solo puntuación** produce una lista vacía, no un error.
  Quien divida por la cantidad de tokens debe comprobar antes que no sea cero.
"""

import re
from collections import Counter

# Todo lo que no sea letra, número o espacio se descarta. En Python 3, ``\w``
# reconoce letras acentuadas y la "ñ", así que no se pierden caracteres propios
# del español.
PATRON_PUNTUACION = r"[^\w\s]"

# Cualquier secuencia de espacios, tabuladores o saltos de línea pasa a uno solo.
PATRON_ESPACIOS = r"\s+"


def limpiar_texto(texto: str) -> str:
    """Normaliza el texto: minúsculas, sin puntuación y con espacios simples.

    Args:
        texto: cadena que se quiere normalizar.

    Returns:
        El texto en minúsculas, sin signos de puntuación, con las secuencias de
        espacios colapsadas en una sola y sin espacios en los extremos.

    Raises:
        TypeError: si ``texto`` no es una cadena. La validación está aquí para
            dar un mensaje claro en vez de dejar que falle más abajo con un
            ``AttributeError`` sobre ``.lower()``.
    """

    if not isinstance(texto, str):
        raise TypeError(f"limpiar_texto espera str, no {type(texto).__name__}")

    texto = texto.lower()
    texto = re.sub(PATRON_PUNTUACION, "", texto)
    texto = re.sub(PATRON_ESPACIOS, " ", texto)

    return texto.strip()


def tokenizar(texto_limpio: str) -> list[str]:
    """Convierte texto ya limpio en una lista de tokens.

    Separar por espacios es suficiente sobre texto normalizado, y evita agregar
    NLTK o spaCy para algo que este proyecto no necesita todavía.

    Args:
        texto_limpio: cadena que ya pasó por ``limpiar_texto``.

    Returns:
        Lista de tokens. Puede ser vacía si el texto no tenía contenido.
    """

    return texto_limpio.split()


def procesar_texto(texto: str) -> list[str]:
    """Limpia y tokeniza en un solo paso.

    Es la función que debería usar cualquiera que calcule una métrica: llamar a
    ``limpiar_texto`` y ``tokenizar`` por separado deja abierta la posibilidad de
    olvidar una de las dos.

    Args:
        texto: cadena cruda, tal como llegó del modelo o de una referencia.

    Returns:
        Lista de tokens normalizados.
    """

    return tokenizar(limpiar_texto(texto))


def vectorizar(tokens: list[str]) -> Counter:
    """Cuenta cuántas veces aparece cada token.

    Es la diferencia entre "aparece una vez" y "aparece dos veces", que un
    ``set`` no distingue: ``["el", "gato", "el"]`` y ``["el", "gato"]`` tienen el
    mismo conjunto de palabras y distinto conteo.

    Args:
        tokens: lista de tokens ya normalizados.

    Returns:
        ``Counter`` con el conteo por token.
    """

    return Counter(tokens)
