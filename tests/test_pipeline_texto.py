"""Tests del pipeline de texto (Bloque 15).

Ninguno necesita mocks: las funciones son puras. No tocan la red, no leen
archivos y no dependen del reloj, así que corren en microsegundos y el resultado
es siempre el mismo.

Se prueban también las consecuencias del diseño que el pipeline toma sin decirlo
—números que sobreviven, emojis que desaparecen, palabras que se pegan al quitar
la puntuación— porque son las que sorprenden cuando aparece una respuesta real.
"""

import pytest

from pipeline_texto import (
    limpiar_texto,
    procesar_texto,
    tokenizar,
    vectorizar,
)


# ---------------------------------------------------------------------------
# El caso que justifica el pipeline
# ---------------------------------------------------------------------------

def test_procesar_texto_normaliza_mayusculas_y_puntuacion():
    """El ejemplo del material: mismo contenido, distinto formato."""
    assert procesar_texto("El Cielo, es AZUL!") == ["el", "cielo", "es", "azul"]


def test_limpiar_texto_deja_el_texto_comparable():
    assert limpiar_texto("El cielo es azul.") == "el cielo es azul"


def test_limpiar_texto_colapsa_espacios_repetidos_y_recorta_extremos():
    assert limpiar_texto("   el   cielo   es   azul  ") == "el cielo es azul"


def test_limpiar_texto_convierte_saltos_de_linea_en_un_espacio():
    assert limpiar_texto("el cielo\nes azul\n\n") == "el cielo es azul"


# ---------------------------------------------------------------------------
# Casos borde
# ---------------------------------------------------------------------------

def test_procesar_texto_solo_puntuacion_devuelve_lista_vacia():
    """Una respuesta vacía o cortada llega así: mejor que no lance un error."""
    assert procesar_texto("¡¡¡...!!!") == []


def test_procesar_texto_con_cadena_vacia_devuelve_lista_vacia():
    assert procesar_texto("") == []


def test_procesar_texto_solo_espacios_devuelve_lista_vacia():
    assert procesar_texto("    \n\t  ") == []


@pytest.mark.parametrize(
    "valor", [None, 42, ["texto"], {"a": 1}], ids=["none", "entero", "lista", "dict"]
)
def test_limpiar_texto_con_algo_que_no_es_texto_lanza_typeerror(valor):
    """El mensaje debe decir qué se recibió, no fallar en ``.lower()``."""
    with pytest.raises(TypeError, match="espera str"):
        limpiar_texto(valor)


def test_procesar_texto_propaga_el_typeerror_de_limpiar_texto():
    with pytest.raises(TypeError, match="no NoneType"):
        procesar_texto(None)


# ---------------------------------------------------------------------------
# Decisiones de diseño del pipeline
# ---------------------------------------------------------------------------

def test_limpiar_texto_conserva_tildes_y_enie():
    """En Python 3, ``\\w`` incluye letras acentuadas: no se pierde el español."""
    assert procesar_texto("El niño comió piña y jamón") == [
        "el",
        "niño",
        "comió",
        "piña",
        "y",
        "jamón",
    ]


def test_limpiar_texto_conserva_los_numeros():
    """Un número dentro del texto no puede desaparecer en la limpieza."""
    assert procesar_texto("Tiene 3 gatos y 2 perros") == [
        "tiene",
        "3",
        "gatos",
        "y",
        "2",
        "perros",
    ]


def test_limpiar_texto_elimina_los_emojis():
    assert procesar_texto("Excelente 😀 muy bueno 👍") == ["excelente", "muy", "bueno"]


def test_limpiar_texto_une_palabras_separadas_solo_por_puntuacion():
    """La puntuación se borra, no se reemplaza por espacio.

    Es una consecuencia aceptable para comparar palabras, pero conviene que esté
    escrita: quien lea "holamundo" va a querer saber por qué.
    """
    assert limpiar_texto("hola,mundo") == "holamundo"


def test_limpiar_texto_es_idempotente():
    """Limpiar dos veces no cambia el resultado."""
    una_vez = limpiar_texto("El Cielo, es AZUL!")

    assert limpiar_texto(una_vez) == una_vez


# ---------------------------------------------------------------------------
# Tokenizar y vectorizar
# ---------------------------------------------------------------------------

def test_tokenizar_separa_por_espacios():
    assert tokenizar("el cielo es azul") == ["el", "cielo", "es", "azul"]


def test_tokenizar_de_texto_limpio_vacio_devuelve_lista_vacia():
    assert tokenizar("") == []


def test_vectorizar_cuenta_repeticiones():
    """Un ``set`` no distingue una aparición de dos; el conteo sí."""
    conteo = vectorizar(["el", "gato", "el", "perro", "el"])

    assert conteo["el"] == 3
    assert conteo["gato"] == 1
    assert conteo["perro"] == 1


def test_vectorizar_palabras_repetidas_da_el_mismo_conjunto_y_distinto_conteo():
    """El límite que el Bloque 16 resuelve: presencia no es frecuencia."""
    repetida = procesar_texto("el gato el gato")
    simple = procesar_texto("el gato")

    assert set(repetida) == set(simple)
    assert vectorizar(repetida) != vectorizar(simple)
