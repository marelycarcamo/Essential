"""Tests de las métricas de coincidencia léxica (Bloque 16).

Dos grupos, con propósitos distintos:

- **El cálculo**: que precisión, cobertura y F1 den el número que corresponde,
  incluyendo los casos donde antes había división por cero.
- **Los límites**: sinónimos, datos equivocados y negaciones. Estos tests no
  comprueban que la métrica esté mal implementada; documentan que está bien
  implementada y aun así da un número engañoso. Si alguien "arregla" la métrica
  para que el caso del sinónimo dé 1.0, estas pruebas fallan y avisan por qué.
"""

import pytest

from metricas import (
    calcular_metricas,
    calcular_precision_cobertura_f1,
    contar_coincidencias,
    ordenar_por_f1,
    resumir_metricas,
)


# ---------------------------------------------------------------------------
# Conteo de coincidencias
# ---------------------------------------------------------------------------

def test_una_repeticion_de_mas_no_infla_el_puntaje():
    """El problema que ``set()`` no resolvía: "el gato el gato el gato" no es "el gato"."""
    referencia = ["el", "gato"]
    respuesta = ["el", "gato", "el", "gato", "el", "gato"]

    assert contar_coincidencias(referencia, respuesta) == 2

    metricas = calcular_precision_cobertura_f1(referencia, respuesta)

    assert metricas["precision"] == pytest.approx(2 / 6)
    assert metricas["cobertura"] == pytest.approx(1.0)


def test_cuenta_todas_las_repeticiones_que_la_referencia_permite():
    referencia = ["el", "gato", "el", "gato"]
    respuesta = ["el", "gato"]

    assert contar_coincidencias(referencia, respuesta) == 2


# ---------------------------------------------------------------------------
# Precisión, cobertura y F1
# ---------------------------------------------------------------------------

def test_ejemplo_calculado_a_mano():
    """El ejemplo del material: 5 coincidencias, 9 y 6 tokens."""
    referencia = ["el", "gato", "duerme", "en", "el", "sofá"]
    respuesta = ["el", "gato", "duerme", "en", "el", "sillón", "todo", "el", "día"]

    metricas = calcular_precision_cobertura_f1(referencia, respuesta)

    assert metricas["precision"] == pytest.approx(5 / 9)
    assert metricas["cobertura"] == pytest.approx(5 / 6)
    assert metricas["f1"] == pytest.approx(2 * (5 / 9) * (5 / 6) / ((5 / 9) + (5 / 6)))


def test_una_respuesta_mas_larga_baja_la_precision_y_sube_la_cobertura():
    """El patrón que anticipa el material sobre respuestas detalladas."""
    referencia = ["el", "gato", "duerme"]

    corta = calcular_precision_cobertura_f1(referencia, ["el", "gato"])
    larga = calcular_precision_cobertura_f1(
        referencia, ["el", "gato", "duerme", "en", "el", "sofá"]
    )

    assert corta["precision"] == pytest.approx(1.0)
    assert corta["cobertura"] == pytest.approx(2 / 3)
    assert larga["precision"] == pytest.approx(3 / 6)
    assert larga["cobertura"] == pytest.approx(1.0)


def test_respuesta_vacia_devuelve_ceros():
    """Una respuesta vacía o cortada no puede lanzar ZeroDivisionError."""
    assert calcular_precision_cobertura_f1(["el", "gato"], []) == {
        "precision": 0.0,
        "cobertura": 0.0,
        "f1": 0.0,
    }


def test_referencia_vacia_devuelve_ceros():
    assert calcular_precision_cobertura_f1([], ["el", "gato"]) == {
        "precision": 0.0,
        "cobertura": 0.0,
        "f1": 0.0,
    }


def test_sin_palabras_en_comun_devuelve_ceros_y_no_falla():
    """Precisión y cobertura en cero hacen que la fórmula de F1 divida por cero."""
    assert calcular_precision_cobertura_f1(["perro"], ["gato"]) == {
        "precision": 0.0,
        "cobertura": 0.0,
        "f1": 0.0,
    }


@pytest.mark.parametrize(
    "referencia, respuesta",
    [("texto crudo", ["gato"]), (["gato"], "texto crudo")],
    ids=["referencia_texto", "respuesta_texto"],
)
def test_pasar_texto_en_vez_de_tokens_lanza_typeerror(referencia, respuesta):
    """Con un texto, ``Counter`` contaría caracteres y el número saldría mal sin avisar."""
    with pytest.raises(TypeError, match="lista de tokens"):
        calcular_precision_cobertura_f1(referencia, respuesta)


# ---------------------------------------------------------------------------
# La función de punta a punta
# ---------------------------------------------------------------------------

def test_calcular_metricas_normaliza_los_dos_lados():
    """Sin pipeline, "El Cielo, es AZUL!" frente a "el cielo es azul" no daría 1.0."""
    metricas = calcular_metricas("El Cielo, es AZUL!", "el cielo es azul")

    assert metricas["precision"] == pytest.approx(1.0)
    assert metricas["cobertura"] == pytest.approx(1.0)
    assert metricas["f1"] == pytest.approx(1.0)


def test_calcular_metricas_con_respuesta_vacia_devuelve_ceros():
    assert calcular_metricas("Python es un lenguaje", "") == {
        "precision": 0.0,
        "cobertura": 0.0,
        "f1": 0.0,
    }


# ---------------------------------------------------------------------------
# Resumen y orden
# ---------------------------------------------------------------------------

def test_resumir_metricas_de_lista_vacia_devuelve_ceros():
    assert resumir_metricas([]) == {
        "precision_promedio": 0.0,
        "cobertura_promedio": 0.0,
        "f1_promedio": 0.0,
    }


def test_resumir_metricas_promedia_los_tres_valores():
    resultados = [
        {"precision": 1.0, "cobertura": 0.5, "f1": 0.5},
        {"precision": 0.5, "cobertura": 1.0, "f1": 0.5},
    ]

    resumen = resumir_metricas(resultados)

    assert resumen["precision_promedio"] == pytest.approx(0.75)
    assert resumen["cobertura_promedio"] == pytest.approx(0.75)
    assert resumen["f1_promedio"] == pytest.approx(0.5)


def test_ordenar_por_f1_pone_el_peor_primero():
    resultados = [
        {"consulta": "buena", "f1": 0.9},
        {"consulta": "mala", "f1": 0.2},
        {"consulta": "regular", "f1": 0.6},
    ]

    assert [r["consulta"] for r in ordenar_por_f1(resultados)] == [
        "mala",
        "regular",
        "buena",
    ]


def test_ordenar_por_f1_no_modifica_la_lista_original():
    resultados = [{"f1": 0.9}, {"f1": 0.2}]

    ordenar_por_f1(resultados)

    assert resultados[0]["f1"] == 0.9


# ---------------------------------------------------------------------------
# Los límites: la métrica está bien y aun así engaña
# ---------------------------------------------------------------------------

def test_sinonimo_perfecto_puntua_como_si_fuera_incorrecto():
    """Dice exactamente lo mismo con otras palabras, y saca 0.5."""
    metricas = calcular_metricas("El auto es veloz.", "El vehículo es rápido.")

    assert metricas["f1"] == pytest.approx(0.5)


def test_dato_equivocado_puntua_alto():
    """Una fecha cambiada es un solo token: la métrica apenas lo nota."""
    metricas = calcular_metricas(
        "Python fue creado en 1991.", "Python fue creado en 1989."
    )

    assert metricas["precision"] == pytest.approx(0.8)
    assert metricas["cobertura"] == pytest.approx(0.8)
    assert metricas["f1"] == pytest.approx(0.8)


def test_una_negacion_ausente_puntua_casi_perfecto():
    """El caso más peligroso: significados opuestos con F1 de 0.909."""
    metricas = calcular_metricas(
        "Python no es un lenguaje compilado.", "Python es un lenguaje compilado."
    )

    assert metricas["precision"] == pytest.approx(1.0)
    assert metricas["cobertura"] == pytest.approx(5 / 6)
    assert metricas["f1"] == pytest.approx(0.909, abs=0.001)
