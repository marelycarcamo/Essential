"""Tests del dataset de personalización (Bloque 29).

Qué se prueba y qué no
----------------------
Se prueba que el dataset esté listo para enviarse y, sobre todo, que el
validador **detecte** los errores cuando se los pone a propósito: un validador
que corre sin lanzar excepción sobre un dataset roto no sirve de nada. Cada
comprobación tiene su caso malo.

Lo que no se prueba aquí es el efecto del entrenamiento sobre el modelo: eso
requiere enviar un trabajo de tuning, que no se hace desde este repositorio.
"""

import json

import pytest
from google.genai import types

from dataset_tuning import (
    CLAVES_ESPERADAS,
    DIMENSIONES_VALIDAS,
    EJEMPLOS_RAW,
    VALORACIONES_VALIDAS,
    chequear_balance,
    construir_ejemplos,
    revisar_balance,
    validar_dataset,
)


def ejemplo(texto: str = "El relator explicó bien.", salida: dict | None = None) -> types.TuningExample:
    salida = salida or {"dimensiones": ["facilitacion"], "valoracion": "positiva"}
    return types.TuningExample(text_input=texto, output=json.dumps(salida))


# --- Dataset real ----------------------------------------------------------


def test_el_dataset_tiene_al_menos_diez_ejemplos():
    assert len(EJEMPLOS_RAW) >= 10


def test_el_dataset_real_no_tiene_problemas():
    assert validar_dataset(construir_ejemplos(EJEMPLOS_RAW)) == []


def test_el_dataset_real_no_tiene_textos_repetidos():
    textos = [comentario for comentario, _ in EJEMPLOS_RAW]

    assert len(textos) == len(set(textos))


def test_todas_las_salidas_usan_el_catalogo():
    for _, salida in EJEMPLOS_RAW:
        assert salida["valoracion"] in VALORACIONES_VALIDAS
        assert all(dimension in DIMENSIONES_VALIDAS for dimension in salida["dimensiones"])


def test_el_dataset_real_esta_balanceado():
    assert revisar_balance(EJEMPLOS_RAW, "valoracion") == []


def test_construir_ejemplos_serializa_sin_escapar_acentos():
    ejemplos = construir_ejemplos(EJEMPLOS_RAW)

    assert "facilitacion" in ejemplos[0].output
    assert "\\u" not in ejemplos[0].output


# --- El validador detecta lo que debe detectar ----------------------------


def test_un_dataset_correcto_no_tiene_problemas():
    assert validar_dataset([ejemplo()]) == []


def test_detecta_texto_vacio():
    problemas = validar_dataset([ejemplo(texto="   ")])

    assert any("text_input vacío" in problema for problema in problemas)


def test_detecta_texto_duplicado():
    problemas = validar_dataset([ejemplo(), ejemplo()])

    assert any("duplicado" in problema for problema in problemas)


def test_detecta_output_vacio():
    roto = types.TuningExample(text_input="algo", output="  ")

    problemas = validar_dataset([roto])

    assert any("output vacío" in problema for problema in problemas)


def test_detecta_output_que_no_es_json():
    roto = types.TuningExample(text_input="algo", output="{no es json")

    problemas = validar_dataset([roto])

    assert any("no es JSON válido" in problema for problema in problemas)


def test_detecta_json_que_no_es_objeto():
    roto = types.TuningExample(text_input="algo", output="[1, 2, 3]")

    problemas = validar_dataset([roto])

    assert any("no es un objeto JSON" in problema for problema in problemas)


def test_detecta_clave_faltante():
    roto = ejemplo(salida={"dimensiones": ["contenidos"]})

    problemas = validar_dataset([roto])

    assert any('falta la clave "valoracion"' in problema for problema in problemas)


def test_detecta_clave_inesperada():
    roto = ejemplo(salida={"dimensiones": ["contenidos"], "valoracion": "mixta", "nota": 5})

    problemas = validar_dataset([roto])

    assert any('clave inesperada "nota"' in problema for problema in problemas)


def test_detecta_dimension_fuera_del_catalogo():
    roto = ejemplo(salida={"dimensiones": ["puntualidad"], "valoracion": "positiva"})

    problemas = validar_dataset([roto])

    assert any("fuera del catálogo" in problema and "puntualidad" in problema for problema in problemas)


def test_detecta_dimensiones_vacias():
    roto = ejemplo(salida={"dimensiones": [], "valoracion": "positiva"})

    problemas = validar_dataset([roto])

    assert any("lista no vacía" in problema for problema in problemas)


def test_detecta_dimensiones_que_no_son_lista():
    roto = ejemplo(salida={"dimensiones": "facilitacion", "valoracion": "positiva"})

    problemas = validar_dataset([roto])

    assert any("lista no vacía" in problema for problema in problemas)


def test_detecta_dimension_repetida():
    roto = ejemplo(salida={"dimensiones": ["contenidos", "contenidos"], "valoracion": "mixta"})

    problemas = validar_dataset([roto])

    assert any("repetida" in problema for problema in problemas)


def test_detecta_valoracion_fuera_del_catalogo():
    roto = ejemplo(salida={"dimensiones": ["contenidos"], "valoracion": "regular"})

    problemas = validar_dataset([roto])

    assert any("valoración fuera del catálogo" in problema for problema in problemas)


def test_las_claves_esperadas_son_las_del_contrato():
    assert CLAVES_ESPERADAS == ("dimensiones", "valoracion")


# --- Balance --------------------------------------------------------------


def test_balance_por_valoracion_cuenta_un_ejemplo_por_valor():
    conteo = chequear_balance(EJEMPLOS_RAW, "valoracion")

    assert sum(conteo.values()) == len(EJEMPLOS_RAW)
    assert conteo["positiva"] == 4


def test_balance_por_dimension_cuenta_menciones():
    conteo = chequear_balance(EJEMPLOS_RAW, "dimensiones")

    # Un comentario con tres dimensiones aporta a las tres.
    assert sum(conteo.values()) > len(EJEMPLOS_RAW)


def test_avisa_cuando_una_categoria_domina():
    desbalanceado = [
        ("uno", {"dimensiones": ["contenidos"], "valoracion": "negativa"}),
        ("dos", {"dimensiones": ["contenidos"], "valoracion": "negativa"}),
        ("tres", {"dimensiones": ["contenidos"], "valoracion": "positiva"}),
    ]

    avisos = revisar_balance(desbalanceado, "valoracion")

    assert any("negativa" in aviso for aviso in avisos)


def test_no_avisa_cuando_el_balance_es_razonable():
    balanceado = [
        ("uno", {"dimensiones": ["contenidos"], "valoracion": "negativa"}),
        ("dos", {"dimensiones": ["contenidos"], "valoracion": "positiva"}),
    ]

    assert revisar_balance(balanceado, "valoracion") == []


def test_avisa_si_no_hay_nada_que_contar():
    avisos = revisar_balance([], "valoracion")

    assert any("no hay ejemplos" in aviso for aviso in avisos)
