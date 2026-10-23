"""Tests para ``parsear_json.parsear_respuesta_json`` (función pura, sin red)."""

import pytest

from parsear_json import parsear_respuesta_json


# ---------------------------------------------------------------------------
# Casos válidos
# ---------------------------------------------------------------------------

def test_json_simple_devuelve_dict():
    assert parsear_respuesta_json('{"a": 1}') == {"a": 1}


def test_json_con_espacios_alrededor():
    assert parsear_respuesta_json('   {"a": 1}   ') == {"a": 1}


def test_json_en_bloque_markdown_con_lenguaje():
    texto = '```json\n{"a": 1}\n```'
    assert parsear_respuesta_json(texto) == {"a": 1}


def test_json_en_bloque_markdown_sin_lenguaje():
    texto = "```\n{\"a\": 1}\n```"
    assert parsear_respuesta_json(texto) == {"a": 1}


def test_claves_esperadas_presentes_no_falla():
    datos = parsear_respuesta_json(
        '{"fortalezas": [], "oportunidades": []}',
        claves_esperadas=["fortalezas", "oportunidades"],
    )
    assert datos == {"fortalezas": [], "oportunidades": []}


def test_estructura_anidada():
    texto = '{"comentarios": [{"dimension": "contenidos", "valoracion": "positiva"}]}'
    datos = parsear_respuesta_json(texto, claves_esperadas=["comentarios"])
    assert datos["comentarios"][0]["dimension"] == "contenidos"


# ---------------------------------------------------------------------------
# Entradas inválidas
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "entrada",
    [None, "", "   ", "\n\t"],
    ids=["None", "vacio", "espacios", "saltos"],
)
def test_entrada_vacia_o_no_texto_lanza_valueerror(entrada):
    with pytest.raises(ValueError, match="vacía|no es texto"):
        parsear_respuesta_json(entrada)


def test_entrada_no_str_lanza_valueerror():
    with pytest.raises(ValueError):
        parsear_respuesta_json({"a": 1})  # dict en lugar de str


def test_json_invalido_lanza_valueerror():
    with pytest.raises(ValueError, match="JSON válido"):
        parsear_respuesta_json('{"a": 1,}')


def test_json_que_no_es_objeto_lanza_valueerror():
    with pytest.raises(ValueError, match="debe ser un objeto"):
        parsear_respuesta_json("[1, 2, 3]")


def test_json_escalar_lanza_valueerror():
    with pytest.raises(ValueError, match="debe ser un objeto"):
        parsear_respuesta_json("42")


def test_claves_faltantes_lanza_valueerror():
    with pytest.raises(ValueError, match="Faltan las claves"):
        parsear_respuesta_json(
            '{"fortalezas": []}',
            claves_esperadas=["fortalezas", "oportunidades"],
        )


def test_claves_faltantes_menciona_cuales():
    with pytest.raises(ValueError) as excinfo:
        parsear_respuesta_json('{"a": 1}', claves_esperadas=["b", "c"])
    mensaje = str(excinfo.value)
    assert "b" in mensaje and "c" in mensaje


def test_claves_esperadas_none_no_valida():
    # Si no se pasan claves, un objeto cualquiera es válido.
    assert parsear_respuesta_json('{"x": 1}', claves_esperadas=None) == {"x": 1}
