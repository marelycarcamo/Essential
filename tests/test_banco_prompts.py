"""Tests de estructura de ``banco_prompts`` (datos, sin red)."""

import pytest

from banco_prompts import BANCO_DE_PROMPTS


def test_banco_no_esta_vacio():
    assert BANCO_DE_PROMPTS
    assert isinstance(BANCO_DE_PROMPTS, list)


def test_cada_prompt_tiene_las_claves_requeridas():
    requeridas = {
        "nombre",
        "system_instruction",
        "ejemplo_usuario",
        "claves_esperadas",
        "response_schema",
    }
    for prompt in BANCO_DE_PROMPTS:
        assert requeridas.issubset(prompt.keys()), prompt.get("nombre")


def test_nombres_unicos():
    nombres = [p["nombre"] for p in BANCO_DE_PROMPTS]
    assert len(nombres) == len(set(nombres))


def test_claves_esperadas_estan_en_el_schema():
    """Las claves obligatorias de la respuesta deben figurar en el schema."""
    for prompt in BANCO_DE_PROMPTS:
        properties = prompt["response_schema"].get("properties", {})
        for clave in prompt["claves_esperadas"]:
            assert clave in properties, f"{prompt['nombre']}: falta {clave}"


def test_schema_es_json_schema_valido_minimo():
    for prompt in BANCO_DE_PROMPTS:
        schema = prompt["response_schema"]
        assert schema["type"] == "object"
        assert "properties" in schema


@pytest.mark.parametrize("prompt", BANCO_DE_PROMPTS, ids=lambda p: p["nombre"])
def test_textos_no_estan_vacios(prompt):
    assert prompt["system_instruction"].strip()
    assert prompt["ejemplo_usuario"].strip()
