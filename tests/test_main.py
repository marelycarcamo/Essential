"""Tests para ``main.ejecutar_prompt`` con dobles (sin red).

Se reemplazan ``armar_peticion``, ``enviar_peticion`` y ``extraer_texto`` por
funciones controladas para verificar el flujo de éxito y de error.
"""

import pytest

import main


PROMPT = {
    "nombre": "prueba",
    "system_instruction": "instrucción",
    "ejemplo_usuario": "pregunta",
    "claves_esperadas": ["comentarios"],
    "response_schema": {"type": "object"},
}


@pytest.fixture
def parchear_red(monkeypatch):
    """Devuelve un helper para configurar las respuestas simuladas de la red."""

    def _configurar(texto):
        monkeypatch.setattr(
            main, "armar_peticion",
            lambda **kw: ("url", {"h": "v"}, {"body": True}),
        )
        monkeypatch.setattr(
            main, "enviar_peticion",
            lambda url, headers, body: {"respuestas": True},
        )
        monkeypatch.setattr(main, "extraer_texto", lambda datos: texto)

    return _configurar


def test_ejecutar_prompt_exitoso(parchear_red, capsys):
    parchear_red('{"comentarios": []}')
    resultado = main.ejecutar_prompt(PROMPT)

    assert resultado["correcto"] is True
    assert resultado["error"] is None
    assert resultado["resultado"] == {"comentarios": []}
    assert resultado["nombre"] == "prueba"
    salida = capsys.readouterr().out
    assert "CORRECTO" in salida


def test_ejecutar_prompt_json_invalido_devuelve_error(parchear_red):
    parchear_red("esto no es json")
    resultado = main.ejecutar_prompt(PROMPT)

    assert resultado["correcto"] is False
    assert resultado["resultado"] is None
    assert "JSON válido" in resultado["error"]


def test_ejecutar_prompt_faltan_claves_devuelve_error(parchear_red):
    parchear_red('{"otra_clave": []}')
    resultado = main.ejecutar_prompt(PROMPT)

    assert resultado["correcto"] is False
    assert "Faltan las claves" in resultado["error"]


def test_ejecutar_prompt_error_de_red(monkeypatch):
    def _falla(**kw):
        raise RuntimeError("sin conexión")

    monkeypatch.setattr(main, "armar_peticion", _falla)
    resultado = main.ejecutar_prompt(PROMPT)

    assert resultado["correcto"] is False
    assert "sin conexión" in resultado["error"]
