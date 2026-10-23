"""Tests para ``consumir_deepseek`` sin tocar la red."""

import pytest

import consumir_deepseek as cd


def test_pregunta_vacia_lanza_valueerror(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")
    with pytest.raises(ValueError, match="no puede estar vacía"):
        cd.armar_peticion("   ")


def test_sin_api_key_lanza_runtimeerror(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="API key"):
        cd.armar_peticion("¿Qué es la IA?")


def test_estructura_basica_de_la_peticion(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)

    url, headers, body = cd.armar_peticion("¿Qué es la IA?")

    assert url == "https://api.deepseek.com/chat/completions"
    assert headers["Authorization"] == "Bearer fake-key"
    assert body["model"] == "deepseek-flash"
    assert body["stream"] is False
    assert body["messages"] == [{"role": "user", "content": "¿Qué es la IA?"}]


def test_base_url_con_barra_final_se_limpia(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/")
    url, _headers, _body = cd.armar_peticion("pregunta")
    assert url == "https://api.deepseek.com/chat/completions"


def test_modelo_personalizado(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-reasoner")
    _url, _headers, body = cd.armar_peticion("pregunta")
    assert body["model"] == "deepseek-reasoner"


def test_extraer_texto_valido():
    datos = {"choices": [{"message": {"content": "hola"}}]}
    assert cd.extraer_texto(datos) == "hola"


def test_extraer_texto_invalido_lanza_runtimeerror():
    with pytest.raises(RuntimeError):
        cd.extraer_texto({"choices": []})
