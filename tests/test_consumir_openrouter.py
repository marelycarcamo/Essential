"""Tests para ``consumir_openrouter`` sin tocar la red.

Se parchea ``os.getenv`` vía ``monkeypatch`` para simular/omitir la API key y
no se llama nunca a ``enviar_peticion`` (o se le pasa un mock si hiciera falta).
"""

import pytest

import consumir_openrouter as co


# ---------------------------------------------------------------------------
# armar_peticion: validaciones
# ---------------------------------------------------------------------------

def test_pregunta_vacia_lanza_valueerror(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
    with pytest.raises(ValueError, match="no puede estar vacía"):
        co.armar_peticion("   ")


def test_pregunta_none_lanza_valueerror(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
    with pytest.raises(ValueError):
        co.armar_peticion(None)


def test_sin_api_key_lanza_runtimeerror(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="API key"):
        co.armar_peticion("¿Qué es la IA?")


# ---------------------------------------------------------------------------
# armar_peticion: armado de la solicitud
# ---------------------------------------------------------------------------

def test_estructura_basica_de_la_peticion(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)

    url, headers, body = co.armar_peticion("¿Qué es la IA?")

    assert url == "https://openrouter.ai/api/v1/chat/completions"
    assert headers["Authorization"] == "Bearer fake-key"
    assert headers["Content-Type"] == "application/json"
    assert body["model"] == "openai/gpt-4o-mini"
    assert body["messages"] == [{"role": "user", "content": "¿Qué es la IA?"}]


def test_system_instruction_se_agrega_primero(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
    _url, _headers, body = co.armar_peticion(
        "pregunta", system_instruction="eres un asistente"
    )
    assert body["messages"][0] == {
        "role": "system",
        "content": "eres un asistente",
    }
    assert body["messages"][1]["role"] == "user"


def test_sin_system_instruction_solo_hay_user(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
    _url, _headers, body = co.armar_peticion("pregunta")
    assert len(body["messages"]) == 1
    assert body["messages"][0]["role"] == "user"


def test_response_schema_genera_response_format(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
    schema = {"type": "object", "properties": {}}
    _url, _headers, body = co.armar_peticion("pregunta", response_schema=schema)

    assert body["response_format"]["type"] == "json_schema"
    json_schema = body["response_format"]["json_schema"]
    assert json_schema["strict"] is True
    assert json_schema["schema"] == schema


def test_sin_schema_no_hay_response_format(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
    _url, _headers, body = co.armar_peticion("pregunta")
    assert "response_format" not in body


def test_modelo_y_base_url_personalizados(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
    monkeypatch.setenv("OPENROUTER_MODEL", "otro/modelo")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://custom.example/v1")
    url, _headers, body = co.armar_peticion("pregunta")
    assert url == "https://custom.example/v1"
    assert body["model"] == "otro/modelo"


# ---------------------------------------------------------------------------
# extraer_texto (pura)
# ---------------------------------------------------------------------------

def test_extraer_texto_valido():
    datos = {"choices": [{"message": {"content": "hola"}}]}
    assert co.extraer_texto(datos) == "hola"


@pytest.mark.parametrize(
    "datos",
    [
        {},
        {"choices": []},
        {"choices": [{"message": {"content": "   "}}]},
        {"choices": [{"message": {"content": 123}}]},
    ],
    ids=["sin_choices", "choices_vacio", "contenido_vacio", "contenido_no_str"],
)
def test_extraer_texto_invalido_lanza_runtimeerror(datos):
    with pytest.raises(RuntimeError):
        co.extraer_texto(datos)


# ---------------------------------------------------------------------------
# calcular_costo (pura)
# ---------------------------------------------------------------------------

def test_calcular_costo():
    datos = {"usage": {"prompt_tokens": 100, "completion_tokens": 50}}
    costo = co.calcular_costo(datos, precio_entrada=0.001, precio_salida=0.002)
    # 100*0.001 + 50*0.002 = 0.1 + 0.1 = 0.2
    assert costo == pytest.approx(0.2)
