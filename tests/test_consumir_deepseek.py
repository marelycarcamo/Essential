"""Tests para ``consumir_deepseek`` sin tocar la red."""

import pytest

import consumir_deepseek as cd


def test_pregunta_vacia_lanza_valueerror(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")
    with pytest.raises(ValueError, match="no puede estar vacía"):
        cd.armar_peticion("   ")


@pytest.mark.parametrize(
    "entrada",
    [[1, 2], 123, {"a": 1}, 4.5],
    ids=["lista", "entero", "diccionario", "flotante"],
)
def test_pregunta_que_no_es_texto_lanza_valueerror(monkeypatch, entrada):
    """Sin la validación de tipo, un no-str explota con AttributeError en .strip()."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")
    with pytest.raises(ValueError, match="debe ser texto"):
        cd.armar_peticion(entrada)


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


# ---------------------------------------------------------------------------
# armar_peticion: instrucción de sistema
# ---------------------------------------------------------------------------

def test_system_instruction_se_agrega_primero(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")
    _url, _headers, body = cd.armar_peticion(
        "pregunta", system_instruction="eres un analista"
    )
    assert body["messages"][0] == {
        "role": "system",
        "content": "eres un analista",
    }
    assert body["messages"][1]["role"] == "user"


def test_sin_system_instruction_solo_hay_user(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake")
    _url, _headers, body = cd.armar_peticion("pregunta")
    assert len(body["messages"]) == 1


# ---------------------------------------------------------------------------
# enviar_peticion: clasificación de errores
# ---------------------------------------------------------------------------

class RespuestaFalsa:
    """Imita lo mínimo de ``requests.Response`` que usa el código."""

    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self.ok = 200 <= status_code < 400
        self.text = "cuerpo de error simulado"
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload


def _enviar_con_respuesta(monkeypatch, respuesta):
    monkeypatch.setattr(cd.requests, "post", lambda *a, **k: respuesta)
    return cd.enviar_peticion("url", {}, {})


@pytest.mark.parametrize("codigo", [401, 403], ids=["401", "403"])
def test_credencial_invalida_lanza_error_autenticacion(monkeypatch, codigo):
    with pytest.raises(cd.ErrorAutenticacion):
        _enviar_con_respuesta(monkeypatch, RespuestaFalsa(codigo))


@pytest.mark.parametrize("codigo", [429, 500, 502, 503], ids=["429", "500", "502", "503"])
def test_falla_momentanea_lanza_error_temporal(monkeypatch, codigo):
    with pytest.raises(cd.ErrorTemporal):
        _enviar_con_respuesta(monkeypatch, RespuestaFalsa(codigo))


def test_error_http_no_reintentable_sigue_siendo_runtimeerror(monkeypatch):
    """Un 400 no es de credenciales ni momentáneo: no debe clasificarse como temporal."""
    with pytest.raises(RuntimeError) as info:
        _enviar_con_respuesta(monkeypatch, RespuestaFalsa(400))

    assert not isinstance(info.value, cd.ErrorTemporal)


@pytest.mark.parametrize(
    "excepcion",
    [cd.requests.exceptions.ConnectionError, cd.requests.exceptions.Timeout],
    ids=["connection_error", "timeout"],
)
def test_problema_de_red_lanza_error_temporal(monkeypatch, excepcion):
    def post_que_falla(*args, **kwargs):
        raise excepcion("fallo simulado")

    monkeypatch.setattr(cd.requests, "post", post_que_falla)

    with pytest.raises(cd.ErrorTemporal):
        cd.enviar_peticion("url", {}, {})


def test_respuesta_exitosa_devuelve_el_json(monkeypatch):
    respuesta = RespuestaFalsa(200, {"choices": [{"message": {"content": "hola"}}]})
    assert _enviar_con_respuesta(monkeypatch, respuesta)["choices"][0]["message"]["content"] == "hola"

