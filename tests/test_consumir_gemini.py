"""Tests del cliente de Gemini (respaldo), sin tocar la red.

Importa probarlo igual que el principal: el único momento en que se usa este
módulo es cuando el principal ya falló, así que no puede estar sin verificar.
"""

import pytest

import consumir_gemini as cg


# ---------------------------------------------------------------------------
# armar_peticion: validaciones de entrada y configuración
# ---------------------------------------------------------------------------

def test_pregunta_vacia_lanza_valueerror(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    with pytest.raises(ValueError, match="no puede estar vacía"):
        cg.armar_peticion("   ")


def test_pregunta_none_lanza_valueerror(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    with pytest.raises(ValueError):
        cg.armar_peticion(None)


@pytest.mark.parametrize(
    "entrada",
    [[1, 2], 123, {"a": 1}, 4.5],
    ids=["lista", "entero", "diccionario", "flotante"],
)
def test_pregunta_que_no_es_texto_lanza_valueerror(monkeypatch, entrada):
    """Sin la validación de tipo, un no-str explota con AttributeError en .strip()."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    monkeypatch.setenv("GEMINI_MODEL", "modelo-x")
    with pytest.raises(ValueError, match="debe ser texto"):
        cg.armar_peticion(entrada)


def test_sin_api_key_lanza_runtimeerror(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="API key"):
        cg.armar_peticion("¿Qué es la IA?")


def test_sin_modelo_configurado_lanza_runtimeerror(monkeypatch):
    """El modelo no tiene valor por defecto: no se adivina una versión."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_MODEL"):
        cg.armar_peticion("¿Qué es la IA?")


# ---------------------------------------------------------------------------
# armar_peticion: armado de la solicitud
# ---------------------------------------------------------------------------

def test_estructura_basica_de_la_peticion(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")
    monkeypatch.delenv("GEMINI_BASE_URL", raising=False)

    url, body = cg.armar_peticion("¿Qué es la IA?")

    assert url == (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-3.8-flash:generateContent?key=fake-key"
    )
    assert body["contents"] == [{"parts": [{"text": "¿Qué es la IA?"}]}]
    assert "systemInstruction" not in body


def test_base_url_personalizada_sin_barra_final_duplicada(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    monkeypatch.setenv("GEMINI_MODEL", "modelo-x")
    monkeypatch.setenv("GEMINI_BASE_URL", "https://ejemplo.test/v1beta/")

    url, _body = cg.armar_peticion("pregunta")

    assert url.startswith("https://ejemplo.test/v1beta/models/modelo-x")
    assert "//models" not in url


def test_system_instruction_se_traduce_a_system_instruction(monkeypatch):
    """Gemini no usa rol 'system': la instrucción va en un campo aparte."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    monkeypatch.setenv("GEMINI_MODEL", "modelo-x")

    _url, body = cg.armar_peticion("pregunta", system_instruction="eres un analista")

    assert body["systemInstruction"] == {"parts": [{"text": "eres un analista"}]}
    assert body["contents"] == [{"parts": [{"text": "pregunta"}]}]


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
    monkeypatch.setattr(cg.requests, "post", lambda *a, **k: respuesta)
    return cg.enviar_peticion("url", {})


def test_la_peticion_lleva_timeout(monkeypatch):
    capturado = {}

    def post_falso(url, json=None, timeout=None):
        capturado["timeout"] = timeout
        return RespuestaFalsa(200, {"candidates": []})

    monkeypatch.setattr(cg.requests, "post", post_falso)
    cg.enviar_peticion("url", {})

    assert capturado["timeout"] == cg.TIMEOUT_SEGUNDOS


@pytest.mark.parametrize("codigo", [401, 403], ids=["401", "403"])
def test_credencial_invalida_lanza_error_autenticacion(monkeypatch, codigo):
    with pytest.raises(cg.ErrorAutenticacion):
        _enviar_con_respuesta(monkeypatch, RespuestaFalsa(codigo))


@pytest.mark.parametrize("codigo", [429, 500, 502, 503], ids=["429", "500", "502", "503"])
def test_falla_momentanea_lanza_error_temporal_con_el_codigo(monkeypatch, codigo):
    """El mensaje debe incluir el código: 429 y 5xx no son el mismo diagnóstico."""
    with pytest.raises(cg.ErrorTemporal, match=str(codigo)):
        _enviar_con_respuesta(monkeypatch, RespuestaFalsa(codigo))


def test_400_no_se_clasifica_como_temporal(monkeypatch):
    """Un 400 no se reintenta: si el cuerpo está mal, reintentar no lo arregla."""
    with pytest.raises(RuntimeError) as info:
        _enviar_con_respuesta(monkeypatch, RespuestaFalsa(400))

    assert not isinstance(info.value, cg.ErrorTemporal)


def test_400_con_api_key_invalid_se_clasifica_como_autenticacion(monkeypatch):
    """Google usa 400, no 401, cuando la clave no es válida.

    Se detecta por ``error.details[].reason``, no por el texto del mensaje.
    """
    cuerpo = {
        "error": {
            "code": 400,
            "status": "INVALID_ARGUMENT",
            "details": [{"reason": "API_KEY_INVALID"}],
        }
    }

    with pytest.raises(cg.ErrorAutenticacion):
        _enviar_con_respuesta(monkeypatch, RespuestaFalsa(400, cuerpo))


def test_400_por_otra_causa_no_se_confunde_con_credencial(monkeypatch):
    cuerpo = {"error": {"code": 400, "details": [{"reason": "INVALID_ARGUMENT"}]}}

    with pytest.raises(RuntimeError) as info:
        _enviar_con_respuesta(monkeypatch, RespuestaFalsa(400, cuerpo))

    assert not isinstance(info.value, cg.ErrorAutenticacion)


def test_400_con_cuerpo_no_json_no_se_confunde_con_credencial(monkeypatch):
    class RespuestaSinJson(RespuestaFalsa):
        def json(self):
            raise ValueError("el cuerpo no es JSON")

    with pytest.raises(RuntimeError) as info:
        _enviar_con_respuesta(monkeypatch, RespuestaSinJson(400))

    assert not isinstance(info.value, cg.ErrorAutenticacion)


@pytest.mark.parametrize(
    "excepcion",
    [cg.requests.exceptions.ConnectionError, cg.requests.exceptions.Timeout],
    ids=["connection_error", "timeout"],
)
def test_problema_de_red_lanza_error_temporal(monkeypatch, excepcion):
    def post_que_falla(*args, **kwargs):
        raise excepcion("fallo simulado")

    monkeypatch.setattr(cg.requests, "post", post_que_falla)

    with pytest.raises(cg.ErrorTemporal):
        cg.enviar_peticion("url", {})


def test_respuesta_exitosa_devuelve_el_json(monkeypatch):
    respuesta = RespuestaFalsa(
        200, {"candidates": [{"content": {"parts": [{"text": "hola"}]}}]}
    )
    datos = _enviar_con_respuesta(monkeypatch, respuesta)
    assert datos["candidates"][0]["content"]["parts"][0]["text"] == "hola"


# ---------------------------------------------------------------------------
# extraer_texto
# ---------------------------------------------------------------------------

def test_extraer_texto_valido():
    datos = {"candidates": [{"content": {"parts": [{"text": "hola"}]}}]}
    assert cg.extraer_texto(datos) == "hola"


@pytest.mark.parametrize(
    "datos",
    [
        {},
        {"candidates": []},
        {"candidates": [{"content": {"parts": [{"text": "   "}]}}]},
        {"candidates": [{"content": {"parts": [{"text": 123}]}}]},
    ],
    ids=["sin_candidates", "candidates_vacio", "texto_vacio", "texto_no_str"],
)
def test_extraer_texto_invalido_lanza_runtimeerror(datos):
    with pytest.raises(RuntimeError):
        cg.extraer_texto(datos)
