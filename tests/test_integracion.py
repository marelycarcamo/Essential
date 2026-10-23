"""Tests de integración del pipeline completo.

A diferencia de los tests unitarios, aquí se ejercita la cadena real:

    armar_peticion -> enviar_peticion -> extraer_texto -> parsear_respuesta_json

Solo se reemplaza el borde de red (``requests.post``) por un doble, de forma que
toda la lógica del proyecto se ejecuta de verdad, sin conectar a Internet.

También hay un test *real* opcional que se salta si no hay ``OPENROUTER_API_KEY``.
"""

import os

import pytest

import consumir_openrouter as co
import parsear_json
from banco_prompts import BANCO_DE_PROMPTS


# ---------------------------------------------------------------------------
# Doble de respuesta HTTP que imita la interfaz de ``requests.Response``
# ---------------------------------------------------------------------------

class RespuestaFalsa:
    """Imita lo mínimo de ``requests.Response`` que usa el código."""

    def __init__(self, payload):
        self._payload = payload
        self.ok = True
        self.status_code = 200

    def json(self):
        return self._payload

    @property
    def text(self):
        import json
        return json.dumps(self._payload)


def _respuesta_chat(content):
    """Construye el cuerpo típico de una respuesta de chat completions."""
    return {
        "model": "openai/gpt-4o-mini",
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }


# ---------------------------------------------------------------------------
# Integración: pipeline feliz de punta a punta (red simulada)
# ---------------------------------------------------------------------------

def test_pipeline_completo_feliz(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)

    contenido = '{"fortalezas": ["claridad"], "oportunidades": ["más tiempo"]}'
    capturado = {}

    def post_falso(url, headers=None, json=None, timeout=None):
        capturado["url"] = url
        capturado["headers"] = headers
        capturado["body"] = json
        capturado["timeout"] = timeout
        return RespuestaFalsa(_respuesta_chat(contenido))

    monkeypatch.setattr(co.requests, "post", post_falso)

    # 1) armar la solicitud con instrucciones y schema.
    url, headers, body = co.armar_peticion(
        "Analiza estos comentarios",
        system_instruction="eres un analista",
        response_schema={"type": "object", "properties": {}},
    )

    # 2) enviar (borde de red doblado).
    datos = co.enviar_peticion(url, headers, body)

    # 3) extraer el texto y 4) parsearlo como el flujo real.
    texto = co.extraer_texto(datos)
    resultado = parsear_json.parsear_respuesta_json(
        texto, claves_esperadas=["fortalezas", "oportunidades"]
    )

    assert resultado == {"fortalezas": ["claridad"], "oportunidades": ["más tiempo"]}
    # La petición salió bien formada y hacia el endpoint correcto.
    assert capturado["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert capturado["headers"]["Authorization"] == "Bearer fake-key"
    assert capturado["timeout"] == 60
    assert capturado["body"]["messages"][0]["role"] == "system"


def test_pipeline_con_contenido_en_bloque_markdown(monkeypatch):
    """Muchos modelos envuelven el JSON en ```json ... ```; debe seguir funcionando."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    contenido = '```json\n{"comentarios": [{"dimension": "contenidos", "valoracion": "positiva"}]}\n```'
    monkeypatch.setattr(
        co.requests, "post",
        lambda *a, **k: RespuestaFalsa(_respuesta_chat(contenido)),
    )

    url, headers, body = co.armar_peticion("pregunta")
    texto = co.extraer_texto(co.enviar_peticion(url, headers, body))
    resultado = parsear_json.parsear_respuesta_json(
        texto, claves_esperadas=["comentarios"]
    )

    assert resultado["comentarios"][0]["dimension"] == "contenidos"


def test_pipeline_json_invalido_se_detecta(monkeypatch):
    """Un modelo que responde texto plano debe terminar en ValueError."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    monkeypatch.setattr(
        co.requests, "post",
        lambda *a, **k: RespuestaFalsa(_respuesta_chat("no soy json")),
    )

    url, headers, body = co.armar_peticion("pregunta")
    texto = co.extraer_texto(co.enviar_peticion(url, headers, body))

    with pytest.raises(ValueError, match="JSON válido"):
        parsear_json.parsear_respuesta_json(texto)


def test_pipeline_error_http_se_propaga(monkeypatch):
    """Un status != 200 debe convertirse en RuntimeError en ``enviar_peticion``."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")

    class RespuestaError(RespuestaFalsa):
        def __init__(self):
            super().__init__({})
            self.ok = False
            self.status_code = 429

    monkeypatch.setattr(co.requests, "post", lambda *a, **k: RespuestaError())

    url, headers, body = co.armar_peticion("pregunta")
    with pytest.raises(RuntimeError, match="HTTP 429"):
        co.enviar_peticion(url, headers, body)


# ---------------------------------------------------------------------------
# Integración real (opt-in): se salta si no hay credenciales
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="Requiere OPENROUTER_API_KEY (test de integración real).",
)
def test_integracion_real_openrouter():
    """Llama de verdad a OpenRouter con el primer prompt del banco.

    Solo corre si hay ``OPENROUTER_API_KEY`` en el entorno/.env. Sirve como
    *smoke test* del contrato extremo a extremo contra el proveedor real.
    """
    prompt = BANCO_DE_PROMPTS[0]

    url, headers, body = co.armar_peticion(
        prompt["ejemplo_usuario"],
        system_instruction=prompt["system_instruction"],
        response_schema=prompt["response_schema"],
    )
    datos = co.enviar_peticion(url, headers, body)
    texto = co.extraer_texto(datos)
    resultado = parsear_json.parsear_respuesta_json(
        texto, claves_esperadas=prompt["claves_esperadas"]
    )

    assert isinstance(resultado, dict)
    for clave in prompt["claves_esperadas"]:
        assert clave in resultado
