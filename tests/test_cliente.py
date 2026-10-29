"""Tests del cliente modular.

Nunca tocan la red y nunca esperan de verdad: el borde de red se reemplaza por
dobles y ``time.sleep`` se intercepta para que los tests sean instantáneos y
deterministas. Si no se interceptara, cada test de reintento tardaría 4 segundos.
"""

import pytest

import cliente
from errores import ErrorAutenticacion, ErrorReintentosAgotados, ErrorTemporal


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

@pytest.fixture
def esperas(monkeypatch):
    """Intercepta ``time.sleep`` y registra cuánto se pidió esperar."""
    registro = []
    monkeypatch.setattr(
        cliente.time, "sleep", lambda segundos: registro.append(segundos)
    )
    return registro


def _parchear_armado(monkeypatch):
    """Evita construir una petición real: el cliente solo necesita la tupla."""
    monkeypatch.setattr(
        cliente,
        "armar_peticion",
        lambda pregunta, system_instruction=None: ("url-falsa", {}, {}),
    )


def _respuesta(contenido):
    """Cuerpo típico de una respuesta de chat completions."""
    return {"choices": [{"message": {"content": contenido}}]}


# ---------------------------------------------------------------------------
# Camino feliz
# ---------------------------------------------------------------------------

def test_consultar_modelo_devuelve_el_texto(monkeypatch, esperas):
    _parchear_armado(monkeypatch)
    monkeypatch.setattr(
        cliente, "enviar_peticion", lambda url, headers, body: _respuesta("Santiago")
    )

    assert cliente.consultar_modelo("¿Cuál es la capital de Chile?") == "Santiago"
    assert esperas == []


def test_system_instruction_llega_a_armar_peticion(monkeypatch, esperas):
    capturado = {}

    def armar_falso(pregunta, system_instruction=None):
        capturado["pregunta"] = pregunta
        capturado["system_instruction"] = system_instruction
        return ("url-falsa", {}, {})

    monkeypatch.setattr(cliente, "armar_peticion", armar_falso)
    monkeypatch.setattr(
        cliente, "enviar_peticion", lambda url, headers, body: _respuesta("ok")
    )

    cliente.consultar_modelo("analiza esto", system_instruction="eres un analista")

    assert capturado["pregunta"] == "analiza esto"
    assert capturado["system_instruction"] == "eres un analista"


def test_consultar_completo_entrega_tambien_la_respuesta_cruda(monkeypatch, esperas):
    """La base para el registro de costo del Bloque 13: los tokens están ahí."""
    _parchear_armado(monkeypatch)
    crudo = _respuesta("hola")
    crudo["usage"] = {"prompt_tokens": 10, "completion_tokens": 5}
    monkeypatch.setattr(cliente, "enviar_peticion", lambda *a: crudo)

    texto, datos = cliente._consultar_completo("pregunta")

    assert texto == "hola"
    assert datos["usage"]["prompt_tokens"] == 10


# ---------------------------------------------------------------------------
# Reintento
# ---------------------------------------------------------------------------

def test_reintenta_y_luego_responde(monkeypatch, esperas):
    _parchear_armado(monkeypatch)
    llamadas = []

    def enviar_que_falla_dos_veces(url, headers, body):
        llamadas.append(1)
        if len(llamadas) < 3:
            raise ErrorTemporal("límite de uso")
        return _respuesta("Santiago")

    monkeypatch.setattr(cliente, "enviar_peticion", enviar_que_falla_dos_veces)

    assert cliente.consultar_modelo("¿Capital de Chile?") == "Santiago"
    assert len(llamadas) == 3
    assert esperas == [2, 2]


def test_agota_los_intentos_y_lanza_error_propio(monkeypatch, esperas):
    _parchear_armado(monkeypatch)
    llamadas = []

    def enviar_que_siempre_falla(url, headers, body):
        llamadas.append(1)
        raise ErrorTemporal("límite de uso")

    monkeypatch.setattr(cliente, "enviar_peticion", enviar_que_siempre_falla)

    with pytest.raises(
        ErrorReintentosAgotados, match="no respondió tras varios intentos"
    ):
        cliente.consultar_modelo("pregunta")

    assert len(llamadas) == 3      # tres intentos, no infinitos
    assert esperas == [2, 2]       # dos esperas: entre el 1-2 y el 2-3


def test_no_reintenta_error_de_autenticacion(monkeypatch, esperas):
    """Un 401 no se arregla esperando: debe fallar al primer intento."""
    _parchear_armado(monkeypatch)
    llamadas = []

    def enviar_rechazado(url, headers, body):
        llamadas.append(1)
        raise ErrorAutenticacion("Credencial inválida o sin permiso.")

    monkeypatch.setattr(cliente, "enviar_peticion", enviar_rechazado)

    with pytest.raises(ErrorAutenticacion):
        cliente.consultar_modelo("pregunta")

    assert len(llamadas) == 1
    assert esperas == []


def test_con_reintento_con_un_solo_intento_no_espera(esperas):
    def falla():
        raise ErrorTemporal("caída")

    with pytest.raises(ErrorTemporal):
        cliente.con_reintento(falla, intentos=1)

    assert esperas == []


def test_con_reintento_propaga_el_error_original(esperas):
    """Al agotar los intentos se relanza el mismo error, no uno nuevo."""
    def falla():
        raise ErrorTemporal("caída del proveedor")

    with pytest.raises(ErrorTemporal, match="caída del proveedor"):
        cliente.con_reintento(falla, intentos=2, espera_segundos=1)

    assert esperas == [1]


def test_con_reintento_no_captura_errores_no_temporales(esperas):
    def falla():
        raise ValueError("no es un error temporal")

    with pytest.raises(ValueError):
        cliente.con_reintento(falla, intentos=3)

    assert esperas == []


# ---------------------------------------------------------------------------
# Compatibilidad con los builtins que ya capturaba el proyecto
# ---------------------------------------------------------------------------

def test_error_de_autenticacion_es_capturable_como_valueerror(monkeypatch, esperas):
    _parchear_armado(monkeypatch)

    def enviar_rechazado(url, headers, body):
        raise ErrorAutenticacion("Credencial inválida o sin permiso.")

    monkeypatch.setattr(cliente, "enviar_peticion", enviar_rechazado)

    with pytest.raises(ValueError):
        cliente.consultar_modelo("pregunta")


def test_reintentos_agotados_es_capturable_como_runtimeerror(monkeypatch, esperas):
    _parchear_armado(monkeypatch)

    def enviar_que_siempre_falla(url, headers, body):
        raise ErrorTemporal("límite de uso")

    monkeypatch.setattr(cliente, "enviar_peticion", enviar_que_siempre_falla)

    with pytest.raises(RuntimeError):
        cliente.consultar_modelo("pregunta")


# ---------------------------------------------------------------------------
# Validación de entrada
# ---------------------------------------------------------------------------

def test_pregunta_vacia_lanza_valueerror(esperas):
    """La validación de entrada del módulo de DeepSeek se propaga sin tocar."""
    with pytest.raises(ValueError, match="no puede estar vacía"):
        cliente.consultar_modelo("   ")

    assert esperas == []


def test_pregunta_que_no_es_texto_lanza_valueerror(esperas):
    """Un tipo equivocado debe dar un mensaje claro, no un AttributeError."""
    with pytest.raises(ValueError, match="debe ser texto"):
        cliente.consultar_modelo([1, 2])

    assert esperas == []
