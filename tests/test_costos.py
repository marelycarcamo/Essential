"""Tests del registro de costo por operación.

Todo se calcula con ``momento`` explícito: el costo de DeepSeek depende de la
hora (punta o valle), así que un test que usara "ahora" pasaría o fallaría según
la hora a la que alguien lo corra.

Los valores esperados salen de la tabla oficial de precios de ``deepseek-flash``
(USD por 1M de tokens):

- entrada con caché acertada: 0,003 valle / 0,006 punta
- entrada sin caché:          0,15  valle / 0,30  punta
- salida:                     0,60  valle / 1,20  punta
"""

import logging
from datetime import datetime, timezone

import pytest

import costos

MODELO = "deepseek-flash"

# Miércoles 28 de octubre de 2026. A las 02:00 UTC cae en la ventana de punta
# 01:00-04:00; a las 12:00 UTC, fuera de toda ventana.
PUNTA = datetime(2026, 10, 28, 2, 0, tzinfo=timezone.utc)
VALLE = datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc)
SABADO = datetime(2026, 10, 31, 8, 0, tzinfo=timezone.utc)


def _uso(entrada=0, salida=0, cache_hit=None, cache_miss=None):
    """Bloque ``usage`` como lo devuelve DeepSeek."""
    datos = {
        "usage": {
            "prompt_tokens": entrada + (cache_hit or 0),
            "completion_tokens": salida,
        }
    }
    if cache_hit is not None:
        datos["usage"]["prompt_cache_hit_tokens"] = cache_hit
    if cache_miss is not None:
        datos["usage"]["prompt_cache_miss_tokens"] = cache_miss
    return datos


# ---------------------------------------------------------------------------
# Cálculo
# ---------------------------------------------------------------------------

def test_suma_entrada_y_salida_con_sus_tarifas():
    """1 000 de entrada sin caché + 500 de salida, en valle."""
    datos = _uso(entrada=1_000, salida=500, cache_hit=0, cache_miss=1_000)

    esperado = (1_000 * 0.15 + 500 * 0.60) / 1_000_000

    assert costos.calcular_costo(datos, MODELO, VALLE) == pytest.approx(esperado)


def test_cache_hit_cuesta_menos_que_cache_miss():
    """Los mismos tokens de entrada, con y sin caché, no valen lo mismo."""
    con_cache = _uso(entrada=1_000_000, cache_hit=1_000_000, cache_miss=0)
    sin_cache = _uso(entrada=1_000_000, cache_hit=0, cache_miss=1_000_000)

    ahorro = costos.calcular_costo(con_cache, MODELO, VALLE)
    costo = costos.calcular_costo(sin_cache, MODELO, VALLE)

    assert ahorro == pytest.approx(0.003)
    assert costo == pytest.approx(0.15)
    assert ahorro < costo


def test_cache_miss_se_deriva_cuando_no_viene_el_desglose():
    """Si el proveedor no informa cache miss, se cobra lo no acertado como miss."""
    datos = {"usage": {"prompt_tokens": 1_000, "completion_tokens": 0}}

    assert costos.calcular_costo(datos, MODELO, VALLE) == pytest.approx(0.15 / 1_000)


def test_respuesta_sin_usage_cuesta_cero():
    """Sin bloque ``usage`` no hay tokens que cobrar: cero, no una excepción."""
    assert costos.calcular_costo({}, MODELO, VALLE) == 0.0


def test_hora_punta_cobra_el_doble_que_la_valle():
    """Las dos ventanas de punta duplican exactamente las tres tarifas."""
    datos = _uso(entrada=1_000_000, salida=1_000_000, cache_hit=0, cache_miss=1_000_000)

    en_valle = costos.calcular_costo(datos, MODELO, VALLE)
    en_punta = costos.calcular_costo(datos, MODELO, PUNTA)

    assert en_punta == pytest.approx(en_valle * 2)


# ---------------------------------------------------------------------------
# Ventanas de hora punta
# ---------------------------------------------------------------------------

def test_la_ventana_incluye_el_inicio_y_excluye_el_fin():
    """A las 01:00 ya es punta; a las 04:00 ya es valle."""
    inicio = datetime(2026, 10, 28, 1, 0, tzinfo=timezone.utc)
    fin = datetime(2026, 10, 28, 4, 0, tzinfo=timezone.utc)

    assert costos.es_hora_punta(inicio) is True
    assert costos.es_hora_punta(fin) is False


def test_la_segunda_ventana_del_dia_tambien_es_punta():
    """06:00-10:00 UTC, no solo 01:00-04:00."""
    dentro = datetime(2026, 10, 28, 9, 0, tzinfo=timezone.utc)

    assert costos.es_hora_punta(dentro) is True


def test_fuera_de_las_ventanas_es_valle():
    """A las 12:00 UTC no hay ninguna ventana abierta."""
    assert costos.es_hora_punta(VALLE) is False


def test_fin_de_semana_es_valle_todo_el_dia():
    """Las ventanas son de lunes a viernes; el sábado a las 08:00 es valle."""
    assert costos.es_hora_punta(SABADO) is False


# ---------------------------------------------------------------------------
# Modelos sin tarifas
# ---------------------------------------------------------------------------

def test_calcular_costo_de_un_modelo_sin_tarifas_falla_fuerte():
    """Calcular es una función pura: ante un modelo desconocido, avisa."""
    with pytest.raises(ValueError, match="No hay tarifas registradas"):
        costos.calcular_costo({}, "modelo-inexistente", VALLE)


def test_registrar_costo_de_un_modelo_sin_tarifas_devuelve_cero_y_avisa(caplog):
    """Registrar no puede tumbar una respuesta ya pagada, pero sí dejar rastro."""
    with caplog.at_level(logging.WARNING):
        monto = costos.registrar_costo({}, "modelo-inexistente", VALLE)

    assert monto == 0.0
    assert "sin costo registrado" in caplog.text


# ---------------------------------------------------------------------------
# Acumulado de la sesión
# ---------------------------------------------------------------------------

def test_registrar_costo_acumula_entre_operaciones():
    """Dos operaciones de 1M de entrada sin caché suman el doble."""
    datos = _uso(entrada=1_000_000, cache_hit=0, cache_miss=1_000_000)

    primera = costos.registrar_costo(datos, MODELO, VALLE)
    segunda = costos.registrar_costo(datos, MODELO, VALLE)

    assert primera == pytest.approx(0.15)
    assert segunda == pytest.approx(0.15)
    assert costos.costo_acumulado() == pytest.approx(0.30)


def test_reiniciar_costo_deja_el_acumulado_en_cero():
    costos.registrar_costo(_uso(entrada=1_000_000, cache_hit=0), MODELO, VALLE)

    costos.reiniciar_costo()

    assert costos.costo_acumulado() == 0.0
