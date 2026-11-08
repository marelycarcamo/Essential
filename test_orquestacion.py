"""Verificación de la orquestación, de extremo a extremo (Bloque 20).

Cubre dos cosas que se rompen sin avisar:

1. **El enrutamiento.** Que cada nivel se vaya a su ruta y que un nivel
   inesperado no deje la cadena sin destino.
2. **La resistencia.** Que un fallo del proveedor se resuelva con el respaldo,
   que sin respaldo haya una respuesta honesta, y que un error de
   programación **no** se disfrace de caída del servicio.

Todo con modelos simulados: ninguna prueba toca la red ni consume cuota.
"""

import pytest
import requests
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from enrutamiento import (
    PLANTILLA_AVANZADA,
    PLANTILLA_INICIAL,
    PLANTILLA_INTERMEDIA,
    RESPUESTA_GENERICA,
    construir_cadena_con_enrutamiento,
    construir_cadena_segura,
    construir_rutas,
    interpretar_nivel,
)


def _modelo_con(respuestas: list[str]) -> MagicMock:
    """MagicMock que responde en orden, como un modelo real."""

    modelo = MagicMock()
    modelo.side_effect = [AIMessage(content=respuesta) for respuesta in respuestas]

    return modelo


def _rutas_falsas() -> dict[str, MagicMock]:
    """Tres rutas simuladas, para ver a cuál se fue la cadena."""

    return {
        "inicial": MagicMock(return_value="material inicial"),
        "intermedio": MagicMock(return_value="material intermedio"),
        "avanzado": MagicMock(return_value="material avanzado"),
    }


# --------------------------------------------------------------------------
# Clasificación del nivel
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("respuesta_cruda", "esperado"),
    [
        ("inicial", "inicial"),
        ("  AVANZADO \n", "avanzado"),
        ("Intermedio.", "intermedio"),
    ],
)
def test_interpretar_nivel_normaliza(respuesta_cruda, esperado):
    assert interpretar_nivel(respuesta_cruda) == esperado


def test_interpretar_nivel_cae_en_el_default_si_no_reconoce():
    assert interpretar_nivel("no tengo idea") == "intermedio"
    assert interpretar_nivel("") == "intermedio"


# --------------------------------------------------------------------------
# Las tres rutas son distintas
# --------------------------------------------------------------------------


def test_las_tres_rutas_usan_plantillas_distintas():
    plantillas = [PLANTILLA_INICIAL, PLANTILLA_INTERMEDIA, PLANTILLA_AVANZADA]

    assert len({plantilla.template for plantilla in plantillas}) == 3
    for plantilla in plantillas:
        # Cada ruta conserva el documento original en su plantilla.
        assert plantilla.input_variables == ["texto"]


def test_construir_rutas_arma_una_por_nivel():
    rutas = construir_rutas(MagicMock())

    assert set(rutas) == {"inicial", "intermedio", "avanzado"}


# --------------------------------------------------------------------------
# La bifurcación
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("nivel", "ruta_esperada"),
    [
        ("inicial", "inicial"),
        ("avanzado", "avanzado"),
        ("desconocido", "intermedio"),
    ],
)
def test_la_cadena_va_a_la_ruta_que_corresponde(nivel, ruta_esperada):
    rutas = _rutas_falsas()
    modelo = _modelo_con([nivel])

    resultado = construir_cadena_con_enrutamiento(modelo, rutas).invoke(
        {"texto": "documento de prueba"}
    )

    assert rutas[ruta_esperada].call_count == 1
    assert resultado == f"material {ruta_esperada}"

    for nombre, ruta in rutas.items():
        if nombre != ruta_esperada:
            assert ruta.call_count == 0, f"también se ejecutó la ruta {nombre}"


def test_la_ruta_recibe_el_documento_original():
    """``RunnablePassthrough.assign`` conserva el texto de la entrada."""

    rutas = _rutas_falsas()
    modelo = _modelo_con(["inicial"])

    construir_cadena_con_enrutamiento(modelo, rutas).invoke(
        {"texto": "texto que no se debe perder"}
    )

    entrada_recibida = rutas["inicial"].call_args.args[0]

    assert entrada_recibida["texto"] == "texto que no se debe perder"
    assert entrada_recibida["nivel"] == "inicial"


# --------------------------------------------------------------------------
# Resistencia: fallbacks y respuesta degradada
# --------------------------------------------------------------------------


def test_la_cadena_completa_hace_dos_llamadas_y_una_sola_ruta():
    modelo = _modelo_con(["avanzado", "material para quien ya usa la herramienta"])

    resultado = construir_cadena_con_enrutamiento(modelo).invoke(
        {"texto": "documento"}
    )

    assert resultado == "material para quien ya usa la herramienta"
    assert modelo.call_count == 2


def test_si_el_principal_falla_responde_el_respaldo():
    modelo_principal = MagicMock()
    modelo_principal.side_effect = requests.exceptions.ConnectionError("503")

    modelo_respaldo = _modelo_con(["inicial", "material del respaldo"])

    cadena = construir_cadena_segura(modelo_principal, modelo_respaldo)
    resultado = cadena.invoke({"texto": "documento"})

    assert resultado == "material del respaldo"
    assert modelo_respaldo.call_count == 2


def test_si_todos_fallan_se_entrega_la_respuesta_generica():
    modelo_principal = MagicMock()
    modelo_principal.side_effect = requests.exceptions.ConnectionError("503")

    cadena = construir_cadena_segura(modelo_principal)
    resultado = cadena.invoke({"texto": "documento"})

    assert resultado == RESPUESTA_GENERICA


def test_un_error_de_programacion_no_se_confunde_con_una_caida():
    """Un KeyError es nuestro, no del proveedor: tiene que verse."""

    modelo_principal = MagicMock()
    modelo_principal.side_effect = KeyError("texto")

    cadena = construir_cadena_segura(modelo_principal)

    with pytest.raises(KeyError):
        cadena.invoke({"texto": "documento"})
