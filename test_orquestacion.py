"""Verificación de la orquestación, de extremo a extremo (Bloque 20).

Cubre dos cosas que se rompen sin avisar:

1. **El enrutamiento.** Que cada nivel se vaya a su ruta y que un nivel
   inesperado no deje la cadena sin destino.
2. **La resistencia.** Que un fallo del proveedor se resuelva con el respaldo,
   que sin respaldo haya una respuesta honesta, y que un error de
   programación **no** se disfrace de caída del servicio.

Todo con modelos simulados: ninguna prueba toca la red ni consume cuota.
"""

import logging

import pytest
import requests
from types import SimpleNamespace
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage
from langchain_google_genai.chat_models import (
    GoogleModelNotFoundError,
    GoogleRateLimitError,
)

from enrutamiento import (
    MAXIMO_PALABRAS,
    PLANTILLA_AVANZADA,
    PLANTILLA_INICIAL,
    PLANTILLA_INTERMEDIA,
    REINTENTOS_POR_DEFECTO,
    RESPUESTA_GENERICA,
    SECCIONES_OBLIGATORIAS,
    TIEMPO_MAXIMO_POR_DEFECTO,
    adaptar_respaldo,
    construir_cadena_con_enrutamiento,
    construir_cadena_segura,
    construir_modelo_gemini,
    construir_rutas,
    interpretar_nivel,
    verificar_material,
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


# --------------------------------------------------------------------------
# El nivel como parámetro: determinismo sin gastar una llamada
# --------------------------------------------------------------------------


def test_con_nivel_explicito_no_se_clasifica():
    rutas = _rutas_falsas()
    modelo = _modelo_con([])  # sin respuestas: si clasificara, reventaría

    resultado = construir_cadena_con_enrutamiento(
        modelo, rutas, nivel="avanzado"
    ).invoke({"texto": "documento"})

    assert resultado == "material avanzado"
    assert modelo.call_count == 0


def test_un_nivel_invalido_cae_en_el_default_y_no_rompe():
    rutas = _rutas_falsas()

    resultado = construir_cadena_con_enrutamiento(
        _modelo_con([]), rutas, nivel="experto"
    ).invoke({"texto": "documento"})

    assert resultado == "material intermedio"
    assert rutas["intermedio"].call_count == 1


def test_un_nivel_invalido_deja_aviso_en_el_log(caplog):
    with caplog.at_level(logging.WARNING, logger="enrutamiento"):
        construir_cadena_con_enrutamiento(
            _modelo_con([]), _rutas_falsas(), nivel="experto"
        ).invoke({"texto": "documento"})

    assert "experto" in caplog.text


def test_la_cadena_segura_propaga_el_nivel_al_respaldo():
    modelo_principal = MagicMock()
    modelo_principal.side_effect = requests.exceptions.ConnectionError("503")

    modelo_respaldo = _modelo_con(["material del respaldo"])

    cadena = construir_cadena_segura(
        modelo_principal, modelo_respaldo, nivel="inicial"
    )

    assert cadena.invoke({"texto": "documento"}) == "material del respaldo"
    # Una sola llamada: el respaldo no tuvo que clasificar el documento.
    assert modelo_respaldo.call_count == 1


def test_adaptar_respaldo_entrega_un_mensaje_y_no_un_objeto_crudo():
    """Regresión de la corrida real: el respaldo devolvía SimpleNamespace.

    ``StrOutputParser`` solo acepta texto o ``BaseMessage``, así que el camino
    de emergencia reventaba dentro del propio mecanismo de emergencia.
    """

    class ModeloCrudo:
        def invoke(self, entrada):
            return SimpleNamespace(content="material del respaldo")

    cadena = construir_cadena_con_enrutamiento(
        adaptar_respaldo(ModeloCrudo()), nivel="inicial"
    )

    assert cadena.invoke({"texto": "documento"}) == "material del respaldo"


def test_el_fallo_del_principal_queda_registrado(caplog):
    """``with_fallbacks`` reintenta en silencio: el log tiene que decirlo."""

    modelo_principal = MagicMock()
    modelo_principal.side_effect = requests.exceptions.ConnectionError("503")

    cadena = construir_cadena_segura(
        modelo_principal, _modelo_con(["material del respaldo"]), nivel="inicial"
    )

    with caplog.at_level(logging.WARNING, logger="enrutamiento"):
        assert cadena.invoke({"texto": "documento"}) == "material del respaldo"

    assert "modelo principal" in caplog.text
    assert "ConnectionError" in caplog.text


def test_un_429_de_cuota_activa_el_respaldo():
    """Regresión del 2026-11-14: el 429 se escapaba de la cadena.

    LangChain normaliza el 429 de Google en ``GoogleRateLimitError``, que no
    hereda de ``google.genai.APIError``. Era el único error de proveedor que no
    estaba en la lista, y es el más frecuente de todos.
    """

    modelo_principal = MagicMock()
    modelo_principal.side_effect = GoogleRateLimitError(
        "429 cuota diaria agotada"
    )

    cadena = construir_cadena_segura(
        modelo_principal, _modelo_con(["material del respaldo"]), nivel="inicial"
    )

    assert cadena.invoke({"texto": "documento"}) == "material del respaldo"


def test_un_404_de_configuracion_no_se_disfraza_de_caida():
    """Un modelo mal escrito tiene que doler, no rotar en silencio."""

    modelo_principal = MagicMock()
    modelo_principal.side_effect = GoogleModelNotFoundError(
        "404 modelo inexistente"
    )

    cadena = construir_cadena_segura(
        modelo_principal, _modelo_con(["material del respaldo"]), nivel="inicial"
    )

    with pytest.raises(GoogleModelNotFoundError):
        cadena.invoke({"texto": "documento"})


# --------------------------------------------------------------------------
# Verificación del material: el contrato, medido
# --------------------------------------------------------------------------

DOCUMENTO = (
    "ALESSIA es un chatbot RAG construido con Python y LangGraph.\n"
    "Divide los PDFs en fragmentos con solapamiento y los guarda en un "
    "índice vectorial."
)

MATERIAL_VALIDO = (
    "**Objetivo:** entender el pipeline de ALESSIA.\n"
    "**Conceptos:**\n"
    "1. **Fragmentación con solapamiento:** como cortar un libro en hojas que "
    "comparten un párrafo.\n"
    "> Divide los PDFs en fragmentos con solapamiento\n"
    "2. **Índice vectorial:** como el lomo que dice en qué página va cada "
    "tema.\n"
    "> ALESSIA es un chatbot RAG construido con Python y LangGraph\n"
    "**Ejercicio:** dibuja el pipeline en cuatro cajas.\n"
)


def test_verificar_material_aprueba_uno_bien_formado():
    revision = verificar_material(MATERIAL_VALIDO, DOCUMENTO)

    assert revision.aprueba
    assert revision.secciones_faltantes == ()
    assert revision.citas_invalidas == ()
    assert revision.palabras > 0


@pytest.mark.parametrize("seccion", SECCIONES_OBLIGATORIAS)
def test_verificar_material_detecta_cada_seccion_faltante(seccion):
    sin_seccion = "\n".join(
        linea
        for linea in MATERIAL_VALIDO.splitlines()
        if not linea.lower().lstrip("*# ").startswith(seccion)
    )

    revision = verificar_material(sin_seccion, DOCUMENTO)

    assert seccion in revision.secciones_faltantes
    assert not revision.aprueba


def test_verificar_material_detecta_una_cita_inventada():
    """Una cita que no está en el documento es la alucinación más barata de ver."""

    material = MATERIAL_VALIDO.replace(
        "ALESSIA es un chatbot RAG construido con Python y LangGraph",
        "ALESSIA usa GPT-4 y no tiene pruebas",
    )

    revision = verificar_material(material, DOCUMENTO)

    assert len(revision.citas_invalidas) == 1
    assert "GPT-4" in revision.citas_invalidas[0]
    assert not revision.aprueba


def test_verificar_material_acepta_una_cita_que_cruza_un_salto_de_linea():
    """Sin colapsar espacios, una cita correcta se marcaría como inventada."""

    material = MATERIAL_VALIDO.replace(
        "Divide los PDFs en fragmentos con solapamiento",
        "Python y LangGraph. Divide los PDFs",
    )

    revision = verificar_material(material, DOCUMENTO)

    assert revision.citas_invalidas == ()


def test_verificar_material_no_cuenta_las_citas_como_palabras_propias():
    cita_larga = "> " + " ".join(["palabra"] * 50) + "\n"

    con_cita = verificar_material(MATERIAL_VALIDO + cita_larga, DOCUMENTO)
    sin_cita = verificar_material(MATERIAL_VALIDO, DOCUMENTO)

    assert con_cita.palabras == sin_cita.palabras


def test_verificar_material_rechaza_un_material_demasiado_largo():
    relleno = " ".join(["palabra"] * (MAXIMO_PALABRAS + 5))
    material = MATERIAL_VALIDO + f"\n{relleno}\n"

    revision = verificar_material(material, DOCUMENTO)

    assert revision.palabras > MAXIMO_PALABRAS
    assert not revision.aprueba


@pytest.mark.parametrize(
    "plantilla",
    [PLANTILLA_INICIAL, PLANTILLA_INTERMEDIA, PLANTILLA_AVANZADA],
    ids=["inicial", "intermedia", "avanzada"],
)
def test_cada_plantilla_pide_el_formato_y_la_cita(plantilla):
    """El contrato vive en un solo sitio; las tres rutas lo heredan."""

    texto = plantilla.template.lower()

    for seccion in SECCIONES_OBLIGATORIAS:
        assert f"**{seccion}:" in texto

    assert "cita literal" in texto


# --------------------------------------------------------------------------
# Configuración del modelo: el techo tiene que ser el que dice ser
# --------------------------------------------------------------------------


def test_el_modelo_queda_sin_reintentos_ocultos(monkeypatch):
    """Regresión del 2026-11-14: con ``max_retries=6`` el techo era timeout × 7.

    Se midió una llamada de 787 s con ``timeout=240``. El número que se lee en
    la configuración tiene que ser el que de verdad se aplica.
    """

    monkeypatch.delenv("LLM_MAX_REINTENTOS", raising=False)
    monkeypatch.delenv("LLM_TIMEOUT_SECONDS", raising=False)

    modelo = construir_modelo_gemini(clave="clave-de-prueba")

    assert modelo.max_retries == REINTENTOS_POR_DEFECTO == 0
    assert modelo.timeout == TIEMPO_MAXIMO_POR_DEFECTO


def test_el_techo_del_modelo_se_puede_configurar(monkeypatch):
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "15")
    monkeypatch.setenv("LLM_MAX_REINTENTOS", "1")

    modelo = construir_modelo_gemini(clave="clave-de-prueba")

    assert modelo.timeout == 15
    assert modelo.max_retries == 1
