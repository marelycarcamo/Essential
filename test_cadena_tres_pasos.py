"""Pruebas de la chain de tres pasos (Bloque 18).

El modelo se mockea con MagicMock: así la cadena se prueba sin gastar
tokens ni depender de la API real. Esto es lo mismo que hicimos en el
Módulo 2 con el cliente, y el material lo pide explícitamente.
"""

import pytest
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser

from cadena_tres_pasos import (
    construir_cadena,
    plantilla_extraccion,
    plantilla_resumen,
    plantilla_formato,
    invocar_con_conmutador,
)


def test_la_chain_llama_el_modelo_tres_veces():
    """Tres pasos = tres llamadas al modelo, una por paso."""
    modelo_falso = MagicMock()
    modelo_falso.side_effect = [
        AIMessage(content="punto 1\npunto 2\npunto 3"),
        AIMessage(content="texto simplificado"),
        AIMessage(content="- viñeta 1\n- viñeta 2"),
    ]

    cadena = construir_cadena(modelo_falso)
    resultado = cadena.invoke({"texto": "texto de prueba"})
    assert modelo_falso.call_count == 3
    assert "viñeta" in resultado


def test_el_paso_2_recibe_la_salida_del_paso_1():
    modelo_falso = MagicMock()
    modelo_falso.side_effect = [
        AIMessage(content="idea clave del texto"),
        AIMessage(content="resumen simple"),
        AIMessage(content="- viñeta"),
    ]
    construir_cadena(modelo_falso).invoke({"texto": "texto"})
    segundo_prompt = modelo_falso.call_args_list[1].args[0]
    assert "idea clave del texto" in segundo_prompt.text


def test_el_paso_3_recibe_el_resumen_del_paso_2():
    modelo_falso = MagicMock()
    modelo_falso.side_effect = [
        AIMessage(content="puntos"),
        AIMessage(content="resumen del paso 2"),
        AIMessage(content="- viñeta"),
    ]
    construir_cadena(modelo_falso).invoke({"texto": "texto"})
    tercer_prompt = modelo_falso.call_args_list[2].args[0]
    assert "resumen del paso 2" in tercer_prompt.text


def test_el_conmutador_pasa_al_siguiente_modelo_si_el_principal_falla():
    modelo_principal = MagicMock()
    modelo_principal.side_effect = RuntimeError("503 unavailable")
    modelo_fallo = MagicMock()
    modelo_fallo.side_effect = RuntimeError("503 unavailable")
    modelo_ok = MagicMock()
    modelo_ok.side_effect = [
        AIMessage(content="punto 1\npunto 2\npunto 3"),
        AIMessage(content="texto simplificado"),
        AIMessage(content="- viñeta 1\n- viñeta 2"),
    ]

    resultado = invocar_con_conmutador(
        modelo_principal, [modelo_fallo, modelo_ok], {"texto": "x"}
    )
    assert "viñeta" in resultado
    assert modelo_principal.call_count == 1
    assert modelo_fallo.call_count == 1
    assert modelo_ok.call_count == 3


def test_el_conmutador_lanza_si_ningun_modelo_funciona():
    modelo_principal = MagicMock()
    modelo_principal.side_effect = RuntimeError("503")
    modelo_fallo = MagicMock()
    modelo_fallo.side_effect = RuntimeError("503")

    with pytest.raises(RuntimeError):
        invocar_con_conmutador(modelo_principal, [modelo_fallo], {"texto": "x"})


@pytest.mark.integracion
def test_la_chain_real_con_gemini():
    import os
    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("sin GEMINI_API_KEY")
    from langchain_google_genai import ChatGoogleGenerativeAI
    modelo = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)
    resultado = construir_cadena(modelo).invoke({"texto": "Hello world"})
    assert isinstance(resultado, str)
    assert len(resultado) > 0
