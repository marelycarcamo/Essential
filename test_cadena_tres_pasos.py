"""Pruebas de la chain de tres pasos (Bloque 18).

El modelo se mockea con MagicMock: así la cadena se prueba sin gastar
tokens ni depender de la API real. Esto es lo mismo que hicimos en el
Módulo 2 con el cliente, y el material lo pide explícitamente.
"""

import pytest
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from cadena_tres_pasos import construir_cadena


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

    # Debe haber llamado al modelo exactamente tres veces (uno por paso)
    assert modelo_falso.call_count == 3
    # El resultado final es un string que contiene el marcador del modelo del paso 3
    assert "viñeta" in resultado


def test_el_paso_2_recibe_la_salida_del_paso_1():
    """La chain es real: el paso 2 solo funciona si el paso 1 le pasó algo.

    Si el paso 2 ignora la salida del paso 1, no es una chain: son tres
    llamadas independientes disfrazadas.
    """
    modelo_falso = MagicMock()
    modelo_falso.side_effect = [
        AIMessage(content="idea clave del texto"),
        AIMessage(content="resumen simple"),
        AIMessage(content="- viñeta"),
    ]

    construir_cadena(modelo_falso).invoke({"texto": "texto"})

    # El segundo prompt recibe la salida del primero (no un texto fijo).
    segundo_prompt = modelo_falso.call_args_list[1].args[0]
    assert "idea clave del texto" in segundo_prompt.text


def test_el_paso_3_recibe_el_resumen_del_paso_2():
    """Mismo razonamiento para el tercer paso: depende del segundo."""
    modelo_falso = MagicMock()
    modelo_falso.side_effect = [
        AIMessage(content="puntos"),
        AIMessage(content="resumen del paso 2"),
        AIMessage(content="- viñeta"),
    ]

    construir_cadena(modelo_falso).invoke({"texto": "texto"})

    tercer_prompt = modelo_falso.call_args_list[2].args[0]
    assert "resumen del paso 2" in tercer_prompt.text