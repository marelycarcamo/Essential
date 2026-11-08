"""Pruebas de la chain de tres pasos (Bloque 18).

El modelo se mockea con ``MagicMock``: así la cadena se prueba sin gastar
tokens ni depender de la API real, igual que en el cliente del Módulo 2.

Lo que estas pruebas cuidan no es el texto que devuelve el modelo —eso cambia
en cada corrida— sino la **composición**: que se llame tres veces, que cada
paso reciba lo que el anterior produjo, que las claves coincidan y que un
error de programación no se confunda con un proveedor caído.
"""

import os

import pytest
import requests
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from cadena_tres_pasos import (
    PLANTILLA_ANALISIS,
    PLANTILLA_MATERIAL,
    PLANTILLA_PLAN,
    construir_cadena,
    interpretar_ficha,
    invocar_con_conmutador,
)


FICHA = "README|avanzado|chunks; embeddings; reranking|Programar en Python"
PLAN = (
    "Objetivo: distinguir las tres etapas del pipeline.\n"
    "1. Chunks, como cortar un libro en fichas.\n"
    "2. Embeddings, como poner cada ficha en un mapa.\n"
    "3. Reranking, como ordenar las fichas por utilidad.\n"
    "Ejercicio: elegir el fragmento más útil para una consulta."
)
MATERIAL = "# Chunks\n\nUn chunk es un fragmento de documento.\n\n# Ejercicio\n\nElige un fragmento."


def _modelo_con(respuestas: list[str]) -> MagicMock:
    """MagicMock que responde en orden, como un modelo real encadenado."""

    modelo = MagicMock()
    modelo.side_effect = [AIMessage(content=respuesta) for respuesta in respuestas]

    return modelo


# --------------------------------------------------------------------------
# Composición de la cadena
# --------------------------------------------------------------------------


def test_la_chain_llama_el_modelo_tres_veces():
    """Tres pasos = tres llamadas, una por paso."""
    modelo_falso = _modelo_con([FICHA, PLAN, MATERIAL])

    resultado = construir_cadena(modelo_falso).invoke({"texto": "texto de prueba"})

    assert modelo_falso.call_count == 3
    assert resultado == MATERIAL


def test_el_paso_2_recibe_la_ficha_del_paso_1():
    modelo_falso = _modelo_con([FICHA, PLAN, MATERIAL])

    construir_cadena(modelo_falso).invoke({"texto": "texto de prueba"})

    segundo_prompt = modelo_falso.call_args_list[1].args[0]
    # No llega la línea cruda: llega la ficha ya interpretada, con el nivel
    # normalizado y los conceptos separados.
    assert "- Nivel del público: avanzado" in segundo_prompt.text
    assert "chunks; embeddings; reranking" in segundo_prompt.text


def test_el_paso_3_recibe_el_plan_del_paso_2():
    modelo_falso = _modelo_con([FICHA, PLAN, MATERIAL])

    construir_cadena(modelo_falso).invoke({"texto": "texto de prueba"})

    tercer_prompt = modelo_falso.call_args_list[2].args[0]
    assert PLAN in tercer_prompt.text


def test_construir_cadena_no_ejecuta_el_modelo():
    modelo_falso = _modelo_con([FICHA])

    construir_cadena(modelo_falso)

    # Armar la cadena no debe gastar una llamada: la ejecución es aparte.
    assert modelo_falso.call_count == 0


def test_las_plantillas_declaran_las_variables_que_usan():
    # Si se renombra una variable en el texto y no en el paso que la entrega,
    # la cadena falla con KeyError en tiempo de ejecución. Esto lo detecta sin
    # llamar al modelo.
    assert PLANTILLA_ANALISIS.input_variables == ["texto"]
    assert set(PLANTILLA_PLAN.input_variables) == {
        "tipo",
        "nivel",
        "conceptos",
        "requisito",
    }
    assert PLANTILLA_MATERIAL.input_variables == ["plan"]


# --------------------------------------------------------------------------
# Interpretación de la ficha
# --------------------------------------------------------------------------


def test_interpretar_ficha_arma_los_cuatro_campos():
    assert interpretar_ficha(FICHA) == {
        "tipo": "README",
        "nivel": "avanzado",
        "conceptos": "chunks; embeddings; reranking",
        "requisito": "Programar en Python",
    }


@pytest.mark.parametrize(
    ("entrada", "nivel_esperado"),
    [
        ("Guía|Avanzado|a; b; c|nada", "avanzado"),
        ("README|  INICIAL |a; b; c|nada", "inicial"),
        ("docstring|Intermedio|a; b; c|nada", "intermedio"),
    ],
)
def test_interpretar_ficha_normaliza_el_nivel(entrada, nivel_esperado):
    assert interpretar_ficha(entrada)["nivel"] == nivel_esperado


def test_interpretar_ficha_completa_campos_faltantes():
    ficha = interpretar_ficha("cualquier cosa sin separadores")

    assert ficha["tipo"] == "cualquier cosa sin separadores"
    assert ficha["nivel"] == "intermedio"
    assert ficha["conceptos"] == "no especificados"
    assert ficha["requisito"] == "ninguno en particular"


def test_interpretar_ficha_rechaza_un_nivel_desconocido():
    assert interpretar_ficha("README|expertísimo|a; b; c|nada")["nivel"] == "intermedio"


# --------------------------------------------------------------------------
# Conmutador de modelos
# --------------------------------------------------------------------------


def test_el_conmutador_pasa_al_siguiente_modelo_si_el_principal_falla():
    modelo_principal = MagicMock()
    modelo_principal.side_effect = requests.exceptions.ConnectionError("503")

    modelo_ok = _modelo_con([FICHA, PLAN, MATERIAL])

    resultado = invocar_con_conmutador(
        modelo_principal, [modelo_ok], {"texto": "x"}
    )

    assert resultado == MATERIAL
    assert modelo_principal.call_count == 1
    assert modelo_ok.call_count == 3


def test_el_conmutador_no_esconde_un_error_de_programacion():
    """Un KeyError es culpa nuestra, no del proveedor: debe verse."""
    modelo_principal = MagicMock()
    modelo_principal.side_effect = KeyError("plan")

    respaldo = _modelo_con([FICHA, PLAN, MATERIAL])

    with pytest.raises(KeyError):
        invocar_con_conmutador(modelo_principal, [respaldo], {"texto": "x"})

    # Y no debe haber intentado el respaldo: rotar de modelo no arregla un
    # diccionario mal armado.
    assert respaldo.call_count == 0


def test_el_conmutador_lanza_si_ningun_modelo_funciona():
    modelo_principal = MagicMock()
    modelo_principal.side_effect = requests.exceptions.ConnectionError("503")

    with pytest.raises(RuntimeError) as error:
        invocar_con_conmutador(modelo_principal, [], {"texto": "x"})

    # La causa original queda encadenada: sin esto, el mensaje final no dice
    # qué pasó realmente.
    assert isinstance(error.value.__cause__, requests.exceptions.ConnectionError)


# --------------------------------------------------------------------------
# Integración (excluida por defecto: ``pytest.ini`` filtra la marca)
# --------------------------------------------------------------------------


@pytest.mark.integracion
def test_la_chain_real_con_gemini():
    from langchain_google_genai import ChatGoogleGenerativeAI

    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("sin GEMINI_API_KEY")

    modelo = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)

    resultado = construir_cadena(modelo).invoke(
        {"texto": "Un pipeline RAG convierte documentos en fragmentos y los busca por similitud."}
    )

    assert isinstance(resultado, str)
    assert resultado.strip()
