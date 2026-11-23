"""Tests de la recuperación y del prompt con citas (Bloque 26).

Qué se prueba y qué no
----------------------
Se prueba la estructura de la consulta: numeración del contexto, mapa de citas,
respeto de ``k``, validación de la pregunta y detección de citas inventadas. El
índice se construye de verdad en un directorio temporal con embeddings falsos,
así que la persistencia y la reconexión se ejercitan igual que en producción,
sin red.

Lo que no se prueba aquí es la calidad semántica de la recuperación: con
embeddings falsos el orden de los fragmentos no significa nada. Eso se mide en
el Bloque 27 con precisión y recall sobre el índice real.

El modelo de generación se inyecta como doble. La prueba de que el modelo real
dice "no tengo información suficiente" es una corrida manual, porque depende de
un servicio externo.
"""

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

import indexar_corpus as ic
from recuperacion import (
    ErrorConsulta,
    citas_usadas,
    construir_prompt,
    formatear_contexto,
    recuperar,
    responder,
    validar_pregunta,
    verificar_citas,
)
from test_indexar_corpus import EmbeddingsFalsos

FRAGMENTOS = [
    Document(
        page_content="El cliente reintenta hasta tres veces ante un error temporal.",
        metadata={"fuente": "01_cliente_modular.md"},
    ),
    Document(
        page_content="El costo se calcula por componente de token.",
        metadata={"fuente": "02_costos.md"},
    ),
    Document(
        page_content="La suite corre en menos de cinco segundos.",
        metadata={"fuente": "05_pruebas.md"},
    ),
]


class ModeloFijo(BaseChatModel):
    """Modelo de chat que devuelve un texto prefijado, sin red."""

    respuesta: str = "Texto de prueba [1]."
    # ``default_factory`` y no ``[]``: un default mutable en un modelo se comparte
    # entre instancias, y el test pasaría por accidente según el orden de corrida.
    recibidos: list = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.recibidos.append(messages)
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.respuesta))]
        )

    @property
    def _llm_type(self) -> str:
        return "modelo-fijo-de-prueba"


@pytest.fixture
def indice(tmp_path):
    """Índice real en disco, con embeddings falsos y tres fragmentos conocidos."""

    embeddings = EmbeddingsFalsos()
    ic.indexar_corpus(embeddings, FRAGMENTOS, directorio=tmp_path)
    return ic.abrir_indice(embeddings, directorio=tmp_path)


# --- Formato del contexto --------------------------------------------------


def test_formatear_contexto_numera_desde_uno():
    contexto, citas = formatear_contexto(FRAGMENTOS)

    assert contexto.startswith("[1] ")
    assert "[3] " in contexto
    assert list(citas) == [1, 2, 3]


def test_formatear_contexto_asocia_cada_numero_a_su_archivo():
    _, citas = formatear_contexto(FRAGMENTOS)

    assert citas[1] == "01_cliente_modular.md"
    assert citas[2] == "02_costos.md"
    assert citas[3] == "05_pruebas.md"


def test_formatear_contexto_marca_desconocida_si_falta_metadata():
    _, citas = formatear_contexto([Document(page_content="sin metadata", metadata={})])

    assert citas[1] == "desconocida"


def test_formatear_contexto_no_inventa_bloques():
    contexto, _ = formatear_contexto(FRAGMENTOS)

    assert contexto.count("[1] ") == 1
    assert "[4]" not in contexto


# --- Prompt ----------------------------------------------------------------


def test_construir_prompt_incluye_la_pregunta_exacta(indice):
    pregunta = "¿Qué pasa si la llamada falla?"

    prompt, _ = construir_prompt(pregunta, indice)

    assert f"Pregunta: {pregunta}" in prompt


def test_construir_prompt_respeta_k(indice):
    prompt, citas = construir_prompt("¿algo?", indice, k=1)

    assert list(citas) == [1]
    assert "[2]" not in prompt


def test_construir_prompt_incluye_instruccion_de_citar(indice):
    prompt, _ = construir_prompt("¿algo?", indice)

    assert "SOLO el contexto" in prompt
    assert "dilo explícitamente" in prompt


@pytest.mark.parametrize("pregunta", ["", "   ", "\n\t "])
def test_construir_prompt_rechaza_pregunta_vacia(indice, pregunta):
    with pytest.raises(ErrorConsulta, match="vacía"):
        construir_prompt(pregunta, indice)


def test_construir_prompt_rechaza_pregunta_que_no_es_texto(indice):
    with pytest.raises(ErrorConsulta, match="debe ser texto"):
        construir_prompt(123, indice)


def test_validar_pregunta_recorta_espacios():
    assert validar_pregunta("  ¿qué es el backoff?  ") == "¿qué es el backoff?"


def test_recuperar_rechaza_k_invalido(indice):
    with pytest.raises(ErrorConsulta, match="al menos 1"):
        recuperar(indice, "¿algo?", k=0)


# --- Citas -----------------------------------------------------------------


def test_citas_usadas_extrae_los_numeros():
    assert citas_usadas("Según [1] y también [3], pero no [12].") == [1, 3, 12]


def test_citas_usadas_ignora_parentesis_y_anios():
    assert citas_usadas("En 1991 (primera versión) no hubo citas.") == []


def test_verificar_citas_acepta_citas_validas():
    assert verificar_citas("El costo es por componente [2].", {1: "a.md", 2: "b.md"}) == []


def test_verificar_citas_detecta_numero_inexistente():
    problemas = verificar_citas("Como dice [5].", {1: "a.md", 2: "b.md"})

    assert any("[5]" in problema for problema in problemas)


def test_verificar_citas_detecta_respuesta_sin_citas():
    problemas = verificar_citas("El costo es por componente.", {1: "a.md"})

    assert any("no cita" in problema for problema in problemas)


def test_verificar_citas_sin_mapa_no_puede_auditar():
    problemas = verificar_citas("Cualquier texto [1].", {})

    assert any("auditar" in problema for problema in problemas)


# --- Consulta completa -----------------------------------------------------


def test_responder_devuelve_la_trazabilidad_completa(indice):
    modelo = ModeloFijo(respuesta="Se reintenta hasta tres veces [1].")

    resultado = responder("¿qué hace el cliente ante un error?", indice, modelo)

    assert resultado["pregunta"] == "¿qué hace el cliente ante un error?"
    assert resultado["respuesta"] == "Se reintenta hasta tres veces [1]."
    assert resultado["citas"]
    assert resultado["problemas_citas"] == []
    assert "Contexto:" in resultado["prompt"]


def test_responder_marca_las_citas_inventadas(indice):
    modelo = ModeloFijo(respuesta="La respuesta está en [9].")

    resultado = responder("¿algo?", indice, modelo)

    assert any("[9]" in problema for problema in resultado["problemas_citas"])


def test_responder_entrega_el_prompt_al_modelo(indice):
    modelo = ModeloFijo()

    responder("¿qué es el backoff?", indice, modelo)

    assert modelo.recibidos
    assert "¿qué es el backoff?" in modelo.recibidos[0][0].content
