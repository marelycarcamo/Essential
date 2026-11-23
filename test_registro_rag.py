"""Tests de la instrumentación de latencia y costo (Bloque 28).

Qué se prueba y qué no
----------------------
Se prueban la traducción del uso de LangChain al formato del proveedor, el
cálculo de costo con las tarifas del proyecto, el cronómetro por etapa, el
formato JSON Lines y el resumen. El modelo y el retriever se reemplazan por
dobles, así que la medición del reloj y de los tokens se verifica sin red.

Lo que no se prueba aquí es la latencia real del proveedor: eso es la corrida de
``registro_rag.py``, cuyos números viven en ``registros_latencia_costo.jsonl``.
"""

import json
from datetime import datetime, timezone

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from registro_rag import (
    agregar_registro,
    calcular_costo_de_uso,
    leer_registros,
    medir_tiempo,
    normalizar_uso,
    preguntas_de_ejemplo,
    registrar_consulta,
    resumir_registros,
)

# Lunes 00:00 UTC: fuera de las ventanas de punta, así que la tarifa es valle y
# el resultado del test no depende de la hora en que se corra.
MOMENTO_VALLE = datetime(2026, 11, 23, 0, 0, tzinfo=timezone.utc)


class RetrieverFalso:
    """Retriever que devuelve dos fragmentos con fuente conocida."""

    def as_retriever(self, search_kwargs=None):
        return self

    def invoke(self, pregunta: str) -> list[Document]:
        return [
            Document(page_content="El cliente reintenta.", metadata={"fuente": "a.md"}, id="a#0"),
            Document(page_content="El costo se calcula.", metadata={"fuente": "b.md"}, id="b#0"),
        ]


class ModeloConUso(BaseChatModel):
    """Modelo de chat que informa tokens, como un proveedor real."""

    contenido: str = "Respuesta de prueba [1]."
    uso: dict = Field(
        default_factory=lambda: {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "input_token_details": {"cache_read": 4},
        }
    )

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content=self.contenido, usage_metadata=self.uso))
            ]
        )

    @property
    def _llm_type(self) -> str:
        return "modelo-con-uso-de-prueba"


class ModeloSinUso(ModeloConUso):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="sin uso"))])


# --- Normalización del uso -------------------------------------------------


def test_normalizar_uso_traduce_las_claves_de_langchain():
    mensaje = AIMessage(
        content="x",
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 40,
            "total_tokens": 140,
            "input_token_details": {"cache_read": 30},
        },
    )

    uso = normalizar_uso(mensaje)

    assert uso["prompt_tokens"] == 100
    assert uso["completion_tokens"] == 40
    assert uso["prompt_cache_hit_tokens"] == 30
    assert uso["prompt_cache_miss_tokens"] == 70


def test_normalizar_uso_sin_desglose_de_cache_cobra_todo_como_miss():
    mensaje = AIMessage(
        content="x",
        usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110},
    )

    uso = normalizar_uso(mensaje)

    assert uso["prompt_cache_hit_tokens"] == 0
    assert uso["prompt_cache_miss_tokens"] == 100


def test_normalizar_uso_sin_metadatos_devuelve_vacio():
    assert normalizar_uso(AIMessage(content="x")) == {}


# --- Costo -----------------------------------------------------------------


def test_costo_sin_datos_de_uso_es_cero():
    assert calcular_costo_de_uso({}, "deepseek-flash", MOMENTO_VALLE) == 0.0


def test_costo_usa_las_tarifas_por_componente():
    uso = {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "prompt_cache_hit_tokens": 4,
        "prompt_cache_miss_tokens": 6,
    }

    costo = calcular_costo_de_uso(uso, "deepseek-flash", MOMENTO_VALLE)

    # Valle: 4*0,003 + 6*0,15 + 5*0,60 = 3,912 por millón de tokens.
    assert costo == pytest.approx(3.912e-6)


def test_costo_falla_si_el_modelo_no_tiene_tarifas():
    uso = {"prompt_tokens": 10, "completion_tokens": 5}

    with pytest.raises(ValueError, match="tarifas"):
        calcular_costo_de_uso(uso, "modelo-inventado", MOMENTO_VALLE)


# --- Cronómetro ------------------------------------------------------------


def test_medir_tiempo_registra_la_etapa():
    tiempos: dict[str, float] = {}

    with medir_tiempo(tiempos, "generacion"):
        pass

    assert "generacion" in tiempos
    assert tiempos["generacion"] >= 0


def test_medir_tiempo_registra_aunque_la_etapa_falle():
    tiempos: dict[str, float] = {}

    with pytest.raises(RuntimeError):
        with medir_tiempo(tiempos, "generacion"):
            raise RuntimeError("falló la llamada")

    assert "generacion" in tiempos


# --- Registro completo -----------------------------------------------------


def test_registrar_consulta_devuelve_todas_las_claves():
    registro = registrar_consulta(
        "¿qué hace el cliente?",
        RetrieverFalso(),
        ModeloConUso(),
        "deepseek-flash",
        k=2,
        momento=MOMENTO_VALLE,
    )

    assert registro["pregunta"] == "¿qué hace el cliente?"
    assert registro["k"] == 2
    assert set(registro["tiempos_ms"]) == {"embed_y_busqueda", "generacion"}
    assert registro["tiempo_total_ms"] >= sum(registro["tiempos_ms"].values()) - 1
    assert registro["tokens"]["prompt_tokens"] == 10
    assert registro["costo_usd"] > 0
    assert registro["fuentes"] == ["a.md", "b.md"]
    assert registro["timestamp"] == MOMENTO_VALLE.isoformat()


def test_registrar_consulta_sin_uso_deja_tokens_en_none():
    registro = registrar_consulta(
        "¿algo?", RetrieverFalso(), ModeloSinUso(), "deepseek-flash", k=1, momento=MOMENTO_VALLE
    )

    assert registro["tokens"] is None
    assert registro["costo_usd"] == 0.0


# --- Archivo JSON Lines ----------------------------------------------------


def test_agregar_registro_agrega_una_linea_por_corrida(tmp_path):
    destino = tmp_path / "registros.jsonl"

    agregar_registro({"pregunta": "uno", "k": 3}, destino)
    agregar_registro({"pregunta": "dos", "k": 3}, destino)

    lineas = destino.read_text(encoding="utf-8").strip().splitlines()
    assert len(lineas) == 2
    assert json.loads(lineas[0])["pregunta"] == "uno"
    assert json.loads(lineas[1])["pregunta"] == "dos"


def test_leer_registros_devuelve_los_objetos(tmp_path):
    destino = tmp_path / "registros.jsonl"
    agregar_registro({"pregunta": "uno"}, destino)

    registros = leer_registros(destino)

    assert registros == [{"pregunta": "uno"}]


def test_leer_registros_falla_si_no_existe(tmp_path):
    with pytest.raises(FileNotFoundError):
        leer_registros(tmp_path / "no-existe.jsonl")


def test_resumir_registros_suma_el_costo():
    registros = [
        {"k": 3, "tiempo_total_ms": 900.0, "costo_usd": 0.00001, "pregunta": "uno", "tokens": {"prompt_tokens": 100}},
        {"k": 3, "tiempo_total_ms": 800.0, "costo_usd": 0.00002, "pregunta": "dos", "tokens": {"prompt_tokens": 120}},
    ]

    resumen = resumir_registros(registros)

    assert "2 corridas" in resumen
    assert "0.00003000" in resumen


def test_resumir_registros_sin_corridas():
    assert resumir_registros([]) == "Sin registros"


# --- Selección de preguntas ------------------------------------------------


def test_preguntas_de_ejemplo_toma_solo_respondibles():
    preguntas = preguntas_de_ejemplo(3)

    assert len(preguntas) == 3
    assert all("licencia" not in pregunta.lower() for pregunta in preguntas)
    assert all("alumnos" not in pregunta.lower() for pregunta in preguntas)
