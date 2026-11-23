"""Tests de la evaluación de la recuperación (Bloque 27).

Qué se prueba y qué no
----------------------
Se prueban las fórmulas (precisión, recall, promedio con casos sin respuesta),
el armado del archivo de resultados, la negativa a sobrescribir una corrida
anterior y la coherencia del conjunto de evaluación. El retriever se reemplaza
por un doble para fijar exactamente qué devuelve, así que el test mide la
fórmula y no el modelo de embeddings.

Lo que no se prueba aquí es si el índice real recupera bien: eso es la corrida
de ``evaluacion_recuperacion.py`` sobre el vector store verdadero, con sus
números en ``resultados_evaluacion.txt``.
"""

import pytest
from langchain_core.documents import Document

from evaluacion_recuperacion import (
    CASOS_EVALUACION,
    evaluar_recuperacion,
    escribir_resultados,
    formatear_resultados,
    precision_at_k,
    promediar,
    recall_at_k,
)

# --- Fórmulas --------------------------------------------------------------


def test_precision_sin_recuperados_es_cero():
    assert precision_at_k([], {"a"}) == 0.0


def test_precision_con_todos_relevantes_es_uno():
    assert precision_at_k(["a", "b"], {"a", "b"}) == 1.0


def test_precision_con_uno_de_tres():
    assert precision_at_k(["a", "x", "y"], {"a"}) == pytest.approx(1 / 3)


def test_precision_sin_relevantes_recuperados_es_cero():
    assert precision_at_k(["x", "y"], {"a"}) == 0.0


def test_recall_sin_relevantes_es_none():
    assert recall_at_k(["a"], set()) is None


def test_recall_con_todos_los_relevantes_es_uno():
    assert recall_at_k(["a", "x"], {"a"}) == 1.0


def test_recall_parcial():
    assert recall_at_k(["a"], {"a", "b"}) == 0.5


def test_promediar_ignora_los_none():
    assert promediar([1.0, None, 0.0]) == 0.5


def test_promediar_sin_valores_es_cero():
    assert promediar([None, None]) == 0.0


# --- Conjunto de evaluación ------------------------------------------------


def test_el_conjunto_tiene_entre_ocho_y_quince_casos():
    assert 8 <= len(CASOS_EVALUACION) <= 15


def test_el_conjunto_tiene_casos_sin_respuesta_en_el_corpus():
    sin_relevantes = [caso for caso in CASOS_EVALUACION if not caso["ids_relevantes"]]

    assert len(sin_relevantes) >= 1


def test_las_etiquetas_apuntan_a_archivos_del_corpus():
    from indexar_corpus import DIRECTORIO_CORPUS

    archivos = {ruta.name for ruta in DIRECTORIO_CORPUS.glob("*.md")}

    for caso in CASOS_EVALUACION:
        for identificador in caso["ids_relevantes"]:
            archivo, _, posicion = identificador.partition("#")
            assert archivo in archivos, f"{identificador} apunta a un archivo inexistente"
            assert posicion.isdigit(), f"{identificador} no tiene posición numérica"


def test_las_preguntas_no_se_repiten():
    preguntas = [caso["pregunta"] for caso in CASOS_EVALUACION]

    assert len(preguntas) == len(set(preguntas))


# --- Evaluación con retriever doble ---------------------------------------


class RetrieverFalso:
    """Devuelve una lista fija de documentos por pregunta."""

    def __init__(self, respuestas: dict[str, list[str]]) -> None:
        self.respuestas = respuestas
        self.ultimo_k = None

    def as_retriever(self, search_kwargs=None):
        self.ultimo_k = (search_kwargs or {}).get("k")
        return self

    def invoke(self, pregunta: str) -> list[Document]:
        return [
            Document(page_content=f"fragmento {identificador}", metadata={}, id=identificador)
            for identificador in self.respuestas.get(pregunta, [])
        ]


def test_evaluar_calcula_porcentajes_por_caso():
    casos = [
        {"pregunta": "uno", "ids_relevantes": {"a"}},
        {"pregunta": "dos", "ids_relevantes": {"b"}},
    ]
    retriever = RetrieverFalso({"uno": ["a", "x", "y"], "dos": ["x", "y", "z"]})

    resultados, precision, recall = evaluar_recuperacion(retriever, casos, k=3)

    assert resultados[0]["precision"] == pytest.approx(1 / 3)
    assert resultados[0]["recall"] == 1.0
    assert resultados[1]["precision"] == 0.0
    assert resultados[1]["recall"] == 0.0
    assert precision == pytest.approx(1 / 6)
    assert recall == 0.5


def test_evaluar_excluye_del_recall_los_casos_sin_respuesta():
    casos = [{"pregunta": "sin", "ids_relevantes": set()}]
    retriever = RetrieverFalso({"sin": ["x", "y"]})

    resultados, precision, recall = evaluar_recuperacion(retriever, casos, k=2)

    assert resultados[0]["recall"] is None
    assert precision == 0.0
    assert recall == 0.0


def test_evaluar_pasa_el_k_al_retriever():
    retriever = RetrieverFalso({})

    evaluar_recuperacion(retriever, [{"pregunta": "x", "ids_relevantes": set()}], k=4)

    assert retriever.ultimo_k == 4


def test_evaluar_recorta_los_ids_a_k():
    casos = [{"pregunta": "uno", "ids_relevantes": {"a"}}]
    retriever = RetrieverFalso({"uno": ["x", "y", "a"]})

    resultados, _, _ = evaluar_recuperacion(retriever, casos, k=2)

    # El fragmento relevante quedó fuera del top-2: la métrica lo refleja.
    assert resultados[0]["ids_recuperados"] == ["x", "y"]
    assert resultados[0]["recall"] == 0.0


# --- Archivo de resultados -------------------------------------------------


def test_formatear_marca_n_a_cuando_no_hay_relevantes():
    resultados = [{"pregunta": "sin", "precision": 0.0, "recall": None}]

    texto = formatear_resultados(resultados, 0.0, 0.0, "k=3, chunk_size=400")

    assert "0.00  n/a  sin" in texto
    assert "Fecha:" in texto
    assert "Configuracion: k=3, chunk_size=400" in texto
    assert "promedio -> precision: 0.000  recall: 0.000" in texto


def test_escribir_resultados_guarda_el_archivo(tmp_path):
    destino = tmp_path / "resultados.txt"

    escribir_resultados(destino, [], 0.5, 0.25, "k=3")

    assert destino.exists()
    assert "promedio -> precision: 0.500  recall: 0.250" in destino.read_text(encoding="utf-8")


def test_escribir_resultados_no_sobrescribe_una_corrida_anterior(tmp_path):
    destino = tmp_path / "resultados.txt"
    destino.write_text("corrida anterior", encoding="utf-8")

    with pytest.raises(FileExistsError, match="ya existe"):
        escribir_resultados(destino, [], 0.5, 0.25, "k=3")

    assert destino.read_text(encoding="utf-8") == "corrida anterior"
