"""Tests de la indexación del corpus (Bloque 24).

Qué se prueba y qué no
----------------------
Se prueba todo lo que se puede probar sin red: carga del corpus, partición en
fragmentos, generación de ids, verificación de integridad y persistencia real en
Chroma usando embeddings falsos. La persistencia se prueba de verdad —se escribe
en un directorio temporal y se vuelve a abrir—, porque es el punto donde esta
actividad puede fallar en silencio: un índice que parece construido y no
persistió se descubre recién en el bloque siguiente.

Lo que no se prueba aquí es el modelo real de embeddings: eso es la corrida del
script, que valida contra la API de Gemini y cuesta cuota.
"""

import hashlib

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from indexar_corpus import (
    DIRECTORIO_CORPUS,
    MINIMO_UTIL,
    ErrorCorpus,
    abrir_indice,
    cargar_documentos,
    construir_embeddings,
    construir_ids,
    indexar_corpus,
    metadatos_coleccion,
    partir_documentos,
    verificar_fragmentos,
    verificar_indice,
)

DIMENSION_FALSA = 8


class EmbeddingsFalsos(Embeddings):
    """Embeddings deterministas y locales: mismo texto, mismo vector, sin red.

    Reproduce la interfaz que Chroma usa de verdad (``embed_documents`` para el
    lote y ``embed_query`` para una consulta) para que el test ejercite el mismo
    camino que el modelo real.
    """

    def __init__(self, dimension: int = DIMENSION_FALSA) -> None:
        self.dimension = dimension
        self.llamadas_lote = 0

    def _vector(self, texto: str) -> list[float]:
        semilla = hashlib.sha256(texto.encode("utf-8")).digest()
        # Se repite el hash hasta llenar la dimensión pedida y se normaliza a
        # [-1, 1] para que las distancias sean comparables entre fragmentos.
        crudo = [byte / 255.0 * 2 - 1 for byte in semilla]
        return [crudo[i % len(crudo)] for i in range(self.dimension)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.llamadas_lote += 1
        return [self._vector(texto) for texto in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


@pytest.fixture
def documento_uno() -> Document:
    return Document(
        page_content=(
            "El cliente reintenta hasta tres veces ante un error temporal. "
            "Los errores de autenticación no se reintentan."
        ),
        metadata={"fuente": "01_cliente.md"},
    )


@pytest.fixture
def documento_dos() -> Document:
    return Document(
        page_content=(
            "El costo se calcula por componente: entrada con acierto de caché, "
            "entrada sin acierto y salida."
        ),
        metadata={"fuente": "02_costos.md"},
    )


# --- Carga del corpus ------------------------------------------------------


def test_cargar_documentos_lee_el_corpus_real():
    documentos = cargar_documentos(DIRECTORIO_CORPUS)

    assert len(documentos) >= 5
    assert all(documento.page_content.strip() for documento in documentos)
    assert all(documento.metadata["fuente"].endswith(".md") for documento in documentos)


def test_cargar_documentos_falla_si_el_directorio_no_existe(tmp_path):
    with pytest.raises(ErrorCorpus, match="No existe el directorio"):
        cargar_documentos(tmp_path / "no-existe")


def test_cargar_documentos_falla_si_no_hay_markdown(tmp_path):
    (tmp_path / "notas.txt").write_text("sin corpus", encoding="utf-8")

    with pytest.raises(ErrorCorpus, match="no tiene archivos .md"):
        cargar_documentos(tmp_path)


# --- Partición en fragmentos ----------------------------------------------


def test_partir_documentos_conserva_la_fuente(documento_uno):
    fragmentos = partir_documentos([documento_uno])

    assert fragmentos
    assert all(fragmento.metadata["fuente"] == "01_cliente.md" for fragmento in fragmentos)


def test_partir_documentos_no_genera_fragmentos_vacios(documento_uno, documento_dos):
    fragmentos = partir_documentos([documento_uno, documento_dos])

    assert all(fragmento.page_content.strip() for fragmento in fragmentos)
    assert verificar_fragmentos(fragmentos) == []


def test_partir_documentos_respeta_el_tamano_maximo():
    largo = "palabra " * 400
    documento = Document(page_content=largo, metadata={"fuente": "largo.md"})

    fragmentos = partir_documentos([documento], tamano=200, solapamiento=0, minimo_util=0)

    assert len(fragmentos) > 1
    # El splitter recorta espacios en los bordes, así que el límite se cumple
    # con holgura; lo que se verifica es que no devuelva el documento entero.
    assert all(len(fragmento.page_content) <= 200 for fragmento in fragmentos)


def test_fusiona_un_fragmento_corto_con_el_anterior():
    documentos = [
        Document(page_content="Primer párrafo completo y suficientemente largo " * 4, metadata={"fuente": "a.md"}),
        Document(page_content="Cola corta.", metadata={"fuente": "a.md"}),
    ]

    fragmentos = partir_documentos(documentos, tamano=400, solapamiento=0, minimo_util=120)

    assert len(fragmentos) == 1
    assert "Cola corta." in fragmentos[0].page_content


def test_no_fusiona_fragmentos_de_documentos_distintos():
    documentos = [
        Document(page_content="Documento uno con texto suficiente.", metadata={"fuente": "a.md"}),
        Document(page_content="Corto.", metadata={"fuente": "b.md"}),
    ]

    fragmentos = partir_documentos(documentos, tamano=400, solapamiento=0, minimo_util=120)

    # El fragmento corto abre su propio documento y no tiene a quién unirse
    # hacia adelante, así que se conserva: fusionarlo mezclaría dos fuentes y la
    # cita dejaría de ser inequívoca.
    assert len(fragmentos) == 2
    assert {f.metadata["fuente"] for f in fragmentos} == {"a.md", "b.md"}


def test_no_fusiona_si_supera_el_tope():
    # El tope es tamano * 1.5 = 600: 600 + 11 caracteres de la cola lo pasa.
    documentos = [
        Document(page_content="x" * 600, metadata={"fuente": "a.md"}),
        Document(page_content="Cola corta.", metadata={"fuente": "a.md"}),
    ]

    fragmentos = partir_documentos(documentos, tamano=400, solapamiento=0, minimo_util=120)

    assert len(fragmentos) == 2


def test_fusiona_el_titulo_inicial_hacia_adelante():
    documentos = [
        Document(
            page_content="# Título del documento\n\nPrimer párrafo con contenido real y suficiente largo.",
            metadata={"fuente": "a.md"},
        )
    ]

    fragmentos = partir_documentos(documentos, tamano=60, solapamiento=0, minimo_util=45)

    assert fragmentos[0].page_content.startswith("# Título del documento")
    assert "Primer párrafo" in fragmentos[0].page_content


def test_el_corpus_real_no_deja_fragmentos_huerfanos():
    documentos = cargar_documentos(DIRECTORIO_CORPUS)

    fragmentos = partir_documentos(documentos)

    assert min(len(f.page_content) for f in fragmentos) >= MINIMO_UTIL


def test_partir_documentos_no_pierde_texto():
    texto = "Primer párrafo sobre errores.\n\nSegundo párrafo sobre costos."
    documento = Document(page_content=texto, metadata={"fuente": "mezcla.md"})

    fragmentos = partir_documentos([documento], tamano=30, solapamiento=0)
    recuperado = " ".join(fragmento.page_content for fragmento in fragmentos)

    assert "errores" in recuperado
    assert "costos" in recuperado


# --- Ids -------------------------------------------------------------------


def test_construir_ids_son_unicos(documento_uno, documento_dos):
    fragmentos = partir_documentos([documento_uno, documento_dos])

    ids = construir_ids(fragmentos)

    assert len(ids) == len(fragmentos)
    assert len(set(ids)) == len(ids)


def test_construir_ids_son_estables_entre_corridas(documento_uno, documento_dos):
    primera = construir_ids(partir_documentos([documento_uno, documento_dos]))
    segunda = construir_ids(partir_documentos([documento_uno, documento_dos]))

    assert primera == segunda


def test_construir_ids_incluyen_el_archivo_de_origen(documento_uno):
    ids = construir_ids(partir_documentos([documento_uno]))

    assert all(identificador.startswith("01_cliente.md#") for identificador in ids)


# --- Verificación de integridad -------------------------------------------


def test_verificar_fragmentos_detecta_contenido_vacio():
    fragmentos = [Document(page_content="   \n ", metadata={"fuente": "x.md"})]

    problemas = verificar_fragmentos(fragmentos)

    assert any("vacío" in problema for problema in problemas)


def test_verificar_fragmentos_detecta_fuente_ausente():
    fragmentos = [Document(page_content="texto válido", metadata={})]

    problemas = verificar_fragmentos(fragmentos)

    assert any("fuente" in problema for problema in problemas)


def test_verificar_fragmentos_detecta_texto_repetido():
    repetido = Document(page_content="el mismo párrafo", metadata={"fuente": "a.md"})
    copia = Document(page_content="el mismo párrafo", metadata={"fuente": "b.md"})

    problemas = verificar_fragmentos([repetido, copia])

    assert any("mismo texto" in problema for problema in problemas)


# --- Persistencia real en Chroma ------------------------------------------


def test_indexar_persiste_y_se_puede_recargar(tmp_path, documento_uno, documento_dos):
    embeddings = EmbeddingsFalsos()
    fragmentos = partir_documentos([documento_uno, documento_dos])

    indexar_corpus(embeddings, fragmentos, directorio=tmp_path)
    recargado = abrir_indice(embeddings, directorio=tmp_path)

    assert recargado._collection.count() == len(fragmentos)


def test_indexar_calcula_los_embeddings_una_sola_vez(
    tmp_path, documento_uno, documento_dos
):
    embeddings = EmbeddingsFalsos()
    fragmentos = partir_documentos([documento_uno, documento_dos])

    indexar_corpus(embeddings, fragmentos, directorio=tmp_path)
    # Recargar el índice no puede volver a embeber: se reconecta al disco.
    total_antes = embeddings.llamadas_lote
    abrir_indice(embeddings, directorio=tmp_path)
    abrir_indice(embeddings, directorio=tmp_path)

    assert embeddings.llamadas_lote == total_antes


def test_indexar_guarda_los_metadatos_de_la_coleccion(
    tmp_path, documento_uno, documento_dos
):
    embeddings = EmbeddingsFalsos()
    fragmentos = partir_documentos([documento_uno, documento_dos])

    indexar_corpus(embeddings, fragmentos, directorio=tmp_path)
    recargado = abrir_indice(embeddings, directorio=tmp_path)

    resumen = verificar_indice(recargado)

    assert resumen["fragmentos"] == len(fragmentos)
    assert resumen["metadatos"]["modelo_embeddings"] == metadatos_coleccion()["modelo_embeddings"]


def test_indexar_rechaza_corpus_vacio(tmp_path):
    with pytest.raises(ErrorCorpus, match="No hay fragmentos"):
        indexar_corpus(EmbeddingsFalsos(), [], directorio=tmp_path)


def test_indexar_rechaza_fragmentos_con_problemas(tmp_path):
    fragmentos = [Document(page_content=" ", metadata={"fuente": "x.md"})]

    with pytest.raises(ErrorCorpus, match="problemas"):
        indexar_corpus(EmbeddingsFalsos(), fragmentos, directorio=tmp_path)


# --- Verificación del índice ----------------------------------------------


def test_verificar_indice_falla_si_esta_vacio(tmp_path):
    embeddings = EmbeddingsFalsos()
    vacio = abrir_indice(embeddings, directorio=tmp_path)

    with pytest.raises(ErrorCorpus, match="vacío"):
        verificar_indice(vacio)


def test_verificar_indice_rechaza_otro_modelo_de_embeddings(
    tmp_path, documento_uno
):
    embeddings = EmbeddingsFalsos()
    fragmentos = partir_documentos([documento_uno])
    metadatos = metadatos_coleccion(modelo_embeddings="otro-modelo")

    indexar_corpus(embeddings, fragmentos, directorio=tmp_path, metadatos=metadatos)
    recargado = abrir_indice(embeddings, directorio=tmp_path)

    with pytest.raises(ErrorCorpus, match="reindexar"):
        verificar_indice(recargado, modelo_esperado="models/gemini-embedding-001")


# --- Entorno ---------------------------------------------------------------


def test_construir_embeddings_falla_si_falta_la_clave(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(SystemExit, match="GEMINI_API_KEY"):
        construir_embeddings()
