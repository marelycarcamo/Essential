"""Tests de la actualización incremental del índice (Bloque 25, requisito del Gate 2).

Qué se prueba y qué no
----------------------
Se prueba que la sincronización agregue solo lo nuevo, reemplace lo modificado,
elimine lo que ya no existe y **no duplique** nada al correr dos veces seguidas.
La duplicación es el error clásico de este paso: ``from_documents`` sobre un
índice con contenido agrega otra copia de cada fragmento, la búsqueda devuelve
dos versiones del mismo párrafo y nada falla.

Los embeddings son falsos y el índice vive en un directorio temporal, así que no
hay red ni cuota.
"""

import pytest
from langchain_core.documents import Document

import indexar_corpus as ic
from test_indexar_corpus import EmbeddingsFalsos


def documento(texto: str, fuente: str = "a.md") -> Document:
    import hashlib

    return Document(
        page_content=texto,
        metadata={
            "fuente": fuente,
            "huella_fuente": hashlib.sha256(texto.encode("utf-8")).hexdigest()[:12],
        },
    )


def contar(directorio, embeddings) -> int:
    return ic.abrir_indice(embeddings, directorio)._collection.count()


# --- Índice vacío ----------------------------------------------------------


def test_actualizar_sobre_indice_vacio_lo_construye(tmp_path):
    embeddings = EmbeddingsFalsos()

    resumen = ic.actualizar_indice(embeddings, [documento("texto uno")], directorio=tmp_path)

    assert resumen["indice_vacio"] is True
    assert resumen["agregadas"] == ["a.md"]
    assert contar(tmp_path, embeddings) > 0


def test_el_indice_construido_en_vacio_lleva_metadata_de_coleccion(tmp_path):
    embeddings = EmbeddingsFalsos()

    ic.actualizar_indice(embeddings, [documento("texto uno")], directorio=tmp_path)
    recargado = ic.abrir_indice(embeddings, tmp_path)

    assert recargado._collection.metadata["modelo_embeddings"] == ic.MODELO_EMBEDDINGS


# --- Sin cambios -----------------------------------------------------------


def test_sin_cambios_no_embe_be_nada(tmp_path):
    embeddings = EmbeddingsFalsos()
    documentos = [documento("texto uno"), documento("texto dos", "b.md")]
    ic.actualizar_indice(embeddings, documentos, directorio=tmp_path)
    antes = contar(tmp_path, embeddings)
    llamadas_antes = embeddings.llamadas_lote

    resumen = ic.actualizar_indice(embeddings, documentos, directorio=tmp_path)

    assert resumen["fragmentos"] == 0
    assert resumen["omitidas"] == ["a.md", "b.md"]
    assert embeddings.llamadas_lote == llamadas_antes
    assert contar(tmp_path, embeddings) == antes


# --- Cambios ---------------------------------------------------------------


def test_un_documento_nuevo_se_agrega(tmp_path):
    embeddings = EmbeddingsFalsos()
    ic.actualizar_indice(embeddings, [documento("texto uno")], directorio=tmp_path)
    antes = contar(tmp_path, embeddings)

    resumen = ic.actualizar_indice(
        embeddings,
        [documento("texto uno"), documento("texto dos", "b.md")],
        directorio=tmp_path,
    )

    assert resumen["agregadas"] == ["b.md"]
    assert contar(tmp_path, embeddings) > antes


def test_un_documento_modificado_reemplaza_al_anterior(tmp_path):
    embeddings = EmbeddingsFalsos()
    ic.actualizar_indice(embeddings, [documento("texto viejo")], directorio=tmp_path)
    antes = contar(tmp_path, embeddings)

    resumen = ic.actualizar_indice(
        embeddings, [documento("texto nuevo")], directorio=tmp_path
    )

    assert resumen["actualizadas"] == ["a.md"]
    assert contar(tmp_path, embeddings) == antes
    textos = ic.abrir_indice(embeddings, tmp_path).get()["documents"]
    assert all("texto viejo" not in texto for texto in textos)


def test_un_documento_eliminado_sale_del_indice(tmp_path):
    embeddings = EmbeddingsFalsos()
    ic.actualizar_indice(
        embeddings, [documento("texto uno"), documento("texto dos", "b.md")], directorio=tmp_path
    )

    resumen = ic.actualizar_indice(embeddings, [documento("texto uno")], directorio=tmp_path)

    assert resumen["eliminadas"] == ["b.md"]
    datos = ic.abrir_indice(embeddings, tmp_path).get()
    assert all(meta["fuente"] != "b.md" for meta in datos["metadatas"])


def test_dos_corridas_seguidas_no_duplican_fragmentos(tmp_path):
    """El error clásico: reindexar sobre un índice con contenido y terminar con el doble."""

    embeddings = EmbeddingsFalsos()
    documentos = [documento("texto uno"), documento("texto dos", "b.md")]

    ic.actualizar_indice(embeddings, documentos, directorio=tmp_path)
    primera = contar(tmp_path, embeddings)
    ic.actualizar_indice(embeddings, documentos, directorio=tmp_path)
    segunda = contar(tmp_path, embeddings)

    assert primera == segunda


def test_un_indice_viejo_sin_huella_se_reindexa_una_vez(tmp_path):
    embeddings = EmbeddingsFalsos()
    # Índice construido "antes" de que existiera huella_fuente.
    viejo = Document(page_content="texto uno", metadata={"fuente": "a.md"})
    ic.indexar_corpus(embeddings, ic.partir_documentos([viejo]), directorio=tmp_path)

    resumen = ic.actualizar_indice(embeddings, [documento("texto uno")], directorio=tmp_path)

    assert resumen["actualizadas"] == ["a.md"]
    assert contar(tmp_path, embeddings) == 1


# --- Huellas ---------------------------------------------------------------


def test_cargar_documentos_calcula_la_huella_de_cada_fuente():
    documentos = ic.cargar_documentos(ic.DIRECTORIO_CORPUS)

    assert all(documento.metadata["huella_fuente"] for documento in documentos)


def test_la_huella_cambia_si_cambia_el_contenido(tmp_path):
    (tmp_path / "a.md").write_text("contenido uno", encoding="utf-8")
    primera = ic.cargar_documentos(tmp_path)[0].metadata["huella_fuente"]
    (tmp_path / "a.md").write_text("contenido dos", encoding="utf-8")
    segunda = ic.cargar_documentos(tmp_path)[0].metadata["huella_fuente"]

    assert primera != segunda


def test_huellas_indexadas_lee_el_indice(tmp_path):
    embeddings = EmbeddingsFalsos()
    ic.actualizar_indice(embeddings, [documento("texto uno")], directorio=tmp_path)

    huellas = ic.huellas_indexadas(ic.abrir_indice(embeddings, tmp_path))

    assert "a.md" in huellas
    assert huellas["a.md"]
