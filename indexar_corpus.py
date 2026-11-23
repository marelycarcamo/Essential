"""Indexación del corpus documental en un vector store persistente (Bloque 24).

Qué hace
--------
Lee los documentos de ``corpus/``, los parte en fragmentos (chunks) respetando su
estructura, calcula el embedding real de cada fragmento con Gemini y guarda el
resultado en disco con Chroma. El índice queda listo para que el flujo de
recuperación lo consulte sin volver a calcular nada.

Por qué metadata e ids explícitos
---------------------------------
- ``fuente``: sin ella, la respuesta no puede citar el documento del que salió.
  La metadata no se puede agregar después sin reindexar, así que se hace ahora.
- ``id`` predecible (``archivo.md#3``): permite comparar corridas entre sí y
  etiquetar a mano el conjunto de evaluación, sin depender del orden interno que
  Chroma le asigne a los documentos.
- ``modelo_embeddings`` en la metadata de la colección: un índice vectorial no
  dice con qué modelo se construyó. Cambiar el modelo sin reindexar no falla
  —si la dimensión coincide, devuelve fragmentos sin relación y el sistema
  parece funcionar—, así que el nombre queda guardado y se verifica al cargar.

Por qué 400 caracteres y 60 de solapamiento
-------------------------------------------
Se midió sobre el corpus real: el splitter corta en párrafos, así que el tamaño
efectivo lo determinan los saltos de línea más que el límite. Con 400/60 el
corpus queda en 34 fragmentos, casi todos de un párrafo completo, y el
solapamiento solo actúa cuando un párrafo se parte. Un fragmento más chico
duplica el número de chunks sin ganar precisión; uno más grande mezcla temas
distintos y ensucia la cita.

Uso
---
    ./.venv/Scripts/python.exe indexar_corpus.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

DIRECTORIO_CORPUS = Path("corpus")
DIRECTORIO_VECTORSTORE = Path("vectorstore")
NOMBRE_COLECCION = "documentacion_essential"

# Verificado contra la API el 2026-11-23: ``models/gemini-embedding-001`` existe
# y devuelve vectores de 3072 dimensiones. DeepSeek no ofrece embeddings, así que
# Gemini es el único proveedor de embeddings del proyecto.
MODELO_EMBEDDINGS = "models/gemini-embedding-001"
VARIABLE_CLAVE = "GEMINI_API_KEY"

# Parámetros de partición, elegidos con la medición descrita en el docstring.
TAMANO_CHUNK = 400
SOLAPAMIENTO = 60

# Un fragmento por debajo de este largo no se sostiene solo: suele ser la cola de
# un párrafo largo ("incorrecto es peor que ningún material."). Ocupa un lugar
# entre los candidatos de la recuperación y, al no tener contexto propio, nunca
# es el fragmento que responde; en el mejor caso desplaza a uno que sí. Por eso
# se fusiona con el fragmento anterior del mismo documento, sin pasar del tope
# de fusión: un fragmento de 900 caracteres volvería a mezclar dos temas.
MINIMO_UTIL = 120
FACTOR_FUSION = 1.5


class ErrorCorpus(RuntimeError):
    """El corpus no existe, está vacío o no se puede leer."""


def cargar_documentos(directorio: Path = DIRECTORIO_CORPUS) -> list[Document]:
    """Carga los ``.md`` del directorio como documentos con su fuente.

    Args:
        directorio: carpeta con los documentos del corpus.

    Returns:
        Un ``Document`` por archivo, en orden alfabético, con
        ``metadata["fuente"]`` igual al nombre del archivo.

    Raises:
        ErrorCorpus: si el directorio no existe o no contiene archivos ``.md``.
    """

    if not directorio.is_dir():
        raise ErrorCorpus(f"No existe el directorio del corpus: {directorio}")

    archivos = sorted(directorio.glob("*.md"))
    if not archivos:
        raise ErrorCorpus(f"El directorio del corpus no tiene archivos .md: {directorio}")

    documentos = [
        Document(
            page_content=archivo.read_text(encoding="utf-8"),
            metadata={"fuente": archivo.name},
        )
        for archivo in archivos
    ]
    logger.info("Cargados %d documentos desde %s", len(documentos), directorio)
    return documentos


def partir_documentos(
    documentos: list[Document],
    tamano: int = TAMANO_CHUNK,
    solapamiento: int = SOLAPAMIENTO,
    minimo_util: int = MINIMO_UTIL,
) -> list[Document]:
    """Parte los documentos en fragmentos, conservando ``fuente`` en cada uno.

    Args:
        documentos: documentos completos.
        tamano: tamaño máximo del fragmento, en caracteres.
        solapamiento: caracteres que se repiten entre fragmentos contiguos.
        minimo_util: largo bajo el cual un fragmento se fusiona con el anterior.
            Con ``0`` no se fusiona nada (útil para probar la partición pura).

    Returns:
        Fragmentos sin contenido vacío, en el orden de los documentos de entrada.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=tamano, chunk_overlap=solapamiento
    )
    fragmentos = splitter.split_documents(documentos)
    # Un salto de línea de más al final de un documento produce un fragmento
    # vacío que se indexaría igual y ocuparía un lugar entre los candidatos.
    fragmentos = [f for f in fragmentos if f.page_content.strip()]

    return _fusionar_fragmentos_cortos(fragmentos, tamano, minimo_util)


def _fusionar_fragmentos_cortos(
    fragmentos: list[Document],
    tamano: int,
    minimo_util: int,
    factor_fusion: float = FACTOR_FUSION,
) -> list[Document]:
    """Fusiona cada fragmento demasiado corto con el anterior del mismo documento.

    La fusión respeta dos límites: no cruza el borde de un documento (el
    fragmento fusionado tendría dos fuentes y la cita sería ambigua) y no
    supera el tope de fusión, para no terminar con un fragmento que mezcle dos
    temas y arruine la precisión de la recuperación.
    """

    if minimo_util <= 0:
        return fragmentos

    tope = int(tamano * factor_fusion)
    resultado: list[Document] = []

    for fragmento in fragmentos:
        anterior = resultado[-1] if resultado else None
        mismo_documento = (
            anterior is not None
            and anterior.metadata.get("fuente") == fragmento.metadata.get("fuente")
        )
        largo = len(fragmento.page_content)
        combinado = (
            len(anterior.page_content) + len(fragmento.page_content) + 1
            if mismo_documento
            else 0
        )

        if mismo_documento and largo < minimo_util and combinado <= tope:
            resultado[-1] = Document(
                page_content=f"{anterior.page_content}\n\n{fragmento.page_content}",
                metadata=dict(anterior.metadata),
            )
        else:
            resultado.append(fragmento)

    return _fusionar_inicios_huerfanos(resultado, minimo_util, tope)


def _fusionar_inicios_huerfanos(
    fragmentos: list[Document],
    minimo_util: int,
    tope: int,
) -> list[Document]:
    """Fusiona un fragmento corto que abre su documento con el que viene después.

    Es el caso del título suelto: el documento empieza con un encabezado que el
    splitter separa del primer párrafo. No tiene fragmento anterior del mismo
    archivo, así que la fusión va hacia adelante. Se conserva el texto del
    título al principio, porque es lo que hace que una pregunta sobre el tema del
    documento lo encuentre.
    """

    resultado: list[Document] = []
    pendiente: Document | None = None

    for fragmento in fragmentos:
        if pendiente is not None:
            misma_fuente = pendiente.metadata.get("fuente") == fragmento.metadata.get(
                "fuente"
            )
            combinado = len(pendiente.page_content) + len(fragmento.page_content) + 1
            if misma_fuente and combinado <= tope:
                resultado.append(
                    Document(
                        page_content=f"{pendiente.page_content}\n\n{fragmento.page_content}",
                        metadata=dict(fragmento.metadata),
                    )
                )
                pendiente = None
                continue
            resultado.append(pendiente)
            pendiente = None

        if len(fragmento.page_content) < minimo_util:
            pendiente = fragmento
        else:
            resultado.append(fragmento)

    if pendiente is not None:
        resultado.append(pendiente)

    return resultado


def construir_ids(fragmentos: list[Document]) -> list[str]:
    """Genera un id predecible y único por fragmento.

    El id combina el nombre del archivo y la posición del fragmento dentro de él
    (``02_costos.md#0``). Es estable entre corridas y no depende del orden que
    Chroma asigne internamente.
    """

    contadores: dict[str, int] = {}
    ids: list[str] = []
    for fragmento in fragmentos:
        fuente = fragmento.metadata["fuente"]
        posicion = contadores.get(fuente, 0)
        contadores[fuente] = posicion + 1
        ids.append(f"{fuente}#{posicion}")
    return ids


def verificar_fragmentos(fragmentos: list[Document]) -> list[str]:
    """Revisa la integridad de los fragmentos antes de gastar llamadas a la API.

    Comprueba que ninguno quede vacío, que todos declaren su fuente y que no
    haya dos fragmentos con el mismo texto. Las tres cosas se detectan
    localmente y no cuestan nada; descubrirlas después de indexar obliga a
    reindexar y a pagar otra vez los embeddings.

    Un fragmento duplicado no es un error visible: ocupa dos lugares entre los
    candidatos de la búsqueda y desplaza a contenido que sí aporta.

    Returns:
        La lista de problemas encontrados. Vacía si todo está en orden.
    """

    problemas: list[str] = []
    vistos: dict[str, int] = {}

    for posicion, fragmento in enumerate(fragmentos):
        contenido = fragmento.page_content.strip()
        if not contenido:
            problemas.append(f"fragmento {posicion}: contenido vacío")
            continue
        if "fuente" not in fragmento.metadata:
            problemas.append(f"fragmento {posicion}: sin metadata 'fuente'")
        if contenido in vistos:
            problemas.append(
                f"fragmento {posicion}: mismo texto que el fragmento {vistos[contenido]}"
            )
        else:
            vistos[contenido] = posicion

    return problemas


def metadatos_coleccion(
    modelo_embeddings: str = MODELO_EMBEDDINGS,
    tamano: int = TAMANO_CHUNK,
    solapamiento: int = SOLAPAMIENTO,
) -> dict:
    """Arma la metadata de la colección: con qué modelo y con qué parámetros se construyó."""

    return {
        "modelo_embeddings": modelo_embeddings,
        "tamano_chunk": tamano,
        "solapamiento": solapamiento,
    }


def indexar_corpus(
    embeddings: Embeddings,
    fragmentos: list[Document],
    directorio: Path = DIRECTORIO_VECTORSTORE,
    metadatos: dict | None = None,
) -> Chroma:
    """Indexa los fragmentos en Chroma y los persiste en disco.

    Args:
        embeddings: función de embeddings a usar.
        fragmentos: fragmentos ya partidos y verificados.
        directorio: carpeta donde se persiste el índice.
        metadatos: metadata de la colección. Si es ``None`` se usa la del modelo
            configurado.

    Returns:
        El vector store ya construido.

    Raises:
        ErrorCorpus: si ``fragmentos`` está vacío.
    """

    if not fragmentos:
        raise ErrorCorpus("No hay fragmentos para indexar")

    problemas = verificar_fragmentos(fragmentos)
    if problemas:
        raise ErrorCorpus("Fragmentos con problemas: " + "; ".join(problemas))

    ids = construir_ids(fragmentos)
    metadatos_finales = metadatos or metadatos_coleccion()
    logger.info(
        "Indexando %d fragmentos con %s en %s",
        len(fragmentos),
        metadatos_finales["modelo_embeddings"],
        directorio,
    )

    return Chroma.from_documents(
        documents=fragmentos,
        embedding=embeddings,
        ids=ids,
        persist_directory=str(directorio),
        collection_name=NOMBRE_COLECCION,
        collection_metadata=metadatos_finales,
    )


def abrir_indice(
    embeddings: Embeddings,
    directorio: Path = DIRECTORIO_VECTORSTORE,
) -> Chroma:
    """Reconecta a un índice ya persistido, sin volver a calcular embeddings."""

    return Chroma(
        persist_directory=str(directorio),
        embedding_function=embeddings,
        collection_name=NOMBRE_COLECCION,
    )


def verificar_indice(
    vectorstore: Chroma,
    modelo_esperado: str = MODELO_EMBEDDINGS,
) -> dict:
    """Comprueba que el índice exista y que se haya construido con el modelo esperado.

    Returns:
        Un resumen con el número de fragmentos y la metadata de la colección.

    Raises:
        ErrorCorpus: si el índice está vacío o si lo construyó otro modelo de
            embeddings, que es el caso en que la búsqueda devuelve basura sin
            lanzar ningún error.
    """

    total = vectorstore._collection.count()
    if total == 0:
        raise ErrorCorpus("El índice está vacío: revisa el directorio de persistencia")

    metadatos = dict(vectorstore._collection.metadata or {})
    modelo_guardado = metadatos.get("modelo_embeddings")
    if modelo_guardado and modelo_guardado != modelo_esperado:
        raise ErrorCorpus(
            "El índice se construyó con "
            f"{modelo_guardado} y se está consultando con {modelo_esperado}. "
            "Hay que reindexar: los vectores de modelos distintos no son comparables."
        )

    return {"fragmentos": total, "metadatos": metadatos}


def construir_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Crea el cliente de embeddings, avisando si falta la clave."""

    if not os.environ.get(VARIABLE_CLAVE):
        raise SystemExit(
            f"Falta {VARIABLE_CLAVE}. Defínela en el archivo .env de la raíz del "
            "proyecto y vuelve a ejecutar."
        )
    return GoogleGenerativeAIEmbeddings(model=MODELO_EMBEDDINGS)


def main() -> int:
    """Indexa el corpus desde cero y comprueba que quedó persistido en disco."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_dotenv()

    try:
        documentos = cargar_documentos()
        fragmentos = partir_documentos(documentos)
        embeddings = construir_embeddings()
        indexar_corpus(embeddings, fragmentos)

        # La prueba de que persistió no es que el script corra: es que una
        # conexión nueva al mismo directorio devuelva el mismo contenido.
        recargado = abrir_indice(embeddings)
        resumen = verificar_indice(recargado)
    except ErrorCorpus as error:
        logger.error("No se pudo indexar el corpus: %s", error)
        return 1

    print(f"Documentos: {len(documentos)}")
    print(f"Fragmentos: {len(fragmentos)}")
    print(f"Persistido en {DIRECTORIO_VECTORSTORE}/: {resumen['fragmentos']} fragmentos")
    print(f"Metadatos de la colección: {resumen['metadatos']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
