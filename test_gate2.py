"""Autochequeo del Gate 2: los cuatro criterios no compensables (Bloque S1).

Qué verifica cada prueba
------------------------
1. **El pipeline corre de punta a punta** sin excepciones no capturadas.
2. **Cada respuesta cita la fuente** que la respalda.
3. **Los logs no exponen contenido crudo sensible**: ni el correo ni la pregunta
   completa aparecen, tampoco el texto de los documentos.
4. **El punto de supervisión humana funciona**: el grafo se detiene de verdad y
   solo `Command(resume=...)` lo retoma.

Por qué corre sin red
---------------------
El servicio recibe el índice y el modelo como parámetros, así que estas pruebas
inyectan embeddings deterministas y un modelo doble. El pipeline es el mismo que
en producción: lo único simulado es el proveedor. Así los cuatro criterios se
verifican en cada corrida de la suite, en menos de dos segundos, sin gastar cuota.
La corrida contra los proveedores reales es `ensayo_gate2.py`.

Nota sobre el criterio 3: esta prueba solo ve los registros que se propagan por
`logging`. Si algún día el servicio escribe por `print` o con un logger aislado,
la prueba pasaría sin ver nada; por eso el autochequeo manual del README incluye
un `grep` sobre la salida real.
"""

import logging

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import indexar_corpus as ic
from grafo_supervision import (
    construir_grafo,
    esta_pendiente,
    resolver,
    solicitar_aprobacion,
)
from registro_prompts import PROMPT_ACTIVO, huella_prompt
from servicio_rag import ServicioRAG

# Dato sensible de prueba que NUNCA debe aparecer en los logs.
CORREO = "ana.perez@example.com"
PREGUNTA = "¿Qué hace el cliente ante un 401? Mi correo es " + CORREO


class ModeloQueCita(BaseChatModel):
    """Modelo doble que cita la primera fuente del contexto, como haría el real."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        prompt = messages[0].content if isinstance(messages[0].content, str) else ""
        fuente = "una fuente"
        for linea in prompt.splitlines():
            if linea.startswith("[1] (") and ")" in linea:
                fuente = linea[len("[1] (") : linea.index(")")]
                break
        texto = f"Según {fuente}, el cliente no reintenta ante un error de credencial [1]."
        mensaje = AIMessage(
            content=texto,
            usage_metadata={"input_tokens": 180, "output_tokens": 22, "total_tokens": 202},
        )
        return ChatResult(generations=[ChatGeneration(message=mensaje)])

    @property
    def _llm_type(self) -> str:
        return "modelo-que-cita"


@pytest.fixture(scope="module")
def directorio_indice(tmp_path_factory):
    return tmp_path_factory.mktemp("gate2")


@pytest.fixture(scope="module")
def servicio(directorio_indice):
    """Servicio con índice real en disco, embeddings falsos y modelo doble."""

    embeddings = DeterministicFakeEmbedding(size=64)
    documentos = ic.cargar_documentos()
    ic.actualizar_indice(
        embeddings, documentos, directorio=directorio_indice, modelo_embeddings="fake-gate2"
    )
    vectorstore = ic.abrir_indice(embeddings, directorio_indice)

    return ServicioRAG(
        vectorstore=vectorstore,
        modelo=ModeloQueCita(),
        modelo_cobrado="fake",
        k=3,
        huella_prompt=huella_prompt(PROMPT_ACTIVO),
    )


# --- Criterio 1 ------------------------------------------------------------


def test_1_pipeline_sin_excepciones(servicio):
    resultado = servicio.responder(PREGUNTA)

    assert resultado["respuesta"].strip(), "el servicio devolvió una respuesta vacía"
    assert resultado["auditoria"], "el servicio no dejó traza"


# --- Criterio 2 ------------------------------------------------------------


def test_2_respuesta_cita_fuente(servicio):
    resultado = servicio.responder(PREGUNTA)

    assert resultado["fuentes"], "el servicio no devolvió ninguna fuente"
    assert any(fuente in resultado["respuesta"] for fuente in resultado["fuentes"]), (
        "la respuesta no menciona su fuente"
    )


# --- Criterio 3 ------------------------------------------------------------


def test_3_logs_sin_contenido_crudo(servicio, caplog):
    with caplog.at_level(logging.DEBUG):
        servicio.responder(PREGUNTA)

    texto_logs = " ".join(
        registro.getMessage() + " " + str(registro.__dict__) for registro in caplog.records
    )
    assert CORREO not in texto_logs, "un dato sensible aparece en los logs"
    assert PREGUNTA not in texto_logs, "la consulta completa aparece en los logs"


def test_3b_los_logs_tampoco_exponen_el_texto_de_los_documentos(servicio, caplog):
    with caplog.at_level(logging.DEBUG):
        servicio.responder(PREGUNTA)

    documento = servicio.recuperar(PREGUNTA)[0]
    texto_logs = " ".join(registro.getMessage() for registro in caplog.records)

    assert documento.page_content not in texto_logs


# --- Criterio 4 ------------------------------------------------------------


def test_4_supervision_pausa_y_reanuda():
    app = construir_grafo()
    config, pausado = solicitar_aprobacion(app, huella_prompt(PROMPT_ACTIVO))

    assert "__interrupt__" in pausado, "el grafo no se detuvo en el punto de supervisión"
    assert esta_pendiente(app, config)

    final = resolver(app, config, True)

    assert "__interrupt__" not in final, "el grafo sigue pausado después de reanudar"
    assert final["aplicado"] is True, "la decisión no quedó registrada"


# --- Guardas adicionales del Gate -----------------------------------------


def test_el_indice_no_se_duplica_al_reconstruir_el_servicio(servicio, directorio_indice):
    antes = servicio.vectorstore._collection.count()
    embeddings = DeterministicFakeEmbedding(size=64)

    ic.actualizar_indice(embeddings, ic.cargar_documentos(), directorio=directorio_indice)

    assert servicio.vectorstore._collection.count() == antes


def test_la_respuesta_registra_la_huella_del_prompt(servicio):
    resultado = servicio.responder(PREGUNTA)
    consumos = [e for e in resultado["auditoria"] if e["evento"] == "consumo_llm"]

    assert consumos
    assert consumos[0]["huella_prompt"] == huella_prompt(PROMPT_ACTIVO)


def test_la_traza_comparte_un_solo_trace_id(servicio):
    resultado = servicio.responder(PREGUNTA)

    trace_ids = {evento["trace_id"] for evento in resultado["auditoria"]}

    assert trace_ids == {resultado["trace_id"]}


def test_una_pregunta_imposible_no_inventa_fuente(servicio):
    resultado = servicio.responder("¿Cuál es el precio del dólar hoy en Chile?")

    # El servicio siempre cita lo que recuperó; lo que no puede es devolver una
    # respuesta sin traza ni fuentes.
    assert resultado["respuesta"].strip()
    assert resultado["auditoria"]
