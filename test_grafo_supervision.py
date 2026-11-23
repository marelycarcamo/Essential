"""Tests del punto de supervisión humana (Bloque 32).

Qué se prueba y qué no
----------------------
Se prueba el contrato completo: que el grafo **pausa** de verdad antes de aplicar,
que no aplica sin una decisión, que el camino de rechazo existe y funciona, que no
se puede aprobar dos veces el mismo hilo y que con un checkpointer durable la
aprobación pendiente sobrevive a que se reconstruya el grafo (que es lo que pasa
cuando el proceso se reinicia y la persona aprueba horas después).

Es el criterio no compensable del Gate 2: "lo tengo documentado" no cuenta, tiene
que pausar y retomar.

Lo que no se prueba aquí es la interfaz humana que consume el interrupt: el
mecanismo entrega el diccionario y espera un booleano, y eso es lo que se verifica.
"""

import pytest
from langgraph.types import Command

from grafo_supervision import (
    aplicar,
    construir_grafo,
    esta_pendiente,
    nuevo_thread_id,
    resolver,
    solicitar_aprobacion,
)


# --- Pausa y reanudación ---------------------------------------------------


def test_solicitar_aprobacion_pausa_el_grafo():
    app = construir_grafo()

    _, estado = solicitar_aprobacion(app, "abc123def456")

    assert "__interrupt__" in estado
    assert estado["__interrupt__"][0].value["huella_propuesta"] == "abc123def456"


def test_el_interrupt_lleva_el_motivo_y_la_huella():
    app = construir_grafo()

    config, _ = solicitar_aprobacion(app, "abc123def456")
    pendiente = app.get_state(config).interrupts[0].value

    assert "motivo" in pendiente
    assert pendiente["huella_propuesta"] == "abc123def456"


def test_esta_pendiente_es_verdadero_antes_y_falso_despues():
    app = construir_grafo()
    config, _ = solicitar_aprobacion(app, "abc123def456")

    assert esta_pendiente(app, config) is True
    resolver(app, config, True)
    assert esta_pendiente(app, config) is False


def test_aprobar_aplica_la_version():
    app = construir_grafo()
    config, _ = solicitar_aprobacion(app, "abc123def456")

    final = resolver(app, config, True)

    assert final["aprobado"] is True
    assert final["aplicado"] is True


def test_rechazar_no_aplica_la_version():
    app = construir_grafo()
    config, _ = solicitar_aprobacion(app, "abc123def456")

    final = resolver(app, config, False)

    assert final["aprobado"] is False
    assert final["aplicado"] is False


def test_el_rechazo_necesita_un_hilo_nuevo():
    app = construir_grafo()
    config, _ = solicitar_aprobacion(app, "uno")
    resolver(app, config, True)

    with pytest.raises(ValueError, match="no tiene ninguna aprobacion pendiente"):
        resolver(app, config, False)


def test_no_se_puede_aprobar_dos_veces_el_mismo_hilo():
    app = construir_grafo()
    config, _ = solicitar_aprobacion(app, "abc123def456")
    resolver(app, config, True)

    with pytest.raises(ValueError, match="no tiene ninguna aprobacion pendiente"):
        resolver(app, config, True)


def test_resolver_exige_un_booleano_explicito():
    app = construir_grafo()
    config, _ = solicitar_aprobacion(app, "abc123def456")

    with pytest.raises(ValueError, match="True o False"):
        resolver(app, config, "no")


def test_el_nodo_supervisar_rechaza_una_decision_que_no_es_booleana():
    """``Command(resume="no")`` activaría la versión, porque un texto no vacío es verdadero.

    El guardia vive dentro del nodo, así que se prueba por el camino real: se
    reanuda el grafo con un texto en vez de un booleano.
    """

    app = construir_grafo()
    config, _ = solicitar_aprobacion(app, "abc123def456")

    with pytest.raises(ValueError, match="booleano"):
        app.invoke(Command(resume="no"), config=config)


def test_aplicar_refleja_la_decision_en_el_estado():
    assert aplicar({"huella_nueva": "abc", "aprobado": True})["aplicado"] is True
    assert aplicar({"huella_nueva": "abc", "aprobado": False})["aplicado"] is False


# --- Hilos -----------------------------------------------------------------


def test_cada_hilo_es_independiente():
    app = construir_grafo()
    config_a, _ = solicitar_aprobacion(app, "aaa")
    config_b, _ = solicitar_aprobacion(app, "bbb")

    resolver(app, config_a, True)

    assert esta_pendiente(app, config_a) is False
    assert esta_pendiente(app, config_b) is True


def test_nuevo_thread_id_es_unico():
    assert nuevo_thread_id() != nuevo_thread_id()


def test_reanudar_un_hilo_inexistente_falla():
    app = construir_grafo()
    config = {"configurable": {"thread_id": "no-existe"}}

    assert esta_pendiente(app, config) is False
    with pytest.raises(ValueError, match="no tiene ninguna aprobacion pendiente"):
        resolver(app, config, True)


# --- Checkpointer durable --------------------------------------------------


def test_con_checkpointer_durable_la_aprobacion_sobrevive_al_reinicio(tmp_path):
    """El grafo se reconstruye desde el mismo archivo, como en un proceso nuevo."""

    ruta = tmp_path / "checkpoints.db"
    app_antes = construir_grafo(durable=True, ruta_checkpoints=ruta)
    config, _ = solicitar_aprobacion(app_antes, "abc123def456", thread_id="durable-1")
    assert esta_pendiente(app_antes, config) is True

    app_despues = construir_grafo(durable=True, ruta_checkpoints=ruta)

    assert esta_pendiente(app_despues, config) is True
    final = resolver(app_despues, config, True)
    assert final["aplicado"] is True


def test_con_checkpointer_en_memoria_el_estado_no_sobrevive_al_reinicio():
    app_antes = construir_grafo()
    config, _ = solicitar_aprobacion(app_antes, "abc123def456", thread_id="memoria-1")

    app_despues = construir_grafo()

    assert esta_pendiente(app_despues, config) is False
