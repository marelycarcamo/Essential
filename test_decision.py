"""Tests del método de decisión de personalización (Bloque 30).

Qué se prueba y qué no
----------------------
Se prueba el puntaje de las tres estrategias en cada combinación relevante, el
manejo del empate, el orden de las cuatro secciones del documento y la negativa a
sobrescribir un documento ya redactado. También se fija el resultado del caso
real: si alguien cambia una respuesta o un puntaje, el test avisa.

Lo que no se prueba aquí es si la decisión es correcta para el negocio: eso lo
sostiene la evidencia anotada junto a cada respuesta, no un test.
"""

import pytest

from decision import (
    PROBLEMA,
    RESPUESTAS_MI_CASO,
    escribir_documento,
    generar_esqueleto_decision,
    recomendar_estrategia,
)


def respuestas(**cambios) -> dict[str, bool]:
    base = {
        "informacion_cambia_seguido": False,
        "problema_de_formato_o_tono": False,
        "dataset_grande_y_de_calidad": False,
        "presupuesto_limitado_o_iterar_rapido": False,
        "necesita_multiples_variantes": False,
    }
    base.update(cambios)
    return base


# --- Puntaje ---------------------------------------------------------------


def test_sin_condiciones_todas_las_tecnicas_empatan_en_cero():
    puntos, ganadoras = recomendar_estrategia(respuestas())

    assert puntos == {"RAG": 0, "Fine-tuning completo": 0, "PEFT": 0}
    assert len(ganadoras) == 3


def test_la_informacion_que_cambia_favorece_rag():
    puntos, ganadoras = recomendar_estrategia(respuestas(informacion_cambia_seguido=True))

    assert puntos["RAG"] == 3
    assert ganadoras == ["RAG"]


def test_el_problema_de_formato_favorece_peft():
    puntos, _ = recomendar_estrategia(respuestas(problema_de_formato_o_tono=True))

    assert puntos["PEFT"] == 2
    assert puntos["Fine-tuning completo"] == 1


def test_el_dataset_grande_favorece_fine_tuning():
    puntos, _ = recomendar_estrategia(respuestas(dataset_grande_y_de_calidad=True))

    assert puntos["Fine-tuning completo"] == 2
    assert puntos["PEFT"] == 1


def test_el_presupuesto_limitado_favorece_rag_y_peft():
    puntos, _ = recomendar_estrategia(respuestas(presupuesto_limitado_o_iterar_rapido=True))

    assert puntos["RAG"] == 2
    assert puntos["PEFT"] == 1


def test_las_variantes_multiples_favorecen_peft():
    puntos, _ = recomendar_estrategia(respuestas(necesita_multiples_variantes=True))

    assert puntos["PEFT"] == 2


# --- Empates ---------------------------------------------------------------


def test_el_empate_devuelve_todas_las_ganadoras():
    _, ganadoras = recomendar_estrategia(
        respuestas(problema_de_formato_o_tono=True, presupuesto_limitado_o_iterar_rapido=True)
    )

    # PEFT 3 y RAG 2 no empatan; el empate se arma con dos condiciones de PEFT.
    assert ganadoras == ["PEFT"]


def test_empate_real_entre_fine_tuning_y_peft():
    _, ganadoras = recomendar_estrategia(
        respuestas(problema_de_formato_o_tono=True, dataset_grande_y_de_calidad=True)
    )

    # PEFT 3, fine-tuning 3.
    assert set(ganadoras) == {"PEFT", "Fine-tuning completo"}


def test_el_empate_no_esconde_la_primera_opcion_del_diccionario():
    _, ganadoras = recomendar_estrategia(
        respuestas(problema_de_formato_o_tono=True, dataset_grande_y_de_calidad=True)
    )

    assert "Fine-tuning completo" in ganadoras


# --- Documento -------------------------------------------------------------


def test_el_esqueleto_tiene_las_cuatro_secciones_en_orden():
    esqueleto = generar_esqueleto_decision(PROBLEMA, {"RAG": 5, "PEFT": 1, "Fine-tuning completo": 0}, ["RAG"])

    posiciones = [
        esqueleto.index("## 1. El problema"),
        esqueleto.index("## 2. Resultado del método de decisión"),
        esqueleto.index("## 3. Técnica elegida y por qué"),
        esqueleto.index("## 4. Qué necesitarías para ejecutarla"),
    ]

    assert posiciones == sorted(posiciones)


def test_el_esqueleto_incluye_el_problema_y_los_puntajes():
    esqueleto = generar_esqueleto_decision(PROBLEMA, {"RAG": 5, "PEFT": 1, "Fine-tuning completo": 0}, ["RAG"])

    assert PROBLEMA in esqueleto
    assert "- RAG: 5 puntos" in esqueleto
    assert "- PEFT: 1 punto" in esqueleto


def test_el_esqueleto_marca_lo_que_falta_redactar():
    esqueleto = generar_esqueleto_decision(PROBLEMA, {"RAG": 5, "PEFT": 1, "Fine-tuning completo": 0}, ["RAG"])

    assert esqueleto.count("[COMPLETAR") == 2


def test_el_esqueleto_declara_el_empate():
    esqueleto = generar_esqueleto_decision(
        PROBLEMA, {"RAG": 0, "PEFT": 3, "Fine-tuning completo": 3}, ["Fine-tuning completo", "PEFT"]
    )

    assert "Empate entre:" in esqueleto
    assert "el puntaje no resolvió el empate" in esqueleto


def test_escribir_documento_no_sobrescribe(tmp_path):
    destino = tmp_path / "decision.md"
    destino.write_text("documento redactado a mano", encoding="utf-8")

    with pytest.raises(FileExistsError, match="ya existe"):
        escribir_documento(destino, "esqueleto nuevo")

    assert destino.read_text(encoding="utf-8") == "documento redactado a mano"


def test_escribir_documento_crea_el_archivo(tmp_path):
    destino = tmp_path / "decision.md"

    escribir_documento(destino, "contenido")

    assert destino.read_text(encoding="utf-8") == "contenido"


# --- Caso real -------------------------------------------------------------


def test_el_caso_real_tiene_las_cinco_respuestas():
    assert set(RESPUESTAS_MI_CASO) == {
        "informacion_cambia_seguido",
        "problema_de_formato_o_tono",
        "dataset_grande_y_de_calidad",
        "presupuesto_limitado_o_iterar_rapido",
        "necesita_multiples_variantes",
    }
    assert all(valor is not None for valor in RESPUESTAS_MI_CASO.values())


def test_el_caso_real_recomienda_rag_sin_empate():
    puntos, ganadoras = recomendar_estrategia(RESPUESTAS_MI_CASO)

    assert puntos == {"RAG": 5, "Fine-tuning completo": 0, "PEFT": 1}
    assert ganadoras == ["RAG"]
