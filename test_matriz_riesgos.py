"""Tests de la matriz de riesgos (Bloque 32).

Qué se prueba y qué no
----------------------
Se prueban las fórmulas, los umbrales de categoría, la marca de límite y —lo que
la rúbrica exige— que toda fila ALTA o CRÍTICA tenga un punto de supervisión y que
la matriz real esté completa. Los tests de validación se hacen con matrices rotas
a propósito: un validador que solo ve matrices buenas no demuestra nada.

Lo que no se prueba aquí es si la probabilidad asignada a cada riesgo es la
correcta: eso es criterio, y se revisa con otra persona, no con pytest.
"""

import pytest

from matriz_riesgos import (
    CATEGORIAS,
    RIESGOS,
    categoria_riesgo,
    esta_en_el_limite,
    evaluar_matriz,
    puntaje_riesgo,
    validar_matriz,
)


def fila(probabilidad: int = 3, impacto: int = 3, supervision: str = "algo") -> dict:
    return {
        "riesgo": "un riesgo concreto",
        "probabilidad": probabilidad,
        "impacto": impacto,
        "mitigacion": "mecanismo (Bloque 31)",
        "supervision": supervision,
    }


# --- Puntaje y categoría ---------------------------------------------------


def test_el_puntaje_es_el_producto():
    assert puntaje_riesgo(3, 4) == 12


@pytest.mark.parametrize("probabilidad,impacto", [(0, 3), (6, 3), (3, 0), (3, 9)])
def test_el_puntaje_rechaza_valores_fuera_de_escala(probabilidad, impacto):
    with pytest.raises(ValueError, match="entre 1 y 5"):
        puntaje_riesgo(probabilidad, impacto)


@pytest.mark.parametrize(
    "puntaje,categoria",
    [(15, "CRITICO"), (16, "CRITICO"), (25, "CRITICO"), (14, "ALTO"), (9, "ALTO"),
     (8, "MEDIO"), (4, "MEDIO"), (3, "BAJO"), (1, "BAJO")],
)
def test_los_umbrales_de_categoria(puntaje, categoria):
    assert categoria_riesgo(puntaje) == categoria


def test_los_umbrales_estan_en_orden_descendente():
    assert CATEGORIAS["CRITICO"] > CATEGORIAS["ALTO"] > CATEGORIAS["MEDIO"] > CATEGORIAS["BAJO"]


# --- Marca de límite -------------------------------------------------------


def test_un_punto_cambia_la_categoria_en_un_riesgo_al_limite():
    # 3x5 = 15 CRITICO; con impacto 4 sería 12 ALTO.
    assert esta_en_el_limite(3, 5) is True


def test_un_riesgo_que_no_esta_cerca_de_un_umbral_no_se_marca():
    # 1x1 = 1 BAJO, y todos sus vecinos también son BAJO.
    assert esta_en_el_limite(1, 1) is False


def test_la_marca_no_rompe_en_los_bordes_de_la_escala():
    assert isinstance(esta_en_el_limite(5, 5), bool)


# --- Evaluación de la matriz ----------------------------------------------


def test_evaluar_agrega_puntaje_categoria_y_limite():
    evaluados = evaluar_matriz([fila(3, 5)])

    assert evaluados[0]["puntaje"] == 15
    assert evaluados[0]["categoria"] == "CRITICO"
    assert "al_limite" in evaluados[0]


def test_evaluar_ordena_de_mayor_a_menor_puntaje():
    evaluados = evaluar_matriz([fila(1, 1), fila(4, 5), fila(2, 2)])

    assert [f["puntaje"] for f in evaluados] == [20, 4, 1]


def test_evaluar_no_modifica_la_lista_original():
    original = fila(2, 2)

    evaluar_matriz([original])

    assert "puntaje" not in original


# --- Validación ------------------------------------------------------------


def test_la_matriz_real_esta_completa():
    assert validar_matriz() == []


def test_la_matriz_real_tiene_al_menos_un_riesgo_alto_o_critico():
    categorias = {fila["categoria"] for fila in evaluar_matriz()}

    assert categorias & {"ALTO", "CRITICO"}


def test_detecta_menos_de_cinco_riesgos():
    problemas = validar_matriz([fila()])

    assert any("el mínimo es 5" in problema for problema in problemas)


def test_detecta_riesgo_alto_sin_supervision():
    problemas = validar_matriz([fila(3, 4, supervision="") for _ in range(5)])

    assert any("sin punto de supervisión" in problema for problema in problemas)


def test_un_riesgo_medio_no_exige_supervision():
    riesgos = [fila(2, 2, supervision="") for _ in range(5)]

    problemas = validar_matriz(riesgos)

    assert not any("sin punto de supervisión" in problema for problema in problemas)


def test_detecta_campo_faltante():
    incompleta = fila()
    del incompleta["mitigacion"]

    problemas = validar_matriz([incompleta] * 5)

    assert any("falta el campo 'mitigacion'" in problema for problema in problemas)


def test_detecta_mitigaciones_sin_mecanismo_existente():
    riesgos = [
        {**fila(2, 2), "mitigacion": "mejorar la seguridad"} for _ in range(5)
    ]

    problemas = validar_matriz(riesgos)

    assert any("el mínimo es 2" in problema for problema in problemas)


def test_detecta_una_matriz_sin_ningun_riesgo_grave():
    riesgos = [fila(1, 1) for _ in range(5)]

    problemas = validar_matriz(riesgos)

    assert any("subestimando" in problema for problema in problemas)


def test_toda_fila_real_declara_su_punto_de_supervision():
    for riesgo in RIESGOS:
        assert "supervision" in riesgo
