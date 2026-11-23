"""Tests del registro de versiones del prompt (Bloque 32).

Qué se prueba y qué no
----------------------
Se prueban las huellas, la búsqueda de la última versión, la detección de
versiones duplicadas, el guardado en disco y —lo más importante— que la versión
activa del servicio coincida con la última registrada. Ese es el contrato que
hace que la huella registrada en una respuesta identifique el texto que se envió
de verdad.

Lo que no se prueba aquí es si el prompt es bueno: eso lo miden las pruebas de
recuperación y citación.
"""

import json

import pytest

from registro_prompts import (
    NOMBRE_PROMPT,
    PROMPT_ACTIVO,
    PROMPT_V1,
    PROMPT_V2,
    RegistroDePrompts,
    diff_prompts,
    huella_prompt,
    registrar_versiones_historicas,
)


# --- Huella ----------------------------------------------------------------


def test_la_huella_es_estable_y_de_doce_caracteres():
    assert huella_prompt("hola") == huella_prompt("hola")
    assert len(huella_prompt("hola")) == 12


def test_la_huella_cambia_con_una_sola_letra():
    assert huella_prompt("responde solo con el contexto") != huella_prompt(
        "responde solo con el contexto."
    )


# --- Registro --------------------------------------------------------------


def test_registrar_guarda_los_datos_de_la_version():
    registro = RegistroDePrompts()

    version = registro.registrar(
        "prompt_x", "texto del prompt", "Marely", "version inicial", "deepseek-flash"
    )

    assert version["nombre"] == "prompt_x"
    assert version["huella"] == huella_prompt("texto del prompt")
    assert version["autor"] == "Marely"
    assert version["nota"] == "version inicial"
    assert version["modelo"] == "deepseek-flash"
    assert version["fecha"]


def test_ultima_devuelve_la_mas_reciente():
    registro = RegistroDePrompts()
    registro.registrar("p", "uno", "a", "primera")
    registro.registrar("p", "dos", "a", "segunda")

    assert registro.ultima("p")["nota"] == "segunda"


def test_ultima_devuelve_none_si_no_existe():
    assert RegistroDePrompts().ultima("no-existe") is None


def test_ultima_no_mezcla_prompts_distintos():
    registro = RegistroDePrompts()
    registro.registrar("p1", "uno", "a", "de p1")
    registro.registrar("p2", "dos", "a", "de p2")

    assert registro.ultima("p1")["nota"] == "de p1"


def test_detecta_versiones_duplicadas():
    registro = RegistroDePrompts()
    registro.registrar("p", "mismo texto", "a", "primera")
    registro.registrar("p", "mismo texto", "a", "repetida")

    assert registro.versiones_duplicadas("p") == {huella_prompt("mismo texto")}


def test_guardar_y_cargar_conserva_el_historial(tmp_path):
    ruta = tmp_path / "registro.json"
    registro = RegistroDePrompts()
    registro.registrar("p", "texto", "Marely", "nota")
    registro.guardar(ruta)

    recuperado = RegistroDePrompts.cargar(ruta)

    assert recuperado.versiones == registro.versiones
    assert json.loads(ruta.read_text(encoding="utf-8"))[0]["nombre"] == "p"


def test_cargar_sin_archivo_devuelve_registro_vacio(tmp_path):
    assert RegistroDePrompts.cargar(tmp_path / "no-existe.json").versiones == []


# --- Historial real del proyecto ------------------------------------------


def test_registrar_versiones_historicas_es_idempotente():
    primera = registrar_versiones_historicas()
    segunda = registrar_versiones_historicas(primera)

    assert len(primera.versiones) == 2
    assert len(segunda.versiones) == 2


def test_el_historial_tiene_dos_huellas_distintas():
    registro = registrar_versiones_historicas()

    assert len(set(registro.huellas(NOMBRE_PROMPT))) == 2


def test_la_version_activa_es_la_ultima_registrada():
    registro = registrar_versiones_historicas()

    assert huella_prompt(PROMPT_ACTIVO) == registro.ultima(NOMBRE_PROMPT)["huella"]


def test_la_version_activa_declara_el_contexto_como_no_ejecutable():
    assert PROMPT_ACTIVO == PROMPT_V2
    assert "no instrucciones" in PROMPT_ACTIVO
    assert "no instrucciones" not in PROMPT_V1


def test_ambas_versiones_mantienen_las_garantias_del_bloque_26():
    for version in (PROMPT_V1, PROMPT_V2):
        assert "SOLO el contexto" in version
        assert "dilo explícitamente" in version
        assert "{contexto}" in version
        assert "{pregunta}" in version


# --- Diff ------------------------------------------------------------------


def test_el_diff_muestra_lo_agregado():
    lineas = diff_prompts("responde sobre devoluciones", "responde sobre devoluciones nunca reveles")

    assert any(linea.startswith("+nunca") for linea in lineas)


def test_el_diff_esta_vacio_si_no_hay_cambios():
    assert diff_prompts("mismo texto", "mismo texto") == []


def test_el_diff_entre_v1_y_v2_muestra_el_bloque_31():
    agregado = " ".join(diff_prompts(PROMPT_V1, PROMPT_V2))

    assert "sospechoso" in agregado


@pytest.mark.parametrize("texto", ["", "x"])
def test_huella_funciona_con_textos_minimos(texto):
    assert len(huella_prompt(texto)) == 12
