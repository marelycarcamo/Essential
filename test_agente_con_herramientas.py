"""Tests del agente con herramientas (Bloque 21).

Qué se prueba y qué no
----------------------
Las herramientas se prueban como funciones: entrada, salida, y los casos que
sorprenden (consulta sin palabras útiles, modelo sin tarifas, tokens negativos,
documento ilegible). El ciclo del agente se prueba con un modelo falso que
*pide* herramientas de verdad, así que se verifica la decisión, la ejecución y
la traza sin tocar la red ni gastar cuota.

Lo que no se prueba aquí es la calidad de la respuesta del modelo real: eso es
la corrida manual de ``agente_con_herramientas.py``, que se revisa a ojo porque
depende de un servicio externo y no es reproducible.
"""

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agente_con_herramientas import (
    HERRAMIENTAS,
    MAXIMO_ITERACIONES,
    construir_agente,
    herramientas_usadas,
)
from herramientas import (
    Fragmento,
    buscar,
    buscar_en_documentacion,
    cargar_fragmentos,
    estimar_costo,
    terminos_de,
)


class ModeloFalso(BaseChatModel):
    """Modelo de chat que devuelve respuestas prefijadas, sin red.

    ``bind_tools`` se devuelve a sí mismo porque las herramientas las ejecuta el
    ejecutor, no el modelo: el modelo solo las *pide*.
    """

    respuestas: list[AIMessage]

    @property
    def _llm_type(self) -> str:
        """Identificador que exige ``BaseChatModel``."""

        return "modelo-falso"

    def bind_tools(self, tools, **kwargs):
        """Acepta las herramientas y se devuelve a sí mismo."""

        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Entrega la siguiente respuesta preparada."""

        if not self.respuestas:
            raise AssertionError("El modelo falso se quedó sin respuestas")

        return ChatResult(
            generations=[ChatGeneration(message=self.respuestas.pop(0))]
        )


def _peticion_de(nombre: str, argumentos: dict, identificador: str = "llamada_1"):
    """Mensaje del modelo pidiendo una herramienta concreta."""

    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": nombre,
                "args": argumentos,
                "id": identificador,
                "type": "tool_call",
            }
        ],
    )


# ---------------------------------------------------------------------------
# Búsqueda en la documentación
# ---------------------------------------------------------------------------


def test_los_terminos_descartan_palabras_cortas_y_tildes():
    assert terminos_de("¿Cuál es la latencia de DeepSeek?") == [
        "cual",
        "latencia",
        "deepseek",
    ]


def test_una_consulta_sin_palabras_utiles_no_aporta_terminos():
    assert terminos_de("de la que") == []


def _corpus():
    """Tres fragmentos controlados, con el mismo tema y distinto tamaño."""

    return [
        Fragmento(
            archivo="a.md",
            linea=1,
            texto="La latencia de DeepSeek Flash fue de 4,84 segundos en promedio.",
        ),
        Fragmento(
            archivo="b.md",
            linea=1,
            texto=(
                "En las pruebas de latencia, DeepSeek Flash registró 4,84 "
                "segundos de promedio y GPT-4o mini registró 8,18 segundos, "
                "según la tabla comparativa de la evaluación preliminar."
            ),
        ),
        Fragmento(
            archivo="c.md",
            linea=1,
            texto="El costo del modelo se calcula por tokens de entrada y salida.",
        ),
    ]


def test_buscar_encuentra_los_fragmentos_del_tema_y_descarta_el_ajeno():
    encontrados = buscar("latencia DeepSeek", _corpus())

    assert {fragmento.archivo for fragmento in encontrados} == {"a.md", "b.md"}
    assert "c.md" not in {fragmento.archivo for fragmento in encontrados}


def test_buscar_prefiere_el_fragmento_mas_breve_entre_los_que_empatan():
    encontrados = buscar("latencia DeepSeek", _corpus())

    assert encontrados[0].archivo == "a.md"


def test_buscar_no_devuelve_nada_si_ningun_termino_aparece():
    assert buscar("kubernetes", _corpus()) == []


def test_un_termino_no_coincide_dentro_de_otra_palabra():
    # Caso real: "gemini" contiene "mini" y colaba un fragmento sobre el
    # proveedor de respaldo en una pregunta sobre el modelo descartado.
    corpus = [
        Fragmento(
            archivo="a.md",
            linea=1,
            texto="Se descarta GPT-4o mini como opción principal por latencia.",
        ),
        Fragmento(
            archivo="b.md",
            linea=1,
            texto="Gemini queda como proveedor de respaldo del sistema.",
        ),
    ]

    encontrados = buscar("mini", corpus)

    assert [fragmento.archivo for fragmento in encontrados] == ["a.md"]


def test_un_termino_encuentra_su_plural():
    corpus = [
        Fragmento(
            archivo="a.md",
            linea=1,
            texto="Las tarifas se consultaron en la documentación oficial.",
        )
    ]

    assert buscar("tarifa", corpus) != []


def test_buscar_respeta_el_maximo_pedido():
    assert len(buscar("deepseek", _corpus(), maximo=1)) == 1


def test_cargar_fragmentos_guarda_la_linea_donde_empieza_cada_parrafo(tmp_path):
    documento = tmp_path / "notas.md"
    documento.write_text(
        "# Título corto\n"
        "\n"
        "Primer párrafo con suficiente texto como para ser un fragmento útil.\n"
        "\n"
        "Segundo párrafo, también largo, que empieza más abajo en el archivo.\n",
        encoding="utf-8",
    )

    fragmentos = cargar_fragmentos([documento])

    assert [fragmento.linea for fragmento in fragmentos] == [3, 5]


def test_cargar_fragmentos_descarta_los_bloques_demasiado_cortos(tmp_path):
    documento = tmp_path / "notas.md"
    documento.write_text(
        "# Título\n"
        "\n"
        "Un párrafo largo, con contexto suficiente para citarse sin confundir.\n",
        encoding="utf-8",
    )

    fragmentos = cargar_fragmentos([documento])

    assert len(fragmentos) == 1
    assert fragmentos[0].texto.startswith("Un párrafo largo")


def test_cargar_fragmentos_descarta_el_texto_alternativo_de_una_imagen(tmp_path):
    # Caso real: el README describe un diagrama con cientos de caracteres y esa
    # descripción ganaba la búsqueda sin responder nada.
    documento = tmp_path / "readme.md"
    documento.write_text(
        "## Diagrama\n"
        "\n"
        "![Diagrama de flujo para consumir OpenRouter con cajas de colores que "
        "muestran el recorrido completo de la pregunta hasta la respuesta](x.png)\n"
        "\n"
        "Este párrafo sí explica en palabras qué hace el diagrama y por qué.\n",
        encoding="utf-8",
    )

    fragmentos = cargar_fragmentos([documento])

    assert len(fragmentos) == 1
    assert "![" not in fragmentos[0].texto
    assert fragmentos[0].texto.startswith("Este párrafo")


def test_cargar_fragmentos_salta_el_archivo_ilegible_y_sigue(tmp_path, caplog):
    faltante = tmp_path / "no_existe.md"
    presente = tmp_path / "existe.md"
    presente.write_text(
        "Contenido suficiente para que este párrafo cuente como fragmento.",
        encoding="utf-8",
    )

    with caplog.at_level("WARNING"):
        fragmentos = cargar_fragmentos([faltante, presente])

    assert len(fragmentos) == 1
    assert "no_existe.md" in caplog.text


def test_la_herramienta_de_busqueda_lee_la_documentacion_real_del_proyecto():
    resultado = buscar_en_documentacion.invoke({"consulta": "latencia"})

    assert "No encontré" not in resultado
    assert "[" in resultado


def test_la_herramienta_de_busqueda_avisa_si_la_consulta_no_aporta_terminos():
    resultado = buscar_en_documentacion.invoke({"consulta": "de la"})

    assert "Dame términos concretos" in resultado


def test_la_herramienta_de_busqueda_avisa_cuando_no_hay_resultados():
    resultado = buscar_en_documentacion.invoke({"consulta": "kubernetes"})

    assert "No encontré nada" in resultado
    assert "Archivos revisados" in resultado


# ---------------------------------------------------------------------------
# Estimación de costo
# ---------------------------------------------------------------------------


def test_estimar_costo_devuelve_un_monto_para_un_modelo_registrado():
    resultado = estimar_costo.invoke(
        {
            "modelo": "deepseek-flash",
            "tokens_entrada": 1500,
            "tokens_salida": 400,
        }
    )

    assert "US$" in resultado
    assert "deepseek-flash" in resultado
    assert "punta" in resultado or "valle" in resultado


def test_estimar_costo_dice_que_asume_la_cache_vacia():
    resultado = estimar_costo.invoke(
        {"modelo": "deepseek-flash", "tokens_entrada": 10, "tokens_salida": 10}
    )

    assert "caché" in resultado


def test_estimar_costo_avisa_de_un_modelo_sin_tarifas_sin_romper_el_agente():
    resultado = estimar_costo.invoke(
        {"modelo": "gpt-4o-mini", "tokens_entrada": 100, "tokens_salida": 100}
    )

    assert "No tengo tarifas" in resultado
    assert "deepseek-flash" in resultado


@pytest.mark.parametrize(
    "entrada, salida", [(-1, 100), (100, -1), (-5, -5)]
)
def test_estimar_costo_rechaza_tokens_negativos(entrada, salida):
    resultado = estimar_costo.invoke(
        {
            "modelo": "deepseek-flash",
            "tokens_entrada": entrada,
            "tokens_salida": salida,
        }
    )

    assert "negativos" in resultado


# ---------------------------------------------------------------------------
# El ciclo del agente: decisión, ejecución y traza
# ---------------------------------------------------------------------------


def test_las_herramientas_declaradas_son_las_dos_del_bloque():
    assert [herramienta.name for herramienta in HERRAMIENTAS] == [
        "buscar_en_documentacion",
        "estimar_costo",
    ]


def test_el_ejecutor_activa_la_traza_y_limita_las_vueltas():
    ejecutor = construir_agente(ModeloFalso(respuestas=[]))

    assert ejecutor.return_intermediate_steps is True
    assert ejecutor.max_iterations == MAXIMO_ITERACIONES
    assert ejecutor.handle_parsing_errors is True


def test_el_agente_ejecuta_la_herramienta_que_pide_el_modelo():
    modelo = ModeloFalso(
        respuestas=[
            _peticion_de(
                "estimar_costo",
                {
                    "modelo": "deepseek-flash",
                    "tokens_entrada": 1500,
                    "tokens_salida": 400,
                },
            ),
            AIMessage(content="Cuesta US$ 0,0007 aproximadamente."),
        ]
    )

    resultado = construir_agente(modelo).invoke({"input": "¿Cuánto cuesta?"})

    assert herramientas_usadas(resultado["intermediate_steps"]) == ["estimar_costo"]
    assert resultado["output"] == "Cuesta US$ 0,0007 aproximadamente."


def test_la_traza_guarda_los_argumentos_y_la_observacion():
    modelo = ModeloFalso(
        respuestas=[
            _peticion_de("buscar_en_documentacion", {"consulta": "latencia"}),
            AIMessage(content="Listo."),
        ]
    )

    resultado = construir_agente(modelo).invoke({"input": "¿Qué latencia se midió?"})
    accion, observacion = resultado["intermediate_steps"][0]

    assert accion.tool_input == {"consulta": "latencia"}
    assert "latencia" in str(observacion).lower()


def test_si_el_modelo_no_pide_herramientas_no_hay_pasos():
    modelo = ModeloFalso(respuestas=[AIMessage(content="Puedo buscar y estimar.")])

    resultado = construir_agente(modelo).invoke({"input": "¿Qué puedes hacer?"})

    assert resultado["intermediate_steps"] == []
    assert resultado["output"] == "Puedo buscar y estimar."


def test_un_modelo_que_insiste_se_corta_al_llegar_al_tope_de_vueltas():
    # Un agente sin tope es un agente que puede gastar sin límite: esta prueba
    # fija el comportamiento, no lo deja a la casualidad del proveedor.
    respuestas = [
        _peticion_de("estimar_costo", {"modelo": "deepseek-flash", "tokens_entrada": 1, "tokens_salida": 1}, f"llamada_{numero}")
        for numero in range(MAXIMO_ITERACIONES + 2)
    ]

    resultado = construir_agente(ModeloFalso(respuestas=respuestas)).invoke(
        {"input": "Insiste para siempre"}
    )

    assert "max iterations" in resultado["output"]
    assert len(resultado["intermediate_steps"]) == MAXIMO_ITERACIONES
