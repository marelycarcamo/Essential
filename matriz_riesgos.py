"""Matriz de riesgos del sistema RAG (Bloque 32).

Qué hace
--------
Puntúa cada riesgo del proyecto en probabilidad × impacto (1 a 5), lo clasifica y
—esto es lo que la convierte en una herramienta y no en una lista de miedos—
exige que cada riesgo **ALTO** o **CRÍTICO** tenga un punto de supervisión humana
concreto asociado.

Por qué cada fila nombra un mecanismo y un bloque
-------------------------------------------------
"Mejorar la seguridad" no es una mitigación: no se puede verificar, no se puede
fechar y no se puede revisar. Cada fila apunta a algo que ya existe en el
repositorio, con su número de bloque; si el mecanismo no existe, entonces el
riesgo sigue abierto y hay que decirlo.

Por qué no todos los riesgos son CRÍTICOS
------------------------------------------
Calificar todo como grave "por las dudas" vacía de sentido la priorización. La
disciplina de puntuar cada fila por separado es lo que permite decidir qué se
atiende primero con el tiempo que hay.

Uso
---
    ./.venv/Scripts/python.exe matriz_riesgos.py
"""

from __future__ import annotations

import sys

# Cada fila: qué puede pasar, qué vía usa para entrar, cuánto se mitiga y con qué
# mecanismo existente, y quién o qué frena la acción antes de que ocurra.
RIESGOS: list[dict] = [
    {
        "riesgo": "Inyección de prompt a través de un documento indexado (texto escrito por terceros dentro del corpus)",
        "probabilidad": 3,
        "impacto": 4,
        "mitigacion": "Filtro de patrones sobre cada documento recuperado + corte a RESPUESTA_SIN_CONTEXTO si no queda contexto limpio, y prompt v2 que declara el contexto como material no ejecutable (Bloque 31)",
        "supervision": "Revisión humana de los eventos documento_descartado_por_inyeccion en la traza antes de indexar un corpus nuevo",
    },
    {
        "riesgo": "Respuesta sin cita o con una cita que apunta a un fragmento que no existe (alucinación de formato)",
        "probabilidad": 3,
        "impacto": 4,
        "mitigacion": "verificar_citas compara cada [n] de la respuesta con el mapa real de fragmentos y devuelve los problemas sin bloquear (Bloque 26); test_2 del Gate exige que la respuesta nombre su fuente",
        "supervision": "Revisión humana de las respuestas con problemas_citas no vacío antes de publicar material generado",
    },
    {
        "riesgo": "Fuga de datos personales en los logs (la pregunta del usuario o el contenido de un documento quedan en texto plano)",
        "probabilidad": 3,
        "impacto": 5,
        "mitigacion": "El logging registra longitud y huella de la pregunta, y la fuente del documento, nunca su texto; test_3 del Gate lo verifica con un correo señuelo (Bloque 31)",
        "supervision": "Revisión humana del archivo de logs antes de cada entrega, además de la prueba automática del criterio",
    },
    {
        "riesgo": "Despliegue de una versión de prompt sin validar a todas las respuestas siguientes",
        "probabilidad": 2,
        "impacto": 5,
        "mitigacion": "Registro de versiones con huella, autor, nota y modelo probado; la respuesta registra la huella del prompt que la generó (Bloque 32)",
        "supervision": "Grafo de supervisión con interrupt() y Command: la versión no se activa sin aprobación humana explícita (Bloque 32)",
    },
    {
        "riesgo": "Índice desactualizado respecto del código: responde con documentación que ya no corresponde",
        "probabilidad": 4,
        "impacto": 3,
        "mitigacion": "El modelo de embeddings y los parámetros de partición quedan en la metadata de la colección y se verifican al abrir el índice; la actualización es incremental por ids predecibles (Bloques 24 y 25)",
        "supervision": "Revisión humana del diff de fuentes del corpus antes de reindexar en producción",
    },
    {
        "riesgo": "Caída del proveedor de embeddings (Gemini) deja el RAG sin recuperación, porque DeepSeek no ofrece ese endpoint",
        "probabilidad": 3,
        "impacto": 3,
        "mitigacion": "verificar_indice al arrancar corta con un mensaje claro en vez de responder sin contexto; el costo del fallo es visible, no silencioso (Bloque 24)",
        "supervision": "Alerta al operador; sin índice verificado el servicio no responde",
    },
    {
        "riesgo": "Costo descontrolado por consultas repetidas o por un k demasiado alto",
        "probabilidad": 2,
        "impacto": 2,
        "mitigacion": "Registro de costo por consulta con tarifas por componente y techo de k medido con precisión y recall (Bloques 27 y 28)",
        "supervision": "",
    },
]

CATEGORIAS = {
    "CRITICO": 15,
    "ALTO": 9,
    "MEDIO": 4,
    "BAJO": 0,
}


def puntaje_riesgo(probabilidad: int, impacto: int) -> int:
    """Producto de probabilidad e impacto, ambos de 1 a 5."""

    if not (1 <= probabilidad <= 5 and 1 <= impacto <= 5):
        raise ValueError(
            f"probabilidad e impacto deben estar entre 1 y 5, no {probabilidad} y {impacto}"
        )
    return probabilidad * impacto


def categoria_riesgo(puntaje: int) -> str:
    """Clasifica el puntaje en CRITICO, ALTO, MEDIO o BAJO."""

    for categoria, umbral in CATEGORIAS.items():
        if puntaje >= umbral:
            return categoria
    return "BAJO"


def esta_en_el_limite(probabilidad: int, impacto: int) -> bool:
    """Indica si un solo punto de diferencia cambia la categoría del riesgo.

    Esos son los candidatos a que otra persona revise la estimación: la decisión
    de categoría está a un punto de moverse, y un punto es exactamente lo que se
    discute en una reunión.

    Advertencia honesta: con los umbrales de este programa (4, 9 y 15) sobre una
    escala de 1 a 5, casi cualquier producto cae a un punto de un límite, así que
    la marca distingue poco. Sirve como recordatorio de que la etiqueta es gruesa
    y el puntaje exacto es el que ordena el trabajo.
    """

    actual = categoria_riesgo(puntaje_riesgo(probabilidad, impacto))
    vecinos = [
        (probabilidad + 1, impacto),
        (probabilidad - 1, impacto),
        (probabilidad, impacto + 1),
        (probabilidad, impacto - 1),
    ]
    for p2, i2 in vecinos:
        if not (1 <= p2 <= 5 and 1 <= i2 <= 5):
            continue
        if categoria_riesgo(puntaje_riesgo(p2, i2)) != actual:
            return True
    return False


def evaluar_matriz(riesgos: list[dict] | None = None) -> list[dict]:
    """Agrega puntaje, categoría y marca de límite a cada fila. No muta la entrada."""

    riesgos = RIESGOS if riesgos is None else riesgos
    evaluados = []
    for original in riesgos:
        fila = dict(original)
        fila["puntaje"] = puntaje_riesgo(fila["probabilidad"], fila["impacto"])
        fila["categoria"] = categoria_riesgo(fila["puntaje"])
        fila["al_limite"] = esta_en_el_limite(fila["probabilidad"], fila["impacto"])
        evaluados.append(fila)
    return sorted(evaluados, key=lambda fila: -fila["puntaje"])


def validar_matriz(riesgos: list[dict] | None = None) -> list[str]:
    """Revisa que la matriz cumpla lo que la rúbrica exige.

    Returns:
        Lista de problemas. Vacía si la matriz está completa.
    """

    riesgos = RIESGOS if riesgos is None else riesgos
    problemas: list[str] = []

    if len(riesgos) < 5:
        problemas.append(f"hay {len(riesgos)} riesgos; el mínimo es 5")

    con_mecanismo = 0
    for posicion, fila in enumerate(riesgos):
        for campo in ("riesgo", "probabilidad", "impacto", "mitigacion", "supervision"):
            if campo not in fila:
                problemas.append(f"fila {posicion}: falta el campo '{campo}'")

        if not str(fila.get("riesgo", "")).strip():
            problemas.append(f"fila {posicion}: el riesgo está vacío")

        mitigacion = str(fila.get("mitigacion", ""))
        if "Bloque" in mitigacion:
            con_mecanismo += 1

        try:
            categoria = categoria_riesgo(
                puntaje_riesgo(fila["probabilidad"], fila["impacto"])
            )
        except (ValueError, KeyError) as error:
            problemas.append(f"fila {posicion}: {error}")
            continue

        if categoria in ("ALTO", "CRITICO") and not str(fila.get("supervision", "")).strip():
            problemas.append(
                f"fila {posicion}: riesgo {categoria} sin punto de supervisión asociado"
            )

    if con_mecanismo < 2:
        problemas.append(
            f"solo {con_mecanismo} mitigaciones nombran un bloque del programa; el mínimo es 2"
        )

    if not any(
        categoria_riesgo(puntaje_riesgo(f["probabilidad"], f["impacto"])) in ("ALTO", "CRITICO")
        for f in riesgos
    ):
        problemas.append(
            "no hay ningún riesgo ALTO ni CRITICO: revisa si estás subestimando probabilidad o impacto"
        )

    return problemas


def main() -> int:
    evaluados = evaluar_matriz()

    print(f"{'CATEGORIA':9} {'PUNTAJE':>7}  RIESGO")
    for fila in evaluados:
        marca = " [AL LIMITE]" if fila["al_limite"] else ""
        print(f"{fila['categoria']:9} {fila['puntaje']:>7}  {fila['riesgo'][:78]}{marca}")

    print("\nRiesgos que conviene revisar con otra persona:")
    for fila in evaluados:
        if fila["al_limite"]:
            print(f"  - [{fila['categoria']}] {fila['riesgo'][:70]}")

    print("\nPuntos de supervisión de los riesgos ALTO o CRÍTICO:")
    for fila in evaluados:
        if fila["categoria"] in ("ALTO", "CRITICO"):
            print(f"  - {fila['riesgo'][:60]}...")
            print(f"    -> {fila['supervision']}")

    problemas = validar_matriz()
    print(f"\n{len(evaluados)} riesgos, {len(problemas)} problemas")
    for problema in problemas:
        print(f"  - {problema}")

    return 1 if problemas else 0


if __name__ == "__main__":
    sys.exit(main())
