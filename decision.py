"""Método de decisión de personalización y esqueleto del documento (Bloque 30).

Qué hace
--------
Puntúa las tres estrategias —RAG, fine-tuning completo y PEFT— a partir de cinco
preguntas respondidas con evidencia del proyecto, y genera el esqueleto del
documento de decisión con las cuatro secciones que un equipo real espera antes
de invertir tiempo en personalizar un modelo.

Por qué las respuestas llevan un número al lado
----------------------------------------------
Porque una respuesta sin evidencia es una impresión. La tabla del material da un
criterio cuantitativo para cada pregunta, y cada valor de ``RESPUESTAS_MI_CASO``
está acompañado del dato que lo sustenta. Si mañana alguien discute la decisión,
la discusión se hace sobre el número, no sobre la memoria de quien la tomó.

Por qué el documento no se regenera solo
----------------------------------------
El esqueleto se genera, pero las secciones 3 y 4 se redactan a mano: son el
criterio y el plan de trabajo, y el puntaje no los puede escribir. Por eso el
script no sobrescribe un documento ya redactado: perdería justo lo que no se
puede reconstruir ejecutando código otra vez.

Uso
---
    ./.venv/Scripts/python.exe decision.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ARCHIVO_DECISION = Path("decision_personalizacion.md")

PROBLEMA = (
    "Las respuestas del banco de prompts no se apoyan en ninguna fuente "
    "verificable: ante la sigla PEP respondió con su significado en el ámbito "
    "financiero, y no hay forma de auditar de dónde sale una afirmación ni de "
    "corregirla sin cambiar el modelo."
)

# Cada clave es una pregunta del método; cada valor es la respuesta y el dato
# que la sostiene (ese dato es lo que va a las secciones 1 y 4).
RESPUESTAS_MI_CASO: dict[str, bool] = {
    # Las 6 fuentes del corpus se crearon en esta misma sesión, y las 17 fuentes
    # restantes del proyecto cambian con cada bloque: el contenido que el sistema
    # necesita consultar no dura un mes quieto.
    "informacion_cambia_seguido": True,
    # Se contaron las 20 respuestas reales guardadas en docs/resultados_metricas.json:
    # las 20 cumplen la instrucción de responder en una sola oración (0 % de
    # incumplimiento, contra un umbral de 15 %). El formato no es el problema.
    "problema_de_formato_o_tono": False,
    # Hay 13 ejemplos etiquetados en dataset_tuning.py; el método pide 200.
    "dataset_grande_y_de_calidad": False,
    # No hay presupuesto para iterar entrenamientos: cada intento cuesta y el
    # plan gratuito no alcanza para medir el efecto de un cambio en menos de un día.
    "presupuesto_limitado_o_iterar_rapido": True,
    # Un solo sistema, un solo idioma y una sola línea de producto.
    "necesita_multiples_variantes": False,
}


def recomendar_estrategia(respuestas: dict[str, bool]) -> tuple[dict[str, int], list[str]]:
    """Puntúa las tres estrategias y devuelve el puntaje y las ganadoras.

    Devuelve **todas** las que empataron en el máximo, no solo la primera del
    diccionario. Presentar la primera como ganadora cuando hay empate es un error
    que esconde justo la parte que necesita criterio humano.
    """

    puntos = {"RAG": 0, "Fine-tuning completo": 0, "PEFT": 0}

    if respuestas["informacion_cambia_seguido"]:
        puntos["RAG"] += 3
    if respuestas["problema_de_formato_o_tono"]:
        puntos["PEFT"] += 2
        puntos["Fine-tuning completo"] += 1
    if respuestas["dataset_grande_y_de_calidad"]:
        puntos["Fine-tuning completo"] += 2
        puntos["PEFT"] += 1
    if respuestas["presupuesto_limitado_o_iterar_rapido"]:
        puntos["RAG"] += 2
        puntos["PEFT"] += 1
    if respuestas["necesita_multiples_variantes"]:
        puntos["PEFT"] += 2

    maximo = max(puntos.values())
    ganadoras = [tecnica for tecnica, puntaje in puntos.items() if puntaje == maximo]
    return puntos, ganadoras


def generar_esqueleto_decision(problema: str, puntos: dict[str, int], ganadoras: list[str]) -> str:
    """Arma el documento con las cuatro secciones y marca lo que falta redactar."""

    empate = len(ganadoras) > 1
    lineas = ["# Decisión de personalización", "", "## 1. El problema", problema, ""]

    lineas.append("## 2. Resultado del método de decisión")
    for tecnica, puntaje in sorted(puntos.items(), key=lambda par: -par[1]):
        lineas.append(f"- {tecnica}: {puntaje} puntos")
    if empate:
        lineas.extend(["", f"**Empate entre: {', '.join(ganadoras)}**"])

    lineas.extend(["", "## 3. Técnica elegida y por qué"])
    if empate:
        lineas.append(
            "[COMPLETAR: el puntaje no resolvió el empate solo -- "
            "justifica aquí con criterio propio]"
        )
    else:
        lineas.append(f"[COMPLETAR: por qué {ganadoras[0]} responde mejor a este caso]")

    lineas.extend(
        [
            "",
            "## 4. Qué necesitarías para ejecutarla",
            "[COMPLETAR: corpus faltante, o cantidad de ejemplos etiquetados que faltan]",
        ]
    )

    return "\n".join(lineas) + "\n"


def escribir_documento(ruta: Path, contenido: str) -> None:
    """Crea el documento. Se niega a sobrescribir uno que ya exista.

    Raises:
        FileExistsError: si el documento ya está redactado.
    """

    if ruta.exists():
        raise FileExistsError(
            f"{ruta} ya existe. Bórralo a mano si quieres regenerar el esqueleto: "
            "sobrescribirlo perdería las secciones redactadas."
        )

    ruta.write_text(contenido, encoding="utf-8")


def main() -> int:
    if any(valor is None for valor in RESPUESTAS_MI_CASO.values()):
        raise SystemExit("Completa las cinco respuestas con True o False antes de continuar.")

    puntos, ganadoras = recomendar_estrategia(RESPUESTAS_MI_CASO)
    print(puntos)
    print("candidatas:", ganadoras)

    esqueleto = generar_esqueleto_decision(PROBLEMA, puntos, ganadoras)
    print()
    print(esqueleto)

    try:
        escribir_documento(ARCHIVO_DECISION, esqueleto)
    except FileExistsError as error:
        print(f"AVISO: {error}")
        return 1

    print(f"Esqueleto guardado en {ARCHIVO_DECISION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
