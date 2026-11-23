"""Cuenta cuántas de las 20 respuestas reales respetan la instrucción de una sola oración.

Es la comprobación que pide el método de decisión del Bloque 30 para responder
"problema_de_formato_o_tono" con un número y no con una impresión: se toma una
lista fija de criterios y se cuenta el incumplimiento sobre respuestas reales ya
guardadas, sin gastar una sola llamada.

Criterio fijo: la consulta pide "en una sola oración", así que una respuesta
cumple si tiene exactamente una oración terminada en punto, signo de
interrogación o exclamación.

Uso
---
    ./.venv/Scripts/python.exe experimentos/contar_oraciones.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
ARCHIVO = RAIZ / "docs" / "resultados_metricas.json"

# Fin de oración seguido de espacio y mayúscula, o fin de texto. Evita contar
# decimales (0.55) y abreviaturas cortas seguidas de minúscula.
FIN_DE_ORACION = re.compile(r"[.!?]+(?=\s+[A-ZÁÉÍÓÚÑ¿¡]|\s*$)")


def contar_oraciones(texto: str) -> int:
    return max(len(FIN_DE_ORACION.findall(texto.strip())), 1)


def main() -> int:
    registros = json.loads(ARCHIVO.read_text(encoding="utf-8"))

    incumplen = 0
    for registro in registros:
        oraciones = contar_oraciones(registro["respuesta"])
        if oraciones > 1:
            incumplen += 1
        print(f"{oraciones:2} oraciones  f1={registro['f1']:.2f}  {registro['consulta'][:50]}")

    total = len(registros)
    print(f"\n{total} respuestas revisadas")
    print(f"{incumplen} no respetan la instrucción ({incumplen / total:.0%})")
    print(f"umbral del método: 3 o más de 20 (15 %)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
