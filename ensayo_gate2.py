"""Ensayo funcional del Gate 2 con proveedores reales (Bloque S1).

Qué hace
--------
Corre el pipeline completo como lo va a correr quien evalúa, pero contra los
proveedores reales: indexa el corpus desde cero en un directorio aparte, recupera,
filtra, cita, genera y registra. Al final comprueba los cuatro criterios no
compensables sobre la corrida de verdad y deja un registro por escrito.

Por qué en un directorio aparte
-------------------------------
Para no tocar el índice de desarrollo: así el ensayo es reproducible desde cero
(que es justamente lo que hay que probar) y no depende de que el índice de trabajo
esté al día.

Costo del ensayo
----------------
Una llamada de lote a los embeddings de Gemini (31 fragmentos) y una consulta real
al modelo de generación. Del orden de US$ 0,0002 con las tarifas del proyecto.

Requisitos
----------
``GEMINI_API_KEY`` y ``DEEPSEEK_API_KEY`` en el ``.env``, y el corpus en ``corpus/``.

Uso
---
    ./.venv/Scripts/python.exe ensayo_gate2.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

import indexar_corpus as ic
from grafo_supervision import construir_grafo, esta_pendiente, resolver, solicitar_aprobacion
from logging_rag import configurar_logging
from recuperacion import construir_modelo
from registro_prompts import PROMPT_ACTIVO, huella_prompt
from servicio_rag import ServicioRAG

DIRECTORIO_ENSAYO = Path("vectorstore_ensayo")
PREGUNTA = "¿Qué hace el cliente cuando el proveedor responde 401?"
CORREO_SENUELO = "ana.perez@example.com"
REGISTRO = Path("docs") / "registro_ensayo_gate2.md"


def main() -> int:
    configurar_logging()
    load_dotenv()

    print("== Paso 1: indexar el corpus desde cero ==")
    embeddings = ic.construir_embeddings()
    documentos = ic.cargar_documentos()
    sincronizacion = ic.actualizar_indice(
        embeddings, documentos, directorio=DIRECTORIO_ENSAYO
    )
    vectorstore = ic.abrir_indice(embeddings, DIRECTORIO_ENSAYO)
    resumen_indice = ic.verificar_indice(vectorstore)
    print(json.dumps(sincronizacion, ensure_ascii=False))
    print(f"fragmentos en el índice del ensayo: {resumen_indice['fragmentos']}")

    print("\n== Paso 2 y 3: recuperar, filtrar, citar, generar y registrar ==")
    servicio = ServicioRAG(
        vectorstore=vectorstore,
        modelo=construir_modelo(),
        modelo_cobrado="deepseek-flash",
        k=3,
        huella_prompt=huella_prompt(PROMPT_ACTIVO),
    )
    # La pregunta del ensayo incluye un dato sensible a propósito: sirve para
    # comprobar que la traza no lo guarda.
    pregunta = f"{PREGUNTA} Mi correo es {CORREO_SENUELO}"
    resultado = servicio.responder(pregunta)
    print(json.dumps(resultado, indent=2, ensure_ascii=False))

    print("\n== Paso 4: punto de supervisión humana ==")
    app = construir_grafo()
    config, pausado = solicitar_aprobacion(app, huella_prompt(PROMPT_ACTIVO))
    print("pausado, pendiente:", app.get_state(config).interrupts[0].value)
    final = resolver(app, config, True)
    print("tras aprobar:", final)

    print("\n== Verificación de los criterios no compensables ==")
    texto_traza = json.dumps(resultado, ensure_ascii=False)
    criterios = {
        "pipeline_corre_de_punta_a_punta": bool(resultado["respuesta"].strip()),
        "respuestas_citan_fuente": bool(resultado["fuentes"])
        and any(fuente in resultado["respuesta"] for fuente in resultado["fuentes"]),
        "logs_sin_contenido_crudo": CORREO_SENUELO not in texto_traza
        and pregunta not in texto_traza,
        "supervision_humana_funciona": final.get("aplicado") is True
        and "__interrupt__" in pausado,
    }
    for criterio, cumple in criterios.items():
        print(f"  {'OK ' if cumple else 'FALLA'}  {criterio}")

    if not all(criterios.values()):
        print("\nEl ensayo no está listo para entregar: revisa los criterios en FALLA.")
        return 1

    registro = _formatear_registro(sincronizacion, resumen_indice, resultado, criterios)
    REGISTRO.parent.mkdir(parents=True, exist_ok=True)
    REGISTRO.write_text(registro, encoding="utf-8")
    print(f"\nRegistro guardado en {REGISTRO}")
    print("Ensayo completo: todos los criterios cumplidos.")
    return 0


def _formatear_registro(
    sincronizacion: dict, resumen_indice: dict, resultado: dict, criterios: dict
) -> str:
    tokens = next(
        (e for e in resultado["auditoria"] if e["evento"] == "consumo_llm"), {}
    )
    lineas = [
        "# Registro del ensayo funcional del Gate 2",
        "",
        f"- **Fecha:** {date.today()}",
        f"- **Pregunta del ensayo:** incluía un correo señuelo para verificar el criterio de logs",
        f"- **Fragmentos en el índice del ensayo:** {resumen_indice['fragmentos']}",
        f"- **Fuentes citadas:** {', '.join(resultado['fuentes']) or 'ninguna'}",
        f"- **Trace id:** `{resultado['trace_id']}`",
        f"- **Tokens:** entrada {tokens.get('input_tokens')} · salida {tokens.get('output_tokens')}",
        f"- **Huella del prompt:** `{tokens.get('huella_prompt')}`",
        "",
        "## Criterios no compensables",
        "",
        "| Criterio | Cumple |",
        "|---|---|",
    ]
    lineas += [
        f"| {criterio} | {'sí' if cumple else 'NO'} |" for criterio, cumple in criterios.items()
    ]
    lineas += [
        "",
        "## Encontrado y corregido en este ensayo",
        "",
        "- El contexto numerado no decía de qué archivo salía cada fragmento, así que",
        "  el modelo podía numerar citas pero no nombrar la fuente. Se agregó el nombre",
        "  del archivo a cada bloque del contexto (`[1] (archivo.md) texto`).",
        "- `abrir_indice` creaba la colección sin la metadata del modelo de embeddings,",
        "  y `from_documents` ya no la aplica sobre una colección existente. Ahora el",
        "  índice se construye sin abrirlo primero, y una colección vacía se recrea.",
        "",
        "## Respuesta completa del servicio",
        "",
        "```json",
        json.dumps(resultado, indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    return "\n".join(lineas)


if __name__ == "__main__":
    sys.exit(main())
