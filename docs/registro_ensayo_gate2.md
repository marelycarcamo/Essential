# Registro del ensayo funcional del Gate 2

- **Fecha:** 2026-11-23
- **Pregunta del ensayo:** incluía un correo señuelo para verificar el criterio de logs
- **Fragmentos en el índice del ensayo:** 31
- **Fuentes citadas:** 01_cliente_modular.md
- **Trace id:** `trace-e053bcbd`
- **Tokens:** entrada 543 · salida 379
- **Huella del prompt:** `55689be7cecc`

## Criterios no compensables

| Criterio | Cumple |
|---|---|
| pipeline_corre_de_punta_a_punta | sí |
| respuestas_citan_fuente | sí |
| logs_sin_contenido_crudo | sí |
| supervision_humana_funciona | sí |

## Encontrado y corregido en este ensayo

- El contexto numerado no decía de qué archivo salía cada fragmento, así que
  el modelo podía numerar citas pero no nombrar la fuente. Se agregó el nombre
  del archivo a cada bloque del contexto (`[1] (archivo.md) texto`).
- `abrir_indice` creaba la colección sin la metadata del modelo de embeddings,
  y `from_documents` ya no la aplica sobre una colección existente. Ahora el
  índice se construye sin abrirlo primero, y una colección vacía se recrea.

## Respuesta completa del servicio

```json
{
  "trace_id": "trace-e053bcbd",
  "respuesta": "Ante un HTTP 401, el cliente clasifica el error como `ErrorAutenticacion`: la credencial no sirve y esperar no la arregla, así que **no se reintenta** [1] (01_cliente_modular.md).",
  "fuentes": [
    "01_cliente_modular.md"
  ],
  "auditoria": [
    {
      "evento": "consulta_recibida",
      "trace_id": "trace-e053bcbd",
      "pregunta_len": 89,
      "pregunta_huella": "bd977039964a",
      "k": 3
    },
    {
      "evento": "documentos_recuperados",
      "trace_id": "trace-e053bcbd",
      "n": 3,
      "k": 3
    },
    {
      "evento": "contexto_filtrado",
      "trace_id": "trace-e053bcbd",
      "n_recuperados": 3,
      "n_limpios": 3,
      "n_descartados": 0
    },
    {
      "evento": "consumo_llm",
      "trace_id": "trace-e053bcbd",
      "input_tokens": 543,
      "output_tokens": 379,
      "total_tokens": 922,
      "huella_prompt": "55689be7cecc"
    }
  ]
}
```
