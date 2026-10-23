"""Banco de prompts para analizar encuestas de satisfacción."""

BANCO_DE_PROMPTS = [
    {
        "nombre": "clasificar_comentario",
        "system_instruction": """
Eres un asistente especializado en analizar encuestas de satisfacción
de cursos de capacitación.

Clasifica los aspectos mencionados en el comentario de un alumno.

Para cada aspecto, identifica:
- dimension: la dimensión de la capacitación evaluada.
- valoracion: la valoración expresada por el alumno.

Dimensiones permitidas:
facilitacion, contenidos, materiales, infraestructura,
coordinacion, tiempo_practico, alimentacion, general.

Valoraciones permitidas:
positiva, negativa, mixta, neutral.

Reglas:
- Un comentario puede mencionar varias dimensiones.
- Separa los aspectos positivos y negativos cuando corresponda.
- No inventes información.
- Si el comentario es ambiguo, utiliza general y neutral.
- Devuelve únicamente un objeto JSON válido.
""",
        "ejemplo_usuario": (
            "Me gustaron los contenidos, pero faltó tiempo "
            "para realizar los ejercicios."
        ),
        "claves_esperadas": ["comentarios"],
        "response_schema": {
            "type": "object",
            "properties": {
                "comentarios": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "dimension": {
                                "type": "string",
                                "enum": [
                                    "facilitacion",
                                    "contenidos",
                                    "materiales",
                                    "infraestructura",
                                    "coordinacion",
                                    "tiempo_practico",
                                    "alimentacion",
                                    "general",
                                ],
                            },
                            "valoracion": {
                                "type": "string",
                                "enum": [
                                    "positiva",
                                    "negativa",
                                    "mixta",
                                    "neutral",
                                ],
                            },
                        },
                        "required": ["dimension", "valoracion"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["comentarios"],
            "additionalProperties": False,
        },
    },
    {
        "nombre": "identificar_oportunidades",
        "system_instruction": """
Eres un analista de encuestas de satisfacción de capacitaciones.

Examina los comentarios proporcionados e identifica:
- Fortalezas mencionadas por los alumnos.
- Oportunidades de mejora mencionadas por los alumnos.

Agrupa comentarios que expresen una misma idea.
No inventes opiniones ni afirmes que algo es recurrente si los
datos no permiten comprobarlo.
No calcules porcentajes ni promedios.
Devuelve únicamente un objeto JSON válido con las claves
fortalezas y oportunidades, ambas listas de textos.
""",
        "ejemplo_usuario": """
Comentarios de alumnos:
1. La facilitadora explicó claramente los contenidos.
2. Me habría gustado disponer de más tiempo para practicar.
3. Las explicaciones fueron claras, pero faltaron ejercicios.
""",
        "claves_esperadas": ["fortalezas", "oportunidades"],
        "response_schema": {
            "type": "object",
            "properties": {
                "fortalezas": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "oportunidades": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["fortalezas", "oportunidades"],
            "additionalProperties": False,
        },
    },
    {
        "nombre": "generar_recomendaciones",
        "system_instruction": """
Eres un asesor especializado en mejora continua de capacitaciones.

A partir de las oportunidades de mejora proporcionadas,
propón acciones concretas, realistas y aplicables por una OTEC.

Para cada recomendación, identifica:
- dimension: la dimensión relacionada con la mejora.
- accion: la acción concreta que se propone.
- prioridad: alta, media o baja.

No inventes problemas que no estén respaldados por la entrada.
Evita recomendaciones vagas como mejorar todo o prestar más atención.
Devuelve únicamente un objeto JSON válido con la clave recomendaciones.
""",
        "ejemplo_usuario": """
Oportunidades de mejora detectadas:
1. Disponer de más tiempo para realizar ejercicios prácticos.
2. Incorporar más ejemplos aplicados a situaciones laborales.
""",
        "claves_esperadas": ["recomendaciones"],
        "response_schema": {
            "type": "object",
            "properties": {
                "recomendaciones": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "dimension": {
                                "type": "string",
                                "enum": [
                                    "facilitacion",
                                    "contenidos",
                                    "materiales",
                                    "infraestructura",
                                    "coordinacion",
                                    "tiempo_practico",
                                    "alimentacion",
                                    "general",
                                ],
                            },
                            "accion": {"type": "string"},
                            "prioridad": {
                                "type": "string",
                                "enum": ["alta", "media", "baja"],
                            },
                        },
                        "required": ["dimension", "accion", "prioridad"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["recomendaciones"],
            "additionalProperties": False,
        },
    },
]
