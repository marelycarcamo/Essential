# Banco de prompts para encuestas de satisfacción

El proyecto analiza comentarios de encuestas de satisfacción de cursos de
capacitación mediante un banco de tres prompts especializados. Cada prompt tiene
una instrucción propia y un esquema de respuesta fijo, para que el resultado se
pueda procesar sin revisar el texto a mano.

## Prompt de clasificación

Identifica las dimensiones de la capacitación mencionadas por los alumnos y
clasifica cada valoración. Las dimensiones contempladas son facilitación,
contenidos, materiales, infraestructura, coordinación, tiempo práctico,
alimentación y valoración general. Las valoraciones posibles son positiva,
negativa, mixta y neutral.

## Prompt de oportunidades

Analiza los comentarios para identificar fortalezas de la capacitación y
oportunidades de mejora. Agrupa los comentarios que expresan ideas similares y
no debe inventar opiniones ni atribuir recurrencia sin evidencia suficiente.

## Prompt de recomendaciones

Propone acciones concretas y aplicables para abordar las oportunidades
detectadas. Cada recomendación incluye una dimensión, una acción y un nivel de
prioridad: alta, media o baja.

## Para qué sirve

El banco apunta a una Organización Técnica de Capacitación, que necesita leer
cien comentarios de un curso y sacar conclusiones que se puedan sostener. El
criterio de éxito no es que el modelo escriba bonito: es que la clasificación se
pueda auditar contra el comentario original.
