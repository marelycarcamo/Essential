# Agente con herramientas y memoria por sesión

El agente razona, elige una herramienta, observa el resultado y repite hasta
responder. La diferencia con una cadena fija es que la cadena siempre sigue el
mismo camino, mientras que el agente decide en cada paso qué herramienta usar
según la descripción que cada una declara.

## Herramientas de lectura

`buscar_en_documentacion` recorre la documentación del proyecto y devuelve los
fragmentos con más coincidencias. Corta por estructura, es decir por párrafo,
título o fila de tabla, en vez de por un número fijo de caracteres, ignora las
imágenes embebidas y compara por inicio de palabra para no confundir `gemini`
con `mini`.

`estimar_costo` reutiliza las tarifas del módulo de costos y devuelve el costo
estimado de una llamada a partir del modelo, los tokens de entrada y los tokens
de salida.

## Herramienta de escritura

`guardar_material` es la única herramienta que escribe en disco. Recibe los
argumentos con un esquema validado campo por campo, exige aprobación humana por
argumento antes de escribir y verifica el contrato del material antes de
guardarlo. El riesgo de esta herramienta no es la escritura, que es reversible y
está fuera del control de versiones: es el contenido. Un material didáctico
incorrecto es peor que ningún material.

## Memoria

La memoria es por sesión. El historial se guarda en memoria del proceso y se
recupera por identificador de sesión, así que el segundo turno entiende
referencias como "ese análisis" y una sesión distinta no ve la conversación de
otra.
