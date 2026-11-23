# Búsqueda documental y pipeline de texto

## Búsqueda actual

La búsqueda documental actual es por palabra clave. Recorre los archivos de
documentación del proyecto, los corta por estructura, puntúa cada fragmento
contando coincidencias y devuelve los mejores. Funciona sin costo y sin red,
pero tiene un límite conocido: si el usuario pregunta con otras palabras que las
del documento, no encuentra nada. Buscar "demora" no encuentra un párrafo que
dice "latencia".

Los tres fallos que aparecieron al construirla fueron reveladores: la herramienta
devolvía el texto alternativo de una imagen del README, trataba una tabla de
cuatro mil quinientos caracteres como si fuera un párrafo, y daba falsos
positivos de subcadena. Los tres se arreglaron cortando por estructura, filtrando
imágenes y comparando por inicio de palabra.

## Pipeline de texto

El pipeline limpia, tokeniza y vectoriza. La limpieza normaliza mayúsculas y
acentos, y una decisión merece explicación: la puntuación se borra, no se
reemplaza por un espacio. Unir "hola,mundo" en "holamundo" es intencional y
tiene su propia prueba.

El límite del pipeline está medido: sobre tres respuestas reales, el parecido con
la referencia subió de 85,7 a 100 por ciento, de 80 a 88,9 por ciento, y en el
tercer caso no se movió, de 36,4 a 36,4 por ciento. Ese tercer caso es la mejor
evidencia de que el pipeline arregla formato y no significado.

## Métricas de texto

Las métricas son precisión, cobertura y F1, calculadas sobre la intersección de
tokens con un contador que toma el mínimo de cada uno. Sobre veinte respuestas
reales el promedio fue F1 0,55, precisión 0,43 y cobertura 0,83: la brecha dice
que el modelo responde completo pero extenso.

Una advertencia que salió de revisar la muestra a mano: tres de los cinco casos
con F1 más bajo eran respuestas correctas y completas. La métrica medía
extensión, no calidad.
