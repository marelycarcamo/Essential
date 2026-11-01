# Límites de la métrica de evaluación

## Qué mide

Coincidencia de palabras entre una respuesta generada y una referencia escrita a
mano, contando repeticiones. Se reporta como precisión (cuánto de la respuesta
está en la referencia), cobertura (cuánto de la referencia aparece en la
respuesta) y F1, que combina las dos.

Mide **parecido léxico**. No mide si la respuesta es correcta.

## Modos de falla conocidos

Cada uno con el caso concreto donde apareció.

- **Paráfrasis correcta castigada.** "¿Qué es un bucle for?" respondió
  "estructura de control que repite un bloque de instrucciones un número
  determinado de veces, normalmente mediante un contador o recorriendo una
  secuencia". Es correcta, completa, y sacó **F1 0.33** solo por extensión.

- **Respuesta equivocada castigada por casualidad.** "¿Qué significa la sigla
  PEP?" respondió con el significado del ámbito financiero ("Persona Expuesta
  Políticamente"), no con "Python Enhancement Proposal". Sacó **F1 0.13**, el
  más bajo de la corrida. Coincidió el número con el error, pero por el motivo
  equivocado: la métrica reaccionó a la falta de palabras en común, no al
  contenido incorrecto.

- **El código en la respuesta diluye la precisión.** "¿Cómo se abre un archivo
  en Python?" respondió con un ejemplo de `with open(...)`. Es correcta y más
  completa que la referencia, pero los tokens de código (`open`, `with`, `r`)
  no existen en la referencia y bajaron el **F1 a 0.35** sin que haya error.

- **Un F1 perfecto mide acuerdo con la referencia, no verdad.** "¿En qué año se
  publicó la primera versión de Python?" respondió `1991` y la referencia decía
  `1991`: **F1 1.00**. Si la referencia hubiera estado mal escrita, una
  respuesta errónea habría sacado lo mismo. La métrica no tiene acceso a la
  verdad: solo compara dos textos que escribió una persona.

- **Sinónimos.** "El auto es veloz" frente a "El vehículo es rápido":
  **F1 0.50**. Dice lo mismo con otras palabras.

- **Datos equivocados.** "Python fue creado en 1991" frente a "Python fue creado
  en 1989": **F1 0.80**. Cambia un hecho y el número apenas se mueve.

- **Negaciones ausentes.** "Python no es un lenguaje compilado" frente a "Python
  es un lenguaje compilado": **F1 0.91**, con significados opuestos.

## Muestra revisada

**10 casos de 20 respuestas**, generadas el **2026-11-01** con `deepseek-flash`:
los 5 con F1 más bajo (0.13 a 0.36) y los 5 con F1 más alto (0.73 a 1.00). Se
excluyeron los del medio a propósito, para cubrir los dos extremos.

La revisión completa, con la rúbrica aplicada caso por caso, está en
`revision_manual.py` y en `docs/revision_manual.json`.

En esa muestra, 4 de los 5 casos con F1 más bajo **no tenían ningún error**: el
número estaba midiendo extensión, no calidad.

## Recomendación de uso

**Confiar en el número solo para lo que sirve:** comparar en volumen entre dos
versiones del cliente o del prompt, y decidir qué casos leer primero.

**No usarlo como criterio de aprobación.** En esta corrida, el promedio fue
precisión **0.43** y cobertura **0.83**: el cliente dice casi todo lo que se
espera, con muchas palabras de más. Cuando la referencia es corta y específica
—como las de este proyecto—, **mirar la precisión antes que el F1**.

**Revisar siempre a mano:**

- cualquier respuesta con fechas, cifras o nombres propios, sin importar el F1;
- cualquier F1 alto sobre una respuesta larga, porque puede ser acuerdo con una
  referencia mal escrita;
- cualquier respuesta que incluya código, porque sus tokens diluyen la
  precisión de forma artificial.

**Revisar también los F1 altos, no solo los bajos.** Es donde se esconden los
falsos positivos, y son los que generan confianza indebida.
