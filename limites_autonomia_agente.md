# Límites de autonomía del agente

Este documento fija qué puede hacer el agente de este proyecto sin intervención
humana, qué no, y cómo se detecta que se está comportando fuera de lo previsto.

El agente vive en `agente_con_herramientas.py` y tiene tres herramientas. La
**autonomía** es el margen que le damos: qué decide el modelo —cuándo usar una
herramienta, cuál, con qué argumentos— y qué decidimos nosotros por él. Todo lo
que el modelo elige se puede elegir mal, y el costo de elegir mal no es igual en
las tres.

---

## 1. Cada herramienta, clasificada por riesgo

| Herramienta | Qué hace | Riesgo | Peor caso si el agente la elige mal |
| :---- | :---- | :---- | :---- |
| `buscar_en_documentacion` | Busca fragmentos en `docs/*.md` y el README | 🟢 **Solo lectura** | Devuelve fragmentos que no responden y la respuesta sale inútil. No modifica nada. |
| `estimar_costo` | Calcula en USD lo que costaría una llamada, con las tarifas de `src/costos.py` | 🟢 **Solo lectura** | Una cifra equivocada sobre una tarifa mal pasada. No gasta dinero: no llama a ningún proveedor, hace aritmética con tarifas fijas. |
| `guardar_material` | Escribe un material `.md` en `material_generado/`, validando el contrato | 🟡 **Escritura** | Sobrescribe un material anterior aprobado, o deja en disco material con errores que después alguien usa para capacitar. |

**Por qué `guardar_material` no es riesgo alto.** El criterio del bloque es
escritura irreversible o con impacto económico. Aquí no hay ninguna de las dos:
el archivo vive en una carpeta que no se versiona (está en `.gitignore`) y se
puede sobrescribir o borrar sin costo. Tampoco envía nada a terceros ni gasta
cuota. Lo que sí puede fallar no es la acción sino el **contenido**: un material
didáctico incorrecto tiene costo de reputación, no económico, y se paga cuando
alguien lo enseña.

**No hay ninguna herramienta de riesgo alto hoy, y eso es una decisión.** No se
agregó una herramienta que publique material en `docs/`, haga `git push` o envíe
correos. Si se agrega, sube a 🔴 y hereda lo que dice la sección 2 sobre
aprobación externa, no el mecanismo que usamos ahora.

**Un límite que no es de ninguna herramienta: las vueltas.** El ejecutor está
construido con `max_iterations=6`. Sin tope, un agente que se equivoca de
herramienta puede iterar indefinidamente y cada vuelta cuesta tokens. Al
llegar al tope, el ejecutor devuelve `"Agent stopped due to max iterations."` y
eso se lee como una falla, no como una respuesta.

---

## 2. Cuáles requieren aprobación humana, y por qué

| Herramienta | ¿Aprobación? | Fundamento |
| :---- | :---- | :---- |
| `buscar_en_documentacion` | **No** | Leer no cambia nada. Pedir permiso para buscar agrega un clic que no previene ningún daño. |
| `estimar_costo` | **No** | Es aritmética con tarifas registradas. El peor caso es una cifra mal explicada, y eso se corrige respondiendo de nuevo. |
| `guardar_material` | **Sí, obligatoria** | Es la única que escribe. La aprobación no está por el riesgo de escribir, sino porque lo que se escribe va a usarse para enseñar. |

**Cómo se implementa, en el código y no en un acuerdo verbal.**
`guardar_material` recibe `aprobacion_humana: bool = False` en su `args_schema`.
Si llega `False`, no toca el disco y devuelve el motivo:

> "No guardé nada: falta la aprobación humana. Muéstrale el material a la
> persona, pídele que lo revise y, cuando confirme, vuelve a llamarme con
> aprobacion_humana=True."

Además, antes de escribir, verifica el contrato del material: las tres
secciones (`Objetivo:`, `Conceptos:`, `Ejercicio:`), que cada cita exista
literalmente en la documentación del proyecto y que no exceda las 200 palabras
propias. Un material que no cumple no se guarda aunque esté aprobado.

**Por qué el argumento y no un `input()` en la consola.** `input()` bloquea el
proceso hasta que alguien escriba, no se puede probar de forma automática y
deja la aprobación fuera de la traza. Con un argumento, la decisión queda
registrada en el `intermediate_steps` de la llamada: se puede auditar quién
aprobó qué y cuándo.

**La limitación, dicha sin adornos.** Un modelo puede pasar
`aprobacion_humana=True` por su cuenta: el argumento es una convención entre el
agente y nosotros, no un control criptográfico. Lo que lo hace defendible es
que la descripción de la herramienta lo prohíbe de forma explícita, que la
traza muestra el valor usado, y que un `True` sin revisión previa es una señal
de alarma detectable al revisar la traza (sección 3). Para una acción
irreversible de verdad haría falta otra cosa: aprobación por un canal separado
del modelo, con tiempo límite y con "no aprobado" como valor por defecto ante
la duda.

---

## 3. Qué se revisa en la traza, y con qué frecuencia

La traza se obtiene con `return_intermediate_steps=True`: cada llamada deja la
herramienta elegida, sus argumentos y lo que devolvió. Se revisa buscando estos
patrones:

| Patrón | Qué puede significar | Ejemplo real de este proyecto |
| :---- | :---- | :---- |
| La misma herramienta, varias veces seguidas, con argumentos parecidos | El agente no avanza, o una observación ambigua lo hace reintentar | En la primera corrida del Bloque 21 el agente buscó **tres veces** la misma pregunta. La causa no era el agente: la herramienta le devolvía el texto alternativo de una imagen y una tabla de 4.500 caracteres. Se corrigió la herramienta. |
| `guardar_material` invocada sin aprobación en la traza | No debería poder ocurrir: la herramienta devuelve el motivo y no escribe | Si aparece un archivo guardado sin `aprobacion_humana=True`, hay un error en el flujo, no en el modelo. |
| `guardar_material` con `aprobacion_humana=True` y sin revisión humana previa | El modelo se autoaprobó | Es la señal más importante de vigilar, por lo dicho en la sección 2. |
| Una respuesta que afirma algo que ninguna observación devolvió | Alucinación: respondió de memoria en vez de la herramienta | El prompt exige citar `archivo:línea`, y `verificar_material` comprueba que la cita exista en la documentación. |
| Herramientas usadas para preguntas que no las necesitan | Fricción y costo sin beneficio | La pregunta "¿qué puedes hacer?" se respondió sin herramientas, como se esperaba. |
| Vueltas hasta el tope de 6 | El agente no supo cerrar | Se revisa qué herramienta repitió y qué observación estaba recibiendo. |

**Frecuencia.** Mientras el agente está en desarrollo y lo usa una sola
persona, se revisan **todas** las corridas reales: son pocas, cuestan segundos y
cada una enseña algo. Cuando haya varias personas usándolo, la cadencia pasa a
ser: **una muestra de 10 trazas por semana**, elegidas entre las más largas
(las que más decisiones tomaron) y las que terminaron con error, más el **100 %
de las trazas que incluyan `guardar_material`**. Esa última no se muestrea: una
escritura aprobada se revisa siempre, porque es el único punto del sistema donde
algo queda en disco.

---

## 4. Qué hacer si el agente se comporta de forma inesperada

El orden importa: primero se mira lo que la herramienta le devolvió al modelo,
después la descripción de la herramienta, y solo al final el prompt. Cambiar el
prompt a ciegas cuando el problema está en lo que recuperó es el error clásico.

| Síntoma | Primera acción | Después |
| :---- | :---- | :---- |
| Elige la herramienta equivocada | Leer su descripción en voz alta: el modelo elige por esa descripción, no por el código | Añadir a la descripción cuándo **no** usarla, con un ejemplo |
| Repite la misma llamada con los mismos argumentos | Mirar la observación que recibió | Si es ambigua o vacía, mejorar el mensaje de la herramienta; no tocar el prompt |
| Intenta guardar sin aprobación | Confirmar en la traza que la herramienta respondió "falta la aprobación" | Reforzar la regla en el prompt; si insiste, desactivar la herramienta |
| Guarda material que no cumple el contrato | No debería poder: la herramienta lo rechaza antes de escribir | Revisar el prompt de redacción, no la herramienta |
| Afirma algo que no está en la documentación | Revisar primero el *retrieval*: qué fragmentos recibió | Después el prompt; nunca al revés |
| Se queda sin vueltas | Ver qué herramienta repitió y qué le devolvió | Corregir la herramienta o bajar el alcance de la pregunta |
| Tarda demasiado o no responde | El modelo tiene 60 s por intento y 0 reintentos internos: corta rápido y se ve el error | Si es un proveedor inestable, se cambia de proveedor, no se alarga el tiempo |

**Cómo se apaga una herramienta.** `HERRAMIENTAS` es una lista en
`agente_con_herramientas.py`. Quitar `guardar_material` de esa lista desactiva
la escritura en una línea, sin tocar el resto del agente: el modelo deja de
tenerla disponible y el prompt sigue siendo válido. Es la primera medida ante
cualquier comportamiento inesperado de la herramienta de escritura, antes de
investigar.

**Qué no hace falta apagar.** El agente no publica, no envía, no paga y no
versiona. Cortar el proceso corta todo: no hay nada corriendo del lado del
proveedor que siga actuando después de la respuesta.

---

## Resumen operativo

- **Tres herramientas, una sola escribe.** La aprobación humana está en el
  código (`aprobacion_humana`), no en un acuerdo.
- **Tope de 6 vueltas** como límite de autonomía y de gasto.
- **El contrato del material se verifica antes de escribir**, no después.
- **Un material guardado se revisa siempre**; el resto de las trazas, por
  muestra.
- **La señal de alarma principal es un `aprobacion_humana=True` sin revisión
  previa.** Es lo único que el argumento no puede impedir por sí solo.
