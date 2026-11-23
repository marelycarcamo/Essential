# Costo por operación

El costo se calcula por componente de token, no con una tarifa única. Un solo
precio por token queda equivocado hasta por un factor de trece, porque el
proveedor cobra distinto cada parte de la petición.

## Componentes de la tarifa

- La entrada y la salida tienen precios distintos.
- La entrada que acierta la caché cuesta cincuenta veces menos que la que no
  acierta.
- Existe hora punta y hora valle, y la valle cuesta la mitad.

Para el modelo `deepseek-flash`, en dólares por un millón de tokens, las tarifas
de valle y punta son: acierto de caché 0,003 y 0,006; fallo de caché 0,15 y
0,30; salida 0,60 y 1,20.

## Ventanas de hora punta

La hora punta se define en UTC, de 01:00 a 04:00 y de 06:00 a 10:00, de lunes a
viernes. El inicio de la ventana entra y el fin no: a las 04:00 exactas ya es
tarifa valle. Los feriados del proveedor no se modelan, y esa limitación está
declarada: modelarlos exigiría un calendario externo y el error es menor
comparado con no medir nada.

## Dónde vive

Las tarifas y la cuenta viven en un módulo propio, separado del cliente. Son
responsabilidades distintas y cambian por motivos distintos: si el proveedor
sube un precio, se toca el módulo de costos y no el cliente.

El acumulado de costo es de la sesión actual y se pierde al reiniciar el
proceso. Es la cuenta de esta ejecución, no un registro histórico.
