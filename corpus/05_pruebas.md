# Pruebas automatizadas

La suite del proyecto se corre con pytest desde la raíz, con el entorno virtual
activo, y completa más de 260 pruebas en menos de cinco segundos. La velocidad
no es un adorno: permite correr la suite antes de cada commit sin pensarlo dos
veces.

## Qué se prueba y qué no

Las pruebas de la suite no llaman a la red. Los proveedores se reemplazan por
dobles que respetan la convención de llamada, y las esperas del backoff se
interceptan para que un test de reintentos no tarde veinte segundos. La prueba
que sí golpea un proveedor real queda excluida por defecto y se corre a mano.

Un doble de test valida la interfaz, no el backend. Por eso existe una regla: la
integración con un tercero se corre al menos una vez contra el servicio real,
porque los dobles permisivos dejan pasar regresiones que después aparecen en
producción.

## Casos borde

Las pruebas cubren el camino feliz y además los casos que suelen olvidarse:
entrada vacía, entrada duplicada, un tipo equivocado donde se esperaba texto, un
error de autenticación que no debe reintentarse y un límite de uso que sí debe
reintentarse. Un test por comportamiento, con un nombre que describe la
condición, para que el test sirva de documentación.

## Trazabilidad

Cada requisito del checklist del proyecto apunta a la prueba que lo cubre. Esa
tabla vive en el README, y su función es concreta: cuando alguien pregunte cómo
se sabe que algo funciona, la respuesta no es "porque lo probé", es el nombre
del test.
