# Cliente modular de modelos

El cliente modular es la pieza que habla con el proveedor de modelos. Expone una
sola función pública, `consultar_modelo()`, y deja el reintento en una función
interna, `con_reintento()`. Quien consume el cliente no elige cuándo reintentar:
el cliente ya decidió por él.

## Clasificación de errores

Los errores se clasifican por lo que hay que hacer con ellos, no por su código
HTTP. `ErrorAutenticacion` cubre los HTTP 401 y 403: la credencial no sirve y
esperar no la arregla, así que no se reintenta. `ErrorTemporal` cubre los HTTP
5xx, los timeouts y los errores de conexión: la falla se resuelve sola, así que
sí se reintenta. `ErrorLimiteDeUso` es una subclase de `ErrorTemporal` para el
HTTP 429: se reintenta igual, pero significa que hay que bajar el ritmo, y
reintentar con la misma cadencia vuelve a chocar contra el mismo límite.
`ErrorReintentosAgotados` es la traducción final, pensada para quien consume el
cliente.

## Backoff exponencial

La espera entre intentos crece como `espera_base * 2**(intento-1)`, con un techo
de 30 segundos y un jitter del 10 por ciento para que varias peticiones
simultáneas no vuelvan a sincronizarse contra el mismo límite.

## Espera frente a reintento

La distinción no es cosmética. Un 429 obliga a bajar el ritmo; un 5xx significa
que el servidor está mal y el ritmo no cambia nada. Mezclar ambos casos en un
mismo mensaje de error deja el diagnóstico sin datos: por eso el mensaje de
error siempre incluye el código HTTP.
