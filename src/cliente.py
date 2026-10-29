"""Cliente modular de consumo de modelos (Bloques 12 y 13).

Un solo punto de entrada — ``consultar_modelo`` — que arma la petición, la
envía con reintento automático, registra el costo de la operación y devuelve el
texto ya extraído.

Quien usa este módulo no necesita saber que existe ``con_reintento``, ni cuántas
veces se reintentó, ni qué código HTTP causó el problema, ni cuánto costó. Esa
es la idea de un cliente modular: esconder la complejidad detrás de una
interfaz simple.

Se construye sobre ``consumir_deepseek``, que es el proveedor elegido en
``docs/decision-arquitectura.md``.

Uso::

    from cliente import consultar_modelo

    try:
        print(consultar_modelo("¿Cuál es la capital de Chile?"))
    except ValueError as error:     # ErrorAutenticacion es un ValueError
        print(f"Error de configuración: {error}")
    except RuntimeError as error:   # ErrorReintentosAgotados es un RuntimeError
        print(f"El servicio no respondió: {error}")
"""

import logging
import random
import time
from typing import Any, Callable

from consumir_deepseek import armar_peticion, enviar_peticion, extraer_texto
from costos import registrar_costo
from errores import ErrorReintentosAgotados, ErrorTemporal

logger = logging.getLogger(__name__)

# Valores por defecto del reintento. Se nombran porque son decisiones, no
# números mágicos. La espera crece en cada intento (backoff exponencial) y tiene
# un techo: sin techo, el octavo reintento esperaría 128 segundos.
INTENTOS_POR_DEFECTO = 3
ESPERA_BASE_POR_DEFECTO = 2.0
ESPERA_MAXIMA_POR_DEFECTO = 30.0

# Aleatoriedad que se suma a cada espera, como fracción de la espera calculada.
# Sirve cuando varios procesos reintentan a la vez: sin jitter esperan todos lo
# mismo y vuelven a chocar contra el límite en el mismo instante.
JITTER_POR_DEFECTO = 0.10


def con_reintento(
    funcion: Callable[[], Any],
    intentos: int = INTENTOS_POR_DEFECTO,
    espera_base: float = ESPERA_BASE_POR_DEFECTO,
    espera_maxima: float = ESPERA_MAXIMA_POR_DEFECTO,
) -> Any:
    """Ejecuta ``funcion`` y la reintenta si falla por un motivo temporal.

    Solo captura ``ErrorTemporal``. Cualquier otro error se propaga tal cual:
    insistir con algo que no es momentáneo no tiene sentido.

    La espera entre intentos se duplica en cada vuelta (``espera_base``, el
    doble, el cuádruple...) hasta un techo de ``espera_maxima``, y se le suma
    un porcentaje aleatorio. Esperar siempre lo mismo no resuelve nada cuando
    el proveedor frenó por exceso de uso: dos segundos después es probable que
    sigas dentro de la misma ventana.

    Args:
        funcion: operación sin argumentos que se quiere proteger. Se pasa como
            ``lambda`` cuando la operación real necesita argumentos.
        intentos: número total de intentos, incluido el primero.
        espera_base: espera del primer reintento. Los siguientes la duplican.
        espera_maxima: techo de la espera, sin importar cuántos intentos lleves.

    Returns:
        Lo que devuelva ``funcion`` en el primer intento exitoso.

    Raises:
        ErrorTemporal: si el último intento permitido también falla. Se
            propaga el error original, no uno nuevo, para no perder el motivo.
    """

    for intento in range(1, intentos + 1):
        try:
            return funcion()
        except ErrorTemporal as error:
            if intento == intentos:
                raise

            espera = min(espera_base * (2 ** (intento - 1)), espera_maxima)
            espera_con_jitter = espera + random.uniform(0, espera * JITTER_POR_DEFECTO)

            logger.warning(
                "Intento %s de %s falló: %s. Reintentando en %.2fs.",
                intento,
                intentos,
                error,
                espera_con_jitter,
            )
            time.sleep(espera_con_jitter)


def consultar_modelo(pregunta: str, system_instruction: str = "") -> str:
    """Devuelve el texto que el modelo responde a ``pregunta``.

    Es el único punto de entrada del cliente: agrupa armado, envío con
    reintento y extracción del texto.

    Args:
        pregunta: texto del usuario. No puede estar vacío.
        system_instruction: contexto permanente para el modelo. Opcional.

    Returns:
        El texto generado por el modelo.

    Raises:
        ErrorAutenticacion: credencial inválida o sin permiso (no se reintenta).
        ErrorReintentosAgotados: el servicio no respondió tras todos los intentos.
    """

    texto, _datos = _consultar_completo(pregunta, system_instruction)
    return texto


def _consultar_completo(
    pregunta: str,
    system_instruction: str = "",
) -> tuple[str, dict]:
    """Igual que ``consultar_modelo``, pero además devuelve la respuesta cruda.

    Se separa para que quien necesite los tokens —el registro de costo, o una
    futura métrica— no tenga que cambiar la firma pública de
    ``consultar_modelo``.
    """

    url, headers, body = armar_peticion(pregunta, system_instruction)

    try:
        datos = con_reintento(lambda: enviar_peticion(url, headers, body))
    except ErrorTemporal as error:
        # Traducimos un detalle interno ("se agotaron los reintentos") a un
        # error con mensaje para quien consume el cliente, no para quien lo
        # escribió. `from error` conserva la causa original en el traceback.
        raise ErrorReintentosAgotados(
            "El servicio no respondió tras varios intentos. Intenta más tarde."
        ) from error

    # El costo se registra solo acá: si se agotaron los reintentos, la petición
    # nunca llegó a consumir tokens.
    registrar_costo(datos)

    return extraer_texto(datos), datos
