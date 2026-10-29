"""Registro de costo por operación (Bloque 13).

Por qué un módulo propio
------------------------
El cliente habla con la API y controla los reintentos; esto son tarifas y una
cuenta. Son responsabilidades distintas y cambian por motivos distintos: si el
proveedor sube un precio, se toca este archivo y no el cliente.

Por qué no hay un solo precio por token
---------------------------------------
El material del bloque usa un único ``COSTO_POR_1000_TOKENS`` porque Gemini
cobra una tarifa simple. DeepSeek no:

- La **entrada** y la **salida** tienen precios distintos (4x en el modelo flash).
- La entrada que **acierta la caché** cuesta 50x menos que la que no.
- Hay **hora punta y hora valle**, y la valle cuesta la mitad.

Con un solo número, el costo registrado puede quedar equivocado por un factor
de 13 según cómo se componga la petición. Por eso se calcula por componente.

Tarifas oficiales (USD por 1M de tokens), consultadas el 2026-10-29 en
https://api-docs.deepseek.com/quick_start/pricing
"""

import logging
import os
from datetime import datetime, time, timezone

logger = logging.getLogger(__name__)

MODELO_POR_DEFECTO = "deepseek-flash"

# USD por 1M de tokens. Cada tarifa es (valle, punta); el índice se elige según
# la hora de la llamada. Se declara aquí y no en el cliente porque es un dato
# del proveedor, no del flujo de la aplicación.
PRECIOS = {
    "deepseek-flash": {
        "entrada_cache_hit": (0.003, 0.006),
        "entrada_cache_miss": (0.15, 0.30),
        "salida": (0.60, 1.20),
    },
    "deepseek-v4-pro": {
        "entrada_cache_hit": (0.022, 0.044),
        "entrada_cache_miss": (0.66, 1.32),
        "salida": (1.98, 3.96),
    },
}

# Hora punta en UTC: 01:00-04:00 y 06:00-10:00, de lunes a viernes.
# Los feriados chinos también son valle, pero no se modelan: exigiría un
# calendario externo, y para una aplicación con usuarios reales el error es
# menor comparado con no medir nada.
VENTANAS_PUNTA_UTC = ((time(1, 0), time(4, 0)), (time(6, 0), time(10, 0)))
DIAS_DE_SEMANA_PUNTA = range(0, 5)  # datetime.weekday(): lunes=0 ... viernes=4

# Acumulado de la sesión actual. Se pierde al reiniciar el proceso, y eso es lo
# esperado: no es un registro histórico, es la cuenta de esta ejecución.
_costo_total_sesion = 0.0


def es_hora_punta(momento: datetime | None = None) -> bool:
    """Indica si ``momento`` cae en una ventana de precio alto.

    Args:
        momento: instante a evaluar. Si es ``None`` se usa la hora actual en
            UTC. Un ``datetime`` sin zona horaria se interpreta como UTC.

    Returns:
        ``True`` si la llamada se cobra a tarifa de punta.
    """

    if momento is None:
        momento = datetime.now(timezone.utc)

    if momento.weekday() not in DIAS_DE_SEMANA_PUNTA:
        return False

    hora = momento.timetz().replace(tzinfo=None)

    # El inicio de la ventana entra y el fin no: a las 04:00 ya es valle.
    return any(inicio <= hora < fin for inicio, fin in VENTANAS_PUNTA_UTC)


def calcular_costo(
    datos: dict,
    modelo: str | None = None,
    momento: datetime | None = None,
) -> float:
    """Calcula el costo en USD de una respuesta del proveedor.

    Lee los tokens reales del bloque ``usage`` de la respuesta: no estima por
    longitud del texto.

    Args:
        datos: respuesta completa del proveedor, ya convertida a diccionario.
        modelo: nombre del modelo cobrado. Si es ``None`` se toma de
            ``DEEPSEEK_MODEL``.
        momento: instante de la llamada, para elegir valle o punta. Si es
            ``None`` se usa la hora actual en UTC.

    Returns:
        Costo estimado en USD.

    Raises:
        ValueError: si el modelo no tiene tarifas registradas. Es preferible
            fallar a registrar un monto inventado.
    """

    modelo = modelo or os.getenv("DEEPSEEK_MODEL", MODELO_POR_DEFECTO)
    tarifas = PRECIOS.get(modelo)

    if tarifas is None:
        raise ValueError(
            f"No hay tarifas registradas para el modelo '{modelo}'. "
            "Agrégalas en costos.PRECIOS con precios oficiales."
        )

    uso = datos.get("usage") or {}

    total_entrada = uso.get("prompt_tokens", 0)
    cache_hit = uso.get("prompt_cache_hit_tokens", 0)

    # Si el proveedor no informa el desglose, todo lo no cacheado se cobra como
    # cache miss, que es la tarifa más cara: nunca subestimar el costo.
    cache_miss = uso.get("prompt_cache_miss_tokens")
    if cache_miss is None:
        cache_miss = max(total_entrada - cache_hit, 0)

    salida = uso.get("completion_tokens", 0)

    tarifa = 1 if es_hora_punta(momento) else 0

    total = (
        cache_hit * tarifas["entrada_cache_hit"][tarifa]
        + cache_miss * tarifas["entrada_cache_miss"][tarifa]
        + salida * tarifas["salida"][tarifa]
    )

    return total / 1_000_000


def registrar_costo(
    datos: dict,
    modelo: str | None = None,
    momento: datetime | None = None,
) -> float:
    """Calcula el costo de una operación y lo suma al total de la sesión.

    Se llama **solo cuando la respuesta llegó con éxito**: una petición que
    agotó sus reintentos nunca consumió tokens.

    Si el modelo no tiene tarifas, se registra un aviso y se devuelve ``0.0``
    en vez de levantar el error: la operación ya se hizo y ya se pagó, así que
    perder la respuesta por un problema de contabilidad sería peor. El aviso
    queda en el log para que la omisión sea visible.

    Args:
        datos: respuesta completa del proveedor.
        modelo: nombre del modelo cobrado. Si es ``None`` se toma del entorno.
        momento: instante de la llamada. Si es ``None`` se usa la hora actual.

    Returns:
        Costo de esta operación en USD, o ``0.0`` si no se pudo determinar.
    """

    global _costo_total_sesion

    try:
        costo = calcular_costo(datos, modelo, momento)
    except ValueError as error:
        logger.warning("Operación sin costo registrado: %s", error)
        return 0.0

    _costo_total_sesion += costo
    logger.info(
        "Costo de la operación: $%.8f (acumulado: $%.8f)", costo, _costo_total_sesion
    )

    return costo


def costo_acumulado() -> float:
    """Devuelve el costo acumulado de la sesión en USD."""

    return _costo_total_sesion


def reiniciar_costo() -> None:
    """Reinicia el acumulado. Pensado para tests y para separar sesiones."""

    global _costo_total_sesion

    _costo_total_sesion = 0.0
