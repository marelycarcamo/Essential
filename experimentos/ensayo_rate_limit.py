"""Ensayo de rate limit y backoff (Bloque 13).

Qué mide
--------
Provoca un ``429`` de verdad y muestra, en vivo, que el backoff exponencial con
techo y jitter recupera la operación sin que quien llama se entere.

Por qué contra Gemini y no contra DeepSeek
------------------------------------------
DeepSeek **no limita por minuto**: permite hasta 2500 peticiones simultáneas y
solo devuelve ``429`` al pasar ese número (documentación de "Rate Limit &
Isolation"). Provocarlo exigiría miles de conexiones abiertas a la vez.

Gemini, en cambio, sí limita por minuto en su nivel gratuito. Es el proveedor de
respaldo y usa la misma clasificación de errores que el principal, así que sirve
para verificar el mecanismo. En el registro del bloque queda escrito el desvío y
su motivo.

Cómo se corre
-------------
    .\\.venv\\Scripts\\python.exe experimentos\\ensayo_rate_limit.py

No gasta dinero: consume cuota gratuita de Gemini. Sí requiere ``GEMINI_API_KEY``
y ``GEMINI_MODEL`` en el ``.env``.
"""

import logging
import sys
import time
from pathlib import Path

# Los módulos del proyecto viven en ``src/`` y se importan "pelados", igual que
# en ``main.py`` y en los tests.
SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import consumir_gemini as gemini  # noqa: E402
from cliente import con_reintento  # noqa: E402
from errores import (  # noqa: E402
    ErrorAutenticacion,
    ErrorReintentosAgotados,
)

# Cuántas peticiones se lanzan seguidas. Con el límite gratuito de Gemini, unas
# pocas decenas bastan para toparlo; si no se topa, el número se sube.
PETICIONES = 30

# Parámetros del reintento. Se acortan respecto de los valores por defecto para
# que el ensayo no tarde minutos: lo que se quiere ver es la forma de la curva
# (1, 2, 4, 6, 6...), no esperar 30 segundos.
INTENTOS = 5
ESPERA_BASE = 1.0
ESPERA_MAXIMA = 6.0

PREGUNTA = "Responde solo con la palabra: ok"


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    url, body = gemini.armar_peticion(PREGUNTA)

    print(f"Modelo:        {url.split('/models/')[1].split(':')[0]}")
    print(f"Peticiones:    {PETICIONES}")
    print(f"Reintentos:    {INTENTOS} por petición, base {ESPERA_BASE}s, techo {ESPERA_MAXIMA}s")
    print("-" * 70)

    contador = {"enviadas": 0, "temporales": 0, "ok": 0, "agotadas": 0, "tokens": 0}

    def enviar_contando():
        """Envía y cuenta, dejando pasar el error para que lo maneje el cliente."""
        contador["enviadas"] += 1
        try:
            datos = gemini.enviar_peticion(url, body)
        except Exception as error:
            if type(error).__name__ == "ErrorTemporal":
                contador["temporales"] += 1
            raise

        contador["ok"] += 1
        contador["tokens"] += (datos.get("usageMetadata") or {}).get(
            "totalTokenCount", 0
        )
        return datos

    inicio = time.perf_counter()

    try:
        for numero in range(1, PETICIONES + 1):
            try:
                con_reintento(
                    enviar_contando,
                    intentos=INTENTOS,
                    espera_base=ESPERA_BASE,
                    espera_maxima=ESPERA_MAXIMA,
                )
            except ErrorReintentosAgotados:
                contador["agotadas"] += 1
                print(f"  petición {numero}: agotó los reintentos")
            except ErrorAutenticacion as error:
                print(f"\nClave rechazada: {error}")
                return 1
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.")

    duracion = time.perf_counter() - inicio

    print("-" * 70)
    print(f"Peticiones enviadas a la red: {contador['enviadas']}")
    print(f"Fallos temporales observados: {contador['temporales']}")
    print(f"Respuestas exitosas:          {contador['ok']}")
    print(f"Operaciones agotadas:         {contador['agotadas']}")
    print(f"Tokens informados por Gemini: {contador['tokens']}")
    print(f"Duración total:               {duracion:.1f}s")

    if contador["temporales"] == 0:
        print(
            "\nNo se topó el límite. Sube PETICIONES o baja el tiempo entre "
            "peticiones: sin un 429 real, el ensayo no prueba nada."
        )
    else:
        print(
            "\nSe topó el límite y el backoff lo absorbió: las operaciones que "
            "aparecen como exitosas se recuperaron después de esperar."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
