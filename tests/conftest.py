"""Configuración compartida de pytest.

Pone ``src/`` y la raíz del proyecto en ``sys.path`` para que los tests puedan
importar los módulos tal como lo hace ``main.py`` (imports "pelados": ``from
consumir_openrouter import ...``) y también ``main``.

Además aísla el acumulado de costo: es estado global de módulo, así que sin
reiniciarlo el resultado de un test dependería de cuántos tests corrieron antes.
"""

import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
SRC = RAIZ / "src"

for ruta in (str(RAIZ), str(SRC)):
    if ruta not in sys.path:
        sys.path.insert(0, ruta)


@pytest.fixture(autouse=True)
def costo_limpio():
    """Deja el acumulado de costo en cero antes y después de cada test."""

    import costos

    costos.reiniciar_costo()
    yield
    costos.reiniciar_costo()
