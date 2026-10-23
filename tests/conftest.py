"""Configuración compartida de pytest.

Pone ``src/`` y la raíz del proyecto en ``sys.path`` para que los tests puedan
importar los módulos tal como lo hace ``main.py`` (imports "pelados": ``from
consumir_openrouter import ...``) y también ``main``.
"""

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
SRC = RAIZ / "src"

for ruta in (str(RAIZ), str(SRC)):
    if ruta not in sys.path:
        sys.path.insert(0, ruta)
