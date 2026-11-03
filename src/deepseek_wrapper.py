"""Wrapper propio para DeepSeek usando sólo ``requests``.

No depende de ``langchain-deepseek`` ni de ``tiktoken``, así que no rompe
el entorno actual en Windows con Smart App Control.

La clase expone ``invoke(self, input)`` compatible con la cadena de
LangChain: recibe un diccionario ``{"texto": "..."}`` y devuelve un
objeto con ``.content`` que ``StrOutputParser`` puede procesar.
"""

import os
import requests
from types import SimpleNamespace

from dotenv import load_dotenv

load_dotenv()

# --- Credenciales desde el .env existente ---
API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


class DeepSeekWrapper:
    """Wrapper mínimo para usar DeepSeek como respaldo en la chain."""

    def __init__(self, api_key=None, base_url=None, model=None):
        self.api_key = api_key or API_KEY
        self.base_url = base_url or BASE_URL
        self.model = model or MODEL
        if not self.api_key:
            raise RuntimeError(
                "Falta DEEPSEEK_API_KEY en las variables de entorno"
            )

    def _payload(self, texto: str) -> dict:
        """Arma el JSON que DeepSeek espera."""
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": texto}],
            "temperature": 0,
        }

    def invoke(self, input_data):
        """Recibe el input que LangChain le pasa al modelo.

        Puede llegar como:
        - Un ``dict`` ``{"texto": "..."}`` (cuando se llama directo).
        - Un ``StringPromptValue`` con ``.text`` (cuando viene de un
          ``PromptTemplate`` en una chain).

        En ambos casos, devolvemos un objeto con ``.content`` para que
        ``StrOutputParser`` pueda extraer el texto.
        """
        if isinstance(input_data, dict):
            texto = input_data.get("texto", "")
            if not texto:
                # Fallback: usar el primer valor string del dict
                for valor in input_data.values():
                    if isinstance(valor, str):
                        texto = valor
                        break
        else:
            texto = getattr(input_data, "text", str(input_data))

        if not texto:
            raise ValueError("No se encontró texto en el input")

        url = f"{self.base_url}/chat/completions"
        payload = self._payload(texto)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        respuesta = requests.post(url, json=payload, headers=headers, timeout=30)
        respuesta.raise_for_status()

        data = respuesta.json()
        contenido = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return SimpleNamespace(content=contenido)