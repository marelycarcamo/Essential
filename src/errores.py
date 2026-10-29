"""Excepciones propias del cliente de modelos.

Por qué existen
---------------
Las excepciones nativas no alcanzan cuando el código que llama tiene que tomar
una decisión: "¿reintento o no?". Con ``RuntimeError`` para todo, quien llama
no puede distinguir un 401 (no reintentar) de un 429 (sí reintentar) sin volver
a inspeccionar el código HTTP por su cuenta.

Por qué heredan de builtins
---------------------------
Cada excepción hereda de una nativa a propósito::

    ErrorAutenticacion      -> ValueError     (no se reintenta)
    ErrorTemporal           -> TimeoutError   (sí se reintenta)
    ErrorReintentosAgotados -> RuntimeError   (se agotaron los intentos)

Así los ``except ValueError`` y ``except RuntimeError`` que ya existen en el
proyecto siguen funcionando igual. Heredar conserva el contrato viejo y le
agrega un nombre que se explica solo.
"""


class ErrorCliente(Exception):
    """Base común de todos los errores del cliente de modelos."""


class ErrorAutenticacion(ErrorCliente, ValueError):
    """La credencial no es válida o no tiene permiso (HTTP 401 o 403).

    No se reintenta: el problema está en la configuración, y esperar no lo
    arregla. Hay que corregir la clave o los permisos y volver a intentar.
    """


class ErrorTemporal(ErrorCliente, TimeoutError):
    """Falla momentánea del proveedor o de la red.

    Cubre los HTTP 429 (límite de uso), 5xx (caída del proveedor) y los
    timeouts o errores de conexión. Sí se reintenta: el problema se resuelve
    solo después de una breve espera.
    """


class ErrorReintentosAgotados(ErrorCliente, RuntimeError):
    """Se agotaron los reintentos de una operación que sí era reintentable.

    Es la traducción del ``ErrorTemporal`` final a un mensaje pensado para
    quien consume el cliente, no para quien lo construyó.
    """
