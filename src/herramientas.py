"""Herramientas del agente (Bloque 21).

Por qué viven aquí y no dentro del archivo del agente
-----------------------------------------------------
Una herramienta no es parte del agente: es una capacidad que el agente *decide*
usar. ``agente_con_herramientas.py`` arma el ciclo de razonamiento; este módulo
sabe leer la documentación del proyecto y calcular tarifas. Si mañana la misma
capacidad se expone en otro agente o en un servicio HTTP, se importa desde aquí
sin copiar código.

Las dos herramientas de este bloque son de **solo lectura** a propósito. El
Bloque 23 pide clasificar cada herramienta por riesgo, y una clasificación seria
necesita al menos una que escriba: esa llega en el Bloque 22, junto con la
aprobación humana que la vuelve segura.

Por qué devuelven texto en vez de lanzar excepciones
----------------------------------------------------
Dentro de un agente, una excepción corta el ciclo completo: la persona recibe un
``Traceback`` en vez de una respuesta. Si la herramienta no puede cumplir, lo
dice —"no hay tarifas de ese modelo", "no encontré nada sobre eso"— y el modelo
decide qué hacer con esa información. El error técnico queda en el log.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.tools import tool

from costos import PRECIOS, calcular_costo, es_hora_punta

logger = logging.getLogger(__name__)

RAIZ = Path(__file__).resolve().parent.parent
DIRECTORIO_DOCUMENTACION = RAIZ / "docs"
ARCHIVOS_EN_LA_RAIZ = (RAIZ / "README.md",)
SUFIJO_MARKDOWN = ".md"

# Un fragmento más corto que esto no es un fragmento: es un título o una fila
# de tabla sin contexto, y responder con él confunde más que no responder.
MINIMO_CARACTERES_FRAGMENTO = 40

# Cuántos fragmentos se devuelven como máximo. Tres es lo que cabe en un
# contexto sin desplazar a la pregunta; más sería recuperar por recuperar.
MAXIMO_FRAGMENTOS = 3

# Palabras de menos de cuatro letras ("de", "la", "que") aparecen en todos los
# párrafos y no discriminan: si se puntuaran, cualquier fragmento empataría.
MINIMO_LARGO_DE_TERMINO = 4

# El texto alternativo de una imagen del README son cientos de caracteres
# describiendo un diagrama: no responde ninguna pregunta, pero por longitud y
# por vocabulario le ganaba a los párrafos que sí responden. Se descarta antes
# de armar los fragmentos.
PATRON_RUIDO_DE_IMAGEN = re.compile(r"!\[|data:image")

# Tope de un fragmento. Una tabla de markdown sin líneas en blanco es un solo
# párrafo de miles de caracteres: entra completo en el contexto, empuja a los
# demás y no se puede citar sin copiar media página.
MAXIMO_CARACTERES_FRAGMENTO = 700


def _sin_tildes(texto: str) -> str:
    """Quita tildes y diéresis conservando la letra: ``tarífa`` → ``tarifa``.

    Así se compara lo que la persona escribe con lo que está escrito en los
    documentos sin obligarla a acertar la ortografía.
    """

    descompuesto = unicodedata.normalize("NFD", texto)

    return "".join(
        caracter
        for caracter in descompuesto
        if unicodedata.category(caracter) != "Mn"
    )


def terminos_de(consulta: str) -> list[str]:
    """Palabras significativas de la consulta, sin tildes y en minúsculas.

    Args:
        consulta: texto libre escrito por la persona.

    Returns:
        Lista de términos de al menos ``MINIMO_LARGO_DE_TERMINO`` letras. Vacía
        si la consulta no aporta nada buscable.
    """

    limpia = _sin_tildes(consulta).lower()

    return [
        palabra
        for palabra in re.findall(r"[a-z0-9]+", limpia)
        if len(palabra) >= MINIMO_LARGO_DE_TERMINO
    ]


@dataclass(frozen=True)
class Fragmento:
    """Un trozo de documentación junto con su origen.

    ``archivo`` y ``linea`` no son decoración: sin ellos la respuesta del agente
    no se puede verificar, y una respuesta que no se puede verificar no sirve
    para decidir nada.
    """

    archivo: str
    linea: int
    texto: str


def archivos_de_documentacion() -> list[Path]:
    """Documentos que el agente puede leer: ``docs/*.md`` y el README."""

    archivos = sorted(DIRECTORIO_DOCUMENTACION.glob(f"*{SUFIJO_MARKDOWN}"))
    archivos += [ruta for ruta in ARCHIVOS_EN_LA_RAIZ if ruta.is_file()]

    return archivos


def cargar_fragmentos(archivos: list[Path] | None = None) -> list[Fragmento]:
    """Parte la documentación en fragmentos citables: párrafo, título o fila.

    El corte sigue la estructura del documento y no una cantidad de caracteres
    arbitraria: cada párrafo, cada título y cada fila de tabla es una unidad con
    sentido propio. Un encabezado agrupado con lo que sigue, o una tabla entera
    tratada como un párrafo, producen fragmentos que hablan de varias cosas y
    que después no se pueden citar.

    Args:
        archivos: documentos a leer. Si es ``None`` se usan los del proyecto.

    Returns:
        Fragmentos en el orden en que aparecen en sus archivos.
    """

    fragmentos: list[Fragmento] = []

    for ruta in archivos if archivos is not None else archivos_de_documentacion():
        try:
            contenido = ruta.read_text(encoding="utf-8")
        except OSError as error:
            # Un documento ilegible no debe tumbar la búsqueda entera: se salta
            # y queda constancia, porque una respuesta sin ese archivo puede
            # estar incompleta.
            logger.warning("No se pudo leer %s: %s", ruta.name, error)
            continue

        for linea, bloque in _bloques(contenido):
            for trozo in _partir_largo(" ".join(bloque)):
                if len(trozo) >= MINIMO_CARACTERES_FRAGMENTO:
                    fragmentos.append(
                        Fragmento(archivo=ruta.name, linea=linea, texto=trozo)
                    )

    return fragmentos


def _bloques(contenido: str) -> list[tuple[int, list[str]]]:
    """Agrupa el texto en bloques: párrafos, títulos y filas de tabla.

    Args:
        contenido: texto completo de un documento.

    Returns:
        Pares ``(línea donde empieza, líneas del bloque)``.
    """

    bloques: list[tuple[int, list[str]]] = []
    bloque: list[str] = []
    inicio = 1

    for numero, linea in enumerate(contenido.splitlines(), start=1):
        if PATRON_RUIDO_DE_IMAGEN.search(linea):
            continue

        limpia = linea.strip()

        if not limpia:
            if bloque:
                bloques.append((inicio, bloque))
                bloque = []
            continue

        if limpia.startswith(("#", "|")):
            if bloque:
                bloques.append((inicio, bloque))
                bloque = []
            bloques.append((numero, [limpia]))
            continue

        if not bloque:
            inicio = numero

        bloque.append(limpia)

    if bloque:
        bloques.append((inicio, bloque))

    return bloques


def _partir_largo(
    texto: str,
    maximo: int = MAXIMO_CARACTERES_FRAGMENTO,
) -> list[str]:
    """Corta un bloque que excede el tope, respetando el final de cada oración.

    Partir por caracteres dejaría fragmentos que empiezan a mitad de una idea;
    partir por oraciones conserva el sentido aunque se pierda algo de contexto.
    """

    if len(texto) <= maximo:
        return [texto]

    trozos: list[str] = []
    actual = ""

    for oracion in re.split(r"(?<=[.!?])\s+", texto):
        if actual and len(actual) + len(oracion) + 1 > maximo:
            trozos.append(actual)
            actual = oracion
            continue

        actual = f"{actual} {oracion}".strip()

    if actual:
        trozos.append(actual)

    return trozos


def puntuar(fragmento: Fragmento, terminos: list[str]) -> int:
    """Cuenta cuántos términos distintos de la consulta aparecen en el fragmento.

    Cada término se busca como **inicio de palabra**, no como subcadena suelta:
    así "tarifa" encuentra "tarifas" —que es lo que se busca— pero "mini" no
    encuentra "gemini", que era un falso positivo real y metía un párrafo sobre
    el proveedor de respaldo en una pregunta sobre el modelo descartado.
    """

    texto = _sin_tildes(fragmento.texto).lower()

    return sum(
        1
        for termino in set(terminos)
        if re.search(rf"\b{re.escape(termino)}", texto)
    )


def buscar(
    consulta: str,
    fragmentos: list[Fragmento] | None = None,
    maximo: int = MAXIMO_FRAGMENTOS,
) -> list[Fragmento]:
    """Devuelve los fragmentos que más términos de la consulta contienen.

    Ordena por cantidad de términos distintos y, a igualdad, por brevedad: entre
    dos fragmentos que cubren lo mismo, el corto es el que responde.

    Args:
        consulta: texto libre a buscar.
        fragmentos: corpus ya cargado, útil para no releer archivos en tests.
        maximo: cuántos fragmentos devolver como máximo.

    Returns:
        Los mejores fragmentos, o lista vacía si ninguno comparte un término
        con la consulta.
    """

    terminos = terminos_de(consulta)

    if not terminos:
        return []

    disponibles = fragmentos if fragmentos is not None else cargar_fragmentos()

    puntuados = [(puntuar(f, terminos), len(f.texto), f) for f in disponibles]
    utiles = [item for item in puntuados if item[0] > 0]
    utiles.sort(key=lambda item: (-item[0], item[1]))

    return [fragmento for _, _, fragmento in utiles[:maximo]]


def _formatear(fragmentos: list[Fragmento]) -> str:
    """Arma el texto que recibe el modelo, con el origen de cada fragmento."""

    return "\n\n".join(
        f"[{fragmento.archivo}:{fragmento.linea}] {fragmento.texto}"
        for fragmento in fragmentos
    )


@tool
def buscar_en_documentacion(consulta: str) -> str:
    """Busca en la documentación del proyecto (docs/*.md y el README).

    Úsala cuando la pregunta sea sobre decisiones, mediciones o límites de este
    proyecto y la respuesta pueda estar escrita: por qué se eligió un modelo,
    qué latencia se midió, qué limitaciones se reconocen. No la uses para
    conversación general ni para cálculos.

    Args:
        consulta: términos a buscar, en lenguaje natural.

    Returns:
        Hasta tres fragmentos con su archivo y línea de origen, o un mensaje
        explicando por qué no hay resultados.
    """

    if not terminos_de(consulta):
        return (
            "No puedo buscar con eso: no hay palabras de cuatro letras o más. "
            "Dame términos concretos, por ejemplo «latencia DeepSeek»."
        )

    try:
        fragmentos = cargar_fragmentos()
    except OSError as error:
        logger.error("Falló la lectura de la documentación: %s", error)
        return (
            "No pude leer la documentación del proyecto. "
            "Revisa que el directorio docs/ exista y tenga permisos de lectura."
        )

    encontrados = buscar(consulta, fragmentos)

    if not encontrados:
        revisados = ", ".join(sorted({f.archivo for f in fragmentos})) or "ninguno"
        return (
            f"No encontré nada sobre «{consulta}» en la documentación. "
            f"Archivos revisados: {revisados}."
        )

    return _formatear(encontrados)


@tool
def estimar_costo(modelo: str, tokens_entrada: int, tokens_salida: int) -> str:
    """Calcula en dólares lo que costaría una llamada a un modelo de DeepSeek.

    Úsala cuando la pregunta sea cuánto cuesta o costaría generar algo: un
    material, una respuesta, una tanda de documentos. No la uses para precios de
    otros proveedores: solo conoce las tarifas registradas de DeepSeek.

    Args:
        modelo: nombre exacto del modelo, por ejemplo «deepseek-flash» o
            «deepseek-v4-pro».
        tokens_entrada: tokens que se envían en el prompt.
        tokens_salida: tokens que se espera recibir en la respuesta.

    Returns:
        El costo en USD con la tarifa vigente en este momento, o un mensaje
        explicando por qué no se pudo calcular.
    """

    if tokens_entrada < 0 or tokens_salida < 0:
        return "Los tokens no pueden ser negativos: revisa los números."

    if modelo not in PRECIOS:
        disponibles = ", ".join(sorted(PRECIOS))
        return (
            f"No tengo tarifas de «{modelo}». "
            f"Modelos con tarifas registradas: {disponibles}."
        )

    datos = {
        "usage": {
            "prompt_tokens": tokens_entrada,
            "completion_tokens": tokens_salida,
        }
    }

    # Se fija el instante una sola vez para que el cálculo y la etiqueta de
    # tarifa hablen del mismo momento, aunque la llamada cruce el cambio de hora.
    momento = datetime.now(timezone.utc)

    try:
        costo = calcular_costo(datos, modelo=modelo, momento=momento)
    except ValueError as error:
        logger.error("Cálculo de costo rechazado: %s", error)
        return f"No pude calcular el costo: {error}"

    tarifa = "punta" if es_hora_punta(momento) else "valle"

    return (
        f"{tokens_entrada} tokens de entrada y {tokens_salida} de salida con "
        f"{modelo}, en hora {tarifa}: US$ {costo:.8f}. "
        "Asume que la caché no acierta, así que es el máximo a pagar."
    )
