"""
Traduccion de una frase en lenguaje natural a SearchFilters, via Groq.

Esta es la parte fragil del proyecto: el resto es SQL determinista, y aqui
entra un modelo que puede devolver cualquier cosa. La regla es que nada de lo
que diga el LLM llega a la BD sin pasar por Pydantic, y que un fallo suyo
degrada la busqueda pero no la rompe.

Tres niveles de defensa, de menos a mas grave:

  1. un campo trae un valor que no esta en el enum -> se descarta ESE campo
     y el resto de la consulta sigue adelante;
  2. el JSON viene roto -> un reintento;
  3. el reintento tambien falla -> se devuelve la frase original como
     consulta_semantica, y la busqueda pasa a ser puramente vectorial.

El caso 3 sigue dando resultados utiles: es la etapa 4 funcionando sola.
"""

import json
import logging
import os

from dotenv import load_dotenv
from groq import Groq
from pydantic import ValidationError

from models import SearchFilters

load_dotenv()

log = logging.getLogger(__name__)

# openai/gpt-oss-120b es el mas capaz de los que ofrece la capa gratuita de
# Groq que ademas soporta response_format json_schema con strict. Sin strict
# habria que parsear texto libre y el numero de reintentos se dispararia.
MODELO = os.getenv("GROQ_MODELO", "openai/gpt-oss-120b")

# 0 porque esto no es una tarea creativa: la misma frase debe dar siempre los
# mismos filtros, o el buscador no es reproducible ni depurable.
TEMPERATURA = 0

cliente = Groq(api_key=os.getenv("GROQ_API_KEY"))


def esquema_json() -> dict:
    """
    JSON Schema de SearchFilters, ajustado a lo que exige Groq en modo strict.

    Se genera a partir del modelo de Pydantic a proposito. Escribirlo a mano
    significaria tener los valores de cada enum en dos sitios, y el dia que
    models.py cambie el prompt seguiria pidiendo los viejos sin que nada falle
    de forma visible: el LLM devolveria valores que la BD ya no tiene.
    """
    esquema = SearchFilters.model_json_schema()

    # Groq rechaza el schema si algun objeto no lo lleva.
    esquema["additionalProperties"] = False

    # En modo strict todas las propiedades tienen que estar en `required`.
    # No obliga a rellenarlas: los campos opcionales son `anyOf [tipo, null]`,
    # asi que el modelo cumple poniendo null explicitamente.
    esquema["required"] = list(esquema["properties"])

    return esquema


SISTEMA = """Eres un extractor de filtros para un buscador de ropa.

Recibes una frase en espanol de un usuario que busca una prenda y devuelves
los filtros estructurados que le corresponden, siguiendo el esquema JSON.

Reglas:
- Rellena un campo SOLO si la frase lo dice o se deduce sin inventar. Ante la
  duda, deja null: un filtro de mas deja al usuario sin resultados.
- consulta_semantica NUNCA puede ir vacia. Ahi va todo lo que no ha entrado en
  los otros campos (tipo de prenda, corte, ocasion, tejido, estilo). Si la
  frase era solo filtros, repite ahi el tipo de prenda.
- No traduzcas consulta_semantica: dejala en espanol."""


def _mensajes(frase: str) -> list[dict]:
    """El system prompt no enumera los valores validos: ya van en el schema."""
    return [
        {"role": "system", "content": SISTEMA},
        {"role": "user", "content": frase},
    ]


def _llamar(frase: str) -> str:
    """Una llamada a Groq. Devuelve el contenido crudo, sin parsear."""
    respuesta = cliente.chat.completions.create(
        model=MODELO,
        messages=_mensajes(frase),
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "SearchFilters",
                "schema": esquema_json(),
                "strict": True,
            },
        },
        temperature=TEMPERATURA,
    )
    return respuesta.choices[0].message.content


def validar_degradando(datos: dict, frase: str) -> SearchFilters:
    """
    Valida contra SearchFilters descartando solo los campos invalidos.

    `model_validate` es todo o nada: si un campo trae "Turquesa" en vez de
    "Turquoise", tira la respuesta entera. Aqui se leen los campos que ha
    senalado la excepcion, se quitan, y se vuelve a validar. Asi una frase con
    cinco filtros de los que uno viene mal conserva los otros cuatro.
    """
    datos = dict(datos)

    # El bucle termina porque cada vuelta borra al menos un campo y el numero
    # de campos es finito; el limite es una red por si acaso.
    for _ in range(len(SearchFilters.model_fields) + 1):
        # consulta_semantica es el unico campo obligatorio, asi que se repone
        # con la frase del usuario si falta, viene vacia o la ha descartado la
        # vuelta anterior. Sin esto, un {} del modelo tiraria la consulta
        # entera por el unico campo que siempre se puede reconstruir.
        if not str(datos.get("consulta_semantica") or "").strip():
            datos["consulta_semantica"] = frase

        try:
            return SearchFilters.model_validate(datos)
        except ValidationError as error:
            descartados = set()
            for fallo in error.errors():
                if not fallo["loc"]:
                    continue
                campo = fallo["loc"][0]
                if campo in datos:
                    del datos[campo]
                    descartados.add(campo)

            if not descartados:
                break  # el fallo no es de un campo concreto: no hay que quitar

            log.warning("Campos descartados por invalidos: %s", sorted(descartados))

    # Red de seguridad: con todos los campos opcionales descartables y
    # consulta_semantica repuesta, no deberia llegarse aqui. Si se llega, que
    # lo coja extraer_filtros y reintente.
    raise ValueError("La respuesta no es validable ni descartando campos")


def _fallback(frase: str) -> SearchFilters:
    """Sin filtros duros: la busqueda queda en puramente semantica."""
    log.error("Groq no devolvio nada usable. Se busca solo por semantica.")
    return SearchFilters(consulta_semantica=frase)


def extraer_filtros(frase: str) -> SearchFilters:
    """
    Frase del usuario -> SearchFilters. Nunca lanza.

    Un reintento ante JSON roto o error de red, y despues fallback. Un
    buscador que devuelve un 500 porque el LLM tuvo un mal dia es peor que
    uno que devuelve resultados aproximados.
    """
    log.info("Consulta: %r", frase)

    for intento in (1, 2):
        try:
            crudo = _llamar(frase)
            filtros = validar_degradando(json.loads(crudo), frase)
        except Exception as error:
            log.warning("Intento %d fallido: %s: %s", intento, type(error).__name__, error)
            continue

        # La garantia de models.py: el LLM puede devolverla vacia aunque el
        # prompt lo prohiba, y sin ella la etapa semantica no tiene entrada.
        if not filtros.consulta_semantica.strip():
            filtros.consulta_semantica = frase

        log.info("Filtros: %s", filtros.model_dump(exclude_defaults=True))
        return filtros

    return _fallback(frase)


if __name__ == "__main__":
    # Comprobacion a mano: python extractor.py "tu frase"
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    frases = sys.argv[1:] or [
        "chaqueta vaquera oversize para una boda en verano, que no sea negra",
        "vestido rojo largo para una fiesta",
        "algo comodo para estar en casa",
    ]
    for frase in frases:
        print(f"\n--- {frase}")
        print(json.dumps(extraer_filtros(frase).model_dump(), indent=2, ensure_ascii=False))
