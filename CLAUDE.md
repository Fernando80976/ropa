# Buscador de ropa por lenguaje natural

## Qué es esto

Buscador sobre un catálogo de ~105k prendas de H&M. El usuario escribe una frase
libre ("chaqueta vaquera oversize para una boda en verano, que no sea negra") y
el sistema devuelve prendas reales.

**Arquitectura en una línea:** un LLM traduce la frase a filtros estructurados +
una consulta semántica; los filtros se aplican como WHERE en SQL y la consulta
semántica ordena los resultados por similitud vectorial. Búsqueda híbrida, no
solo vectorial.

## Contexto importante sobre quién mantiene esto

Es un proyecto de portfolio de un desarrollador junior de backend Python. El
objetivo no es que funcione, es que **sea explicable en una entrevista técnica**.
Consecuencias directas:

- Prefiere código obvio a código listo. Sin metaprogramación, sin abstracciones
  preventivas, sin patrones que no resuelvan un problema presente.
- Cada fichero debe poder leerse de arriba abajo y entenderse sin saltos.
- Si una decisión tiene alternativa razonable, deja un comentario de una línea
  explicando por qué esta y no la otra.
- Comentarios y docstrings en español. Nombres de variables y funciones en
  español también, salvo términos técnicos consolidados (embedding, query).

## Stack

| Capa | Elección | Por qué |
|---|---|---|
| API | FastAPI + uvicorn | tipado, `/docs` gratis |
| BD | SQLite + extensión `sqlite-vec` | cero instalación; Docker no era viable en este equipo |
| Acceso a datos | módulo `sqlite3` de la stdlib, SQL a mano | sin ORM: el SQL es parte de lo que se quiere demostrar |
| Embeddings | `onnxruntime` + `tokenizers`, modelo **solo inglés** (`all-MiniLM-L6-v2`, int8) | el catálogo está en inglés y el LLM ya traduce la consulta; cabe en 512 MB de RAM |
| LLM | Groq (SDK compatible con OpenAI) | capa gratuita, latencia baja, JSON estructurado |
| Front | Jinja2 + HTMX | sin build step, sin node_modules |
| Tests | pytest | |

Entorno: **Windows + PowerShell**, entorno virtual en `.venv`. Docker no está
instalado y no se va a instalar: no propongas soluciones que dependan de él.

## Estructura del proyecto

```
buscador-ropa/
  models.py        # SearchFilters: el contrato LLM <-> BD. LEER SIEMPRE ANTES DE TOCAR NADA
  db.py            # conexión, carga de sqlite-vec, creación de esquema
  ingest.py        # articles.csv -> SQLite (script de un solo uso)
  embeddings.py    # cálculo y almacenamiento de vectores (onnxruntime, sin torch)
  extractor.py     # frase -> SearchFilters vía Groq
  buscador.py      # fusión: filtros SQL + ranking semántico
  main.py          # app FastAPI y rutas
  preparar_modelo.py  # descarga el modelo de embeddings (no va en git)
  preparar_datos.py   # descarga ropa.db ya indexada (no va en git)
  templates/       # Jinja2
  static/          # hoja de estilos propia
  tests/
  modelo_onnx/     # NO versionado, lo crea preparar_modelo.py
  articles.csv     # NO versionado
  ropa.db          # NO versionado
  .env             # NO versionado
```

## Reglas del dataset (articles.csv)

Estas cosas ya están comprobadas contra los datos reales. No las re-descubras ni
las contradigas:

- 105.542 filas, 25 columnas. **No hay columna de precio** (vive en
  `transactions_train.csv`, normalizado). El precio queda fuera del alcance.
- **Hay duplicados por variante de color:** solo 45.875 `prod_name` únicos y
  43.404 `detail_desc` únicos. Los embeddings se calculan **una vez por
  `product_code`**, no por fila. En los resultados, las variantes se agrupan.
- `detail_desc` tiene 416 nulos. Es el texto principal a embeber, concatenado
  con `prod_name` y `product_type_name`.
- Filtrar en el ingest las categorías que no son ropa: `Furniture`,
  `Stationery`, `Cosmetic`, `Interior textile`, `Fun`, `Items`,
  `Garment and Shoe care`, `Bags`, `Unknown` (~220 filas en total).
- Un tercio del catálogo (34.711 filas) es `Baby/Children`. Sin filtro de
  público, una consulta de adulto devuelve ropa de bebé.

## Decisiones cerradas — no las revisites sin preguntar

1. `product_type_name` (131 valores), `department_name` (250) y `section_name`
   (56) **no son filtros**. Son demasiados o son jerga comercial interna. El
   tipo concreto de prenda se resuelve por similitud semántica.
2. El color se filtra con `perceived_colour_master_name` (20 familias) +
   `perceived_colour_value_name` (8 tonos), que son ortogonales. **No** con
   `colour_group_name`, que tiene 50 valores ruidosos.
3. Los valores de los `Literal` en `models.py` están en inglés porque son los
   valores literales de la BD. Las descripciones de los campos están en español
   porque acaban en el JSON Schema que ve el LLM.
4. Todo lo que el LLM no consigue estructurar va al campo
   `consulta_semantica`, que nunca puede quedar vacío.
5. **`consulta_semantica` se pide en inglés**, aunque el usuario escriba en
   español, y el modelo de embeddings es solo inglés. Parece al revés de lo
   razonable, así que conviene saber por qué:

   - El LLM ya está reescribiendo la frase. Pedirle que además la traduzca no
     añade ni una llamada ni latencia apreciable.
   - Así se compara inglés contra inglés (`detail_desc` lo está) en vez de
     cruzar dos idiomas, que es un problema peor planteado.
   - Y sobre todo: un modelo multilingüe necesita un vocabulario de 250.002
     tokens, y **el tokenizador solo ocupa 262 MB en RAM**, más que el propio
     modelo. Medido: 451 MB el multilingüe frente a 99 MB el inglés. Es la
     diferencia entre caber o no caber en un servidor gratuito de 512 MB —
     el primer despliegue murió justo ahí, con "Out of memory".

   El precio está documentado en `models.py`: si Groq no responde, el fallback
   manda la frase en español a un modelo que solo entiende inglés. Se acepta
   porque ese camino ya era el peor (se queda sin ningún filtro duro).
6. El modelo no se versiona ni se descarga al arrancar: lo baja
   `preparar_modelo.py` con **la revisión fijada por hash**, no por `main`. Si
   el repositorio de origen publicase otro fichero con el mismo nombre, los
   vectores dejarían de corresponderse con los indexados y el ranking se
   volvería ruido **sin dar ningún error**.

## Cómo tratar la salida del LLM

Esta es la parte del proyecto que más se va a mirar. Nunca confíes en la
respuesta del modelo:

- Validar **siempre** contra `SearchFilters` con Pydantic antes de tocar la BD.
- Si el JSON viene roto o un valor está fuera del enum: descartar ese campo
  concreto, no la consulta entera. Degradar, no fallar.
- Si los filtros dejan cero resultados, relajarlos por orden de rigidez y
  **avisar al usuario** de que se amplió la búsqueda. Un buscador que devuelve
  vacío es un buscador roto.
- Registrar en logs la frase de entrada y los filtros extraídos. Es lo que
  permite depurar prompts.

## Comandos

```powershell
.\.venv\Scripts\Activate.ps1        # SIEMPRE, en cada terminal nueva
python preparar_modelo.py           # una vez: baja el modelo a modelo_onnx/
python preparar_datos.py            # una vez: baja ropa.db ya indexada
uvicorn main:app --reload
pytest -q                           # los tests de integración van aparte:
pytest -q -m integracion            #   llaman a Groq de verdad
```

Si se cambia de modelo de embeddings hay que **reindexar entero**, porque los
vectores de dos modelos distintos no comparten espacio vectorial y mezclarlos
da un ranking sin sentido, sin ningún error visible:

```powershell
python embeddings.py --recalcular
```

## Cosas que no hacer

- No añadir dependencias sin preguntar antes.
- No modificar `models.py` sin avisar de qué cambia y por qué.
- No proponer React, Next, Docker, Postgres local ni Alembic.
- No inventar nombres de columnas: si dudas, léelas de `articles.csv`.
- No hardcodear la API key. Todo por `.env` + `python-dotenv`.
- No cargar las 105k filas mientras se desarrolla: trabajar con una muestra y
  dejar la carga completa para el final.