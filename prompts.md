# Prompts para Claude Code, en orden

Uno por sesión. **Commit entre etapa y etapa.** Si una etapa no funciona, no
sigas: arréglala antes. El orden importa — cada etapa se puede probar sola, y
eso es lo que evita depurar cinco cosas a la vez.

---

## Etapa 1 — Base de datos y esquema

> Lee `CLAUDE.md` y `models.py` antes de empezar.
>
> Crea `db.py` con:
> - una función `conectar()` que abra `ropa.db` (ruta desde la variable de
>   entorno `DB_PATH`), cargue la extensión `sqlite-vec` y devuelva la conexión
>   con `row_factory` configurado para acceder a las columnas por nombre;
> - una función `crear_esquema(con)` que cree, si no existen:
>   - tabla `articulos` con `article_id` como clave primaria y las columnas que
>     necesito según `models.py`: `product_code`, `prod_name`,
>     `product_type_name`, `product_group_name`, `index_group_name`,
>     `perceived_colour_master_name`, `perceived_colour_value_name`,
>     `graphical_appearance_name`, `detail_desc`;
>   - índices en las columnas que se usan como filtro;
>   - tabla virtual `vec_productos` con `sqlite-vec`, para vectores de 384
>     dimensiones, indexada por `product_code`.
>
> Añade un `if __name__ == "__main__"` que cree el esquema y liste las tablas,
> para poder comprobarlo a mano. No escribas nada más todavía.

**Compruebas:** `python db.py` crea el fichero y lista las tablas.

---

## Etapa 2 — Ingest

> Crea `ingest.py`: carga `articles.csv` con pandas y lo inserta en la tabla
> `articulos`.
>
> - Acepta un argumento `--limite N` (por defecto 5000) para trabajar con una
>   muestra durante el desarrollo.
> - Descarta las filas cuyo `product_group_name` esté en la lista de categorías
>   que no son ropa (está en `CLAUDE.md`).
> - Descarta las filas con `detail_desc` nulo.
> - Inserta por lotes con `executemany`, no fila a fila.
> - Al terminar, imprime cuántas filas se insertaron, cuántos `product_code`
>   distintos hay y el reparto por `index_group_name`.
>
> Que sea idempotente: si se ejecuta dos veces, no duplica.

**Compruebas:** los números del resumen cuadran y una consulta manual devuelve
prendas.

---

## Etapa 3 — Búsqueda estructurada, sin IA todavía

> Crea `buscador.py` con una función `buscar_por_filtros(con, filtros:
> SearchFilters, limite: int = 24) -> list[dict]` que construya el WHERE a
> partir de los campos no vacíos de `SearchFilters`.
>
> - Usa parámetros SQL, nunca interpolación de strings.
> - `colores` es un IN; `colores_excluidos` es un NOT IN.
> - Agrupa por `product_code` para no devolver la misma prenda repetida en
>   varias variantes de color.
> - Ignora por completo `consulta_semantica` en esta etapa.
>
> Y crea `main.py`: app FastAPI con un endpoint `GET /buscar` que reciba los
> filtros como parámetros de query y devuelva el JSON.

**Compruebas:** en `/docs`, filtrando por `Menswear` + `Blue` + `Dark` salen
prendas coherentes. **Este es tu primer commit que ya vale como proyecto.**

---

## Etapa 4 — Capa semántica

> Crea `embeddings.py`:
> - carga un modelo multilingüe de `sentence-transformers` de 384 dimensiones
>   (elige uno y justifica la elección en un comentario);
> - función `texto_de(fila) -> str` que concatene `prod_name`,
>   `product_type_name` y `detail_desc`;
> - función `indexar(con)` que calcule un embedding **por `product_code`**
>   (una sola vez por producto, no por variante de color), en lotes, con barra
>   de progreso, y los guarde en `vec_productos`;
> - función `embeder_consulta(texto) -> list[float]`.
>
> El modelo se carga una sola vez a nivel de módulo, no en cada llamada.
>
> Añade a `buscador.py` una función `buscar_semantica(con, texto, limite)` que
> devuelva los `product_code` más cercanos con su distancia.

**Compruebas:** buscar "algo abrigado para el invierno" devuelve abrigos, sin
haber tocado ningún LLM. **Si esto no funciona en español, el modelo elegido no
sirve — pruébalo antes de seguir.**

---

## Etapa 5 — El extractor LLM

> Crea `extractor.py` con `extraer_filtros(frase: str) -> SearchFilters`, que
> llame a Groq pidiendo salida JSON conforme al JSON Schema de `SearchFilters`.
>
> Requisitos, y esta es la parte importante del proyecto:
> - el system prompt se genera a partir del schema de Pydantic, no se escribe
>   a mano duplicando los valores de los enums;
> - valida la respuesta con `SearchFilters.model_validate`;
> - si un campo concreto trae un valor fuera del enum, descarta **ese campo**
>   y sigue; no tires la consulta entera;
> - si el JSON viene roto, un reintento y luego un fallback que devuelva
>   `SearchFilters` con solo `consulta_semantica` = la frase original;
> - loguea frase de entrada y filtros extraídos.
>
> Añade `tests/test_extractor.py` con al menos seis frases de ejemplo en
> español, comprobando qué campos deben salir rellenos. Mockea la llamada a
> Groq en los tests salvo en uno marcado como test de integración.

**Compruebas:** `pytest -q` en verde y las seis frases producen filtros
sensatos.

---

## Etapa 6 — Fusión

> Añade a `buscador.py` la función `buscar(con, frase) -> dict` que orqueste
> todo: extraer filtros, aplicar el WHERE, y ordenar ese subconjunto por
> similitud semántica con `consulta_semantica`.
>
> - Los filtros duros van primero, el ranking semántico después y solo sobre
>   lo que sobrevive.
> - Si el resultado es cero, relaja los filtros en este orden: `estampado`,
>   `tono`, `colores`, `categoria`, y deja `publico` para el final. Devuelve
>   qué filtros se relajaron.
> - La respuesta incluye: resultados, filtros aplicados, filtros relajados.
>
> Cambia el endpoint a `GET /buscar?q=<frase>`.

**Compruebas:** una frase compleja completa devuelve resultados con sentido.

---

## Etapa 7 — Frontend

> Monta el front con Jinja2 y HTMX (por CDN, sin build):
> - `templates/index.html`: input de búsqueda, y una rejilla de tarjetas de
>   resultado;
> - `templates/_resultados.html`: fragmento parcial que HTMX inserta;
> - **un panel lateral que muestre el JSON de filtros que el LLM extrajo**, y
>   un aviso visible si se relajó alguno;
> - estado de carga mientras se resuelve la consulta.
>
> Sin CSS de framework: hoja propia, sobria, legible.

**Compruebas:** alguien que no sabe nada del proyecto lo usa sin explicaciones.
Ese panel de filtros es lo que convierte la demo de "parece magia" a "se ve la
ingeniería".

---

## Etapa 8 — README

> Escribe el `README.md`: qué problema resuelve, diagrama de arquitectura en
> Mermaid, **por qué búsqueda híbrida y no solo vectorial**, cómo se controlan
> las alucinaciones del LLM, limitaciones conocidas (no hay precio, catálogo en
> inglés, taxonomía de H&M) y roadmap.
>
> Instrucciones de instalación que funcionen en Windows. Hueco arriba para un
> GIF.

Este documento lo leerán más que el código. Escríbelo tú y que Claude Code solo
lo revise.

---

## Etapa 9 — Datos completos y despliegue

Carga ya sí las 105k filas (`--limite 0`), graba el GIF, y despliega en Render
o Railway con un `Dockerfile`. Si el plan gratuito no aguanta el modelo de
embeddings en RAM, reduce el catálogo antes de renunciar al despliegue.

---

## Etapa 10 — Opcionales, solo si lo anterior está vivo

Búsqueda por imagen con CLIP. Precio calculado desde
`transactions_train.csv`. Caché de consultas frecuentes.
