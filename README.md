---
title: Buscador de ropa por lenguaje natural
emoji: 👕
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Busqueda hibrida sobre 105k prendas, en lenguaje natural
---

# Buscador de ropa por lenguaje natural

> **Este README todavia es un esqueleto.** La etapa 8 del plan es escribirlo
> a mano: es el documento que mas gente va a leer del proyecto.
>
> La cabecera YAML de arriba **no se puede borrar**: es lo que le dice a
> Hugging Face Spaces como construir y servir la app. El texto de debajo si
> es para reescribir entero.

Buscador sobre un catalogo de ~105.000 prendas de H&M. El usuario escribe una
frase libre ("chaqueta vaquera oversize para una boda en verano, que no sea
negra") y el sistema devuelve prendas reales.

## Como funciona

Un LLM traduce la frase a filtros estructurados mas una consulta semantica.
Los filtros se aplican como `WHERE` en SQL; la consulta semantica ordena por
similitud vectorial lo que sobrevive al filtro. Busqueda hibrida, no solo
vectorial.

Ese orden importa: buscando solo por vectores, "zapatillas para correr"
devuelve pantalones de chandal antes que zapatillas, y "pijama de nino"
devuelve pijamas de mujer.

## Pendiente de escribir (etapa 8)

- Diagrama de arquitectura en Mermaid.
- Por que busqueda hibrida y no solo vectorial.
- Como se controlan las alucinaciones del LLM.
- Limitaciones conocidas: no hay precio, catalogo en ingles, taxonomia de H&M.
- Roadmap.
- Instrucciones de instalacion para Windows.
- Hueco arriba para el GIF.
