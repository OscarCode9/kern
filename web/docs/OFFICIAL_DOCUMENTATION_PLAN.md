# Plan y Guia de Documentacion Oficial de Kern

## 1. Objetivo

Construir una documentacion oficial que sirva para dos audiencias al mismo tiempo:

- Humanos: aprender, usar, depurar y contribuir.
- LLMs/agentes: parsear reglas sin ambiguedad, generar codigo correcto, y resolver tareas automaticamente.

La meta no es solo "explicar", sino crear una fuente canonica que reduzca errores de implementacion y de generacion.

## 2. Principios de calidad

1. Canonico: una sola verdad por regla.
2. Ejecutable: cada regla importante tiene ejemplo verificable.
3. Versionado: cada cambio de sintaxis o semantica queda trazable.
4. No ambiguo: evitar sinonimos para el mismo concepto.
5. LLM-friendly: estructura estable, headings consistentes, datos estructurados.

## 3. Que debe llevar una documentacion oficial de lenguaje

## 3.1 Base minima (MVP)

1. Vision y alcance del lenguaje.
2. Quickstart de 5 minutos.
3. Especificacion de gramatica (sintaxis + semantica).
4. Referencia de construcciones (por keyword/feature).
5. Reglas de compatibilidad y versionado.
6. Catalogo de errores comunes y recovery.
7. Ejemplos canonicos (input -> output esperado).
8. Guia de toolchain (transpiler, compiler, CLI).
9. Guia de contribucion y pruebas.
10. Changelog por version.

## 3.2 Nivel produccion

1. RFC process para cambios grandes.
2. Politica de deprecaciones.
3. Test suite publica de conformidad.
4. Benchmarks de performance/costo.
5. "Migration guides" entre versiones.

## 4. Estructura recomendada para el front (`/web`)

```text
web/docs/
  README.md
  00-overview.md
  01-quickstart.md
  02-grammar/
    syntax.md
    semantics.md
    ambiguity-rules.md
  03-reference/
    functions.md
    control-flow.md
    imports.md
    classes.md
    exceptions.md
    expressions.md
  04-toolchain/
    transpiler.md
    compiler.md
    cli.md
  05-examples/
    canonical-examples.md
    roundtrip-examples.md
  06-errors/
    parser-errors.md
    transpile-errors.md
    compile-errors.md
  07-llm/
    llm-contract.md
    prompting-rules.md
    machine-readable-index.json
  08-contributing/
    style-guide.md
    testing.md
  09-changelog/
    v0.2.md
```

## 5. Contrato para LLMs (muy importante)

Cada pagina tecnica debe incluir:

1. Nombre canonico de la regla.
2. Sintaxis formal.
3. Restricciones.
4. Casos validos.
5. Casos invalidos.
6. Transformacion esperada (si aplica).
7. Tests de referencia.

Formato recomendado por seccion:

```md
## Rule: if-block
### Syntax
...
### Valid
...
### Invalid
...
### Roundtrip
Python -> Kern -> Python
### Tests
...
```

## 6. Estandares de escritura (humanos + maquinas)

1. Un concepto por seccion.
2. Titulo estable y corto.
3. Evitar lenguaje figurado en specs.
4. Ejemplos con codigo minimo reproducible.
5. Siempre indicar version de la regla.
6. Etiquetas consistentes: `Syntax`, `Valid`, `Invalid`, `Notes`, `Tests`.

## 7. Activos machine-readable requeridos

1. `machine-readable-index.json` con:
   - lista de reglas
   - version
   - archivos fuente
   - estado (`stable`, `experimental`, `deprecated`)
2. `canonical-tests.jsonl` con casos de entrada/salida esperada.
3. `errors.json` con codigos de error y sugerencia de fix.

## 8. Roadmap de implementacion

## Fase 1 (MVP, 3-5 dias)

1. Crear estructura base de `web/docs/`.
2. Publicar Quickstart, Grammar, Toolchain.
3. Agregar 20 ejemplos canonicos.

## Fase 2 (Confiabilidad, 1 semana)

1. Agregar secciones `Valid/Invalid/Tests` en reglas clave.
2. Publicar catalogo de errores.
3. Publicar `machine-readable-index.json`.

## Fase 3 (Conformidad, 1 semana)

1. Enlazar doc con tests automaticos.
2. Agregar matrix de compatibilidad por version.
3. Agregar guias de migracion.

## Fase 4 (Paper/externo)

1. Version publica estable del spec.
2. Benchmark reproducible documentado.
3. FAQ tecnico y limites conocidos.

## 9. Definition of Done (DoD)

Una version de doc se considera lista cuando:

1. Todas las reglas core tienen `Syntax + Valid + Invalid + Tests`.
2. Hay al menos 1 ejemplo roundtrip por construccion core.
3. Existe changelog de la version publicada.
4. `machine-readable-index.json` esta actualizado.
5. Un contribuidor nuevo puede correr quickstart sin soporte directo.

## 10. Primer sprint recomendado (accionable hoy)

1. Crear `web/docs/README.md` como indice.
2. Mover el spec actual a `web/docs/02-grammar/syntax.md`.
3. Crear `web/docs/04-toolchain/transpiler.md` y `compiler.md`.
4. Crear `web/docs/07-llm/llm-contract.md`.
5. Publicar 10 casos canonicos en `canonical-examples.md`.

Si esta base queda lista, ya tienes "documentacion oficial usable" y no solo notas de proyecto.
