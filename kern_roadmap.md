# KERN — Roadmap Completo
> Token-efficient programming language for LLMs · Oscar Hdz · 2025

---

## ANTES DE ESCRIBIR UNA LÍNEA DE CÓDIGO

### Paso 1 — Leer los papers relacionados (2-3 días)

Tienes que conocer el trabajo previo para diferenciarte. Lee estos tres:

- **SimPy** (ISSTA 2024) → https://arxiv.org/abs/2404.16333  
  El más cercano a tu idea. Léelo completo, anota qué no resuelven.

- **Token Sugar** (2025) → https://arxiv.org/abs/2512.08266  
  Enfoque diferente pero mismo problema. Ve qué benchmarks usan.

- **Token Efficiency Languages** → https://martinalderson.com/posts/which-programming-languages-are-most-token-efficient/  
  Datos duros de qué lenguajes son más eficientes ya hoy.

**Output esperado:** Un documento de 1 página con tus notas de:
- Qué resuelven ellos
- Qué NO resuelven
- Dónde entra Kern como diferenciador

---

### Paso 2 — Diseñar la gramática de Kern en papel (3-5 días)

Antes de tocar código, tienes que definir cómo se ve Kern. Esto lo haces tú, no el LLM.

**Las decisiones que debes tomar:**

```
# ¿Cómo se ve una función?
Python:  def add(a, b):
             return a + b

Kern:    fn add(a,b)=a+b        ← opción A
Kern:    ^add(a,b)->a+b         ← opción B
Kern:    add:fn(a,b)a+b         ← opción C
# Tú decides cuál. Debe ser unambigua y compacta.
```

**Constructs que tienes que definir uno por uno:**

| Python | Kern (tú decides) |
|--------|-------------------|
| `def nombre(args):` | `fn nombre(args)=` |
| `class Nombre:` | ??? |
| `if cond:` | ??? |
| `for x in lista:` | ??? |
| `while cond:` | ??? |
| `import modulo` | ??? |
| `return valor` | ??? |
| `try/except` | ??? |
| `async/await` | ??? |
| `lambda x: expr` | ??? |
| Decoradores `@` | ??? |
| f-strings | ??? |
| Type hints `x: int` | ??? |

**Reglas de diseño que debes seguir:**
1. Toda expresión debe ser reversible a Python sin ambigüedad
2. Menos caracteres = menos tokens (pero sin inventar símbolos raros que el tokenizer fragmenta)
3. Usa ASCII puro, evita unicode especial
4. El LLM tiene que poder generarlo predeciblemente

**Output esperado:** Un archivo `kern_grammar_spec.md` con cada construct definido y un ejemplo.

---

### Paso 3 — Validar la gramática contra tokenizers reales (1-2 días)

Antes de construir nada, valida que tu sintaxis realmente ahorra tokens.

```python
import tiktoken  # tokenizer de OpenAI

enc = tiktoken.get_encoding("cl100k_base")

python_code = 'def add(a, b):\n    return a + b'
kern_code   = 'fn add(a,b)=a+b'

print(f"Python: {len(enc.encode(python_code))} tokens")
print(f"Kern:   {len(enc.encode(kern_code))} tokens")
```

Hazlo con 20-30 ejemplos de diferente complejidad. Si no ahorras consistentemente más de 15%, replantea la gramática.

**Output esperado:** Una tabla con los resultados y el % de ahorro promedio.

---

## FASE 1 — EL TRANSPILADOR

### Paso 4 — Entender el módulo `ast` de Python (1 día)

Es la herramienta central del proyecto. Sin esto no puedes ni hablar con Opus de forma inteligente.

```python
import ast

code = """
def greet(name):
    return f"Hello {name}"
"""

tree = ast.parse(code)
print(ast.dump(tree, indent=2))
```

Corre ese código, mira el output, entiende qué nodos existen. Lee la documentación oficial: https://docs.python.org/3/library/ast.html

Los nodos más importantes que vas a necesitar manejar:
- `FunctionDef`, `AsyncFunctionDef`
- `ClassDef`
- `Return`, `Assign`, `AugAssign`
- `If`, `For`, `While`
- `Import`, `ImportFrom`
- `Call`, `Attribute`
- `Lambda`, `ListComp`, `DictComp`

**Output esperado:** Que puedas leer un AST dump y entender qué está pasando.

---

### Paso 5 — Construir el transpilador con Opus (1-2 semanas)

Aquí usas Claude Opus 4.6 con el contexto que ya tienes.

**El prompt estructurado que debes darle:**

```
Contexto: Estoy construyendo Kern, un lenguaje de programación 
compacto para LLMs que compila a Python. 

Gramática definida: [pega tu kern_grammar_spec.md]

Tarea: Construye un transpilador Python → Kern usando el módulo 
ast de Python. Debe ser:
1. Determinístico (mismo input = mismo output siempre)
2. Reversible (todo Kern debe poder volver a Python)
3. Que maneje estos constructs: [lista]

Ejemplos de input/output:
Input Python: def add(a,b): return a+b
Output Kern:  fn add(a,b)=a+b

Input Python: [ejemplo 2]
Output Kern:  [ejemplo 2]

Empieza con los 5 constructs más comunes y después expandimos.
```

**Itera así:**
- Semana 1: `FunctionDef`, `If`, `For`, `Return`, `Assign` → 80% del código real
- Semana 2: `Class`, `Import`, `Try/Except`, `Lambda`, edge cases

**Output esperado:** `kern_transpiler.py` que convierte Python básico a Kern y back.

---

### Paso 6 — El compilador inverso Kern → Python (3-4 días)

Igual de importante que el transpilador. Sin esto no hay supervisión humana.

Es básicamente el proceso inverso: parseas Kern con un parser simple (puedes usar `lark` o `pyparsing`) y generas Python válido.

```bash
pip install lark
```

Opus también te ayuda a generar este parser una vez que tienes la gramática definida.

**Output esperado:** `kern_compiler.py` que convierte Kern → Python ejecutable.

---

### Paso 7 — Test suite completo (3-4 días)

```python
# Cada test debe verificar el round-trip completo
def test_roundtrip(python_code):
    kern_code    = transpile(python_code)      # Python → Kern
    python_back  = compile_kern(kern_code)     # Kern → Python
    
    # Mismo AST = misma semántica
    assert ast.dump(ast.parse(python_code)) == ast.dump(ast.parse(python_back))
    
    # Menos tokens
    assert count_tokens(kern_code) < count_tokens(python_code)
```

Usa HumanEval como dataset de prueba, son 164 funciones Python reales y están públicas.

**Output esperado:** 90%+ de los casos de HumanEval pasan el round-trip sin errores.

---

## FASE 2 — EL BENCHMARK

### Paso 8 — Medir reducción de tokens (1 semana)

Con el transpilador funcionando, mides contra datasets reales.

**Datasets a usar:**
- **HumanEval** (164 problemas) → https://github.com/openai/human-eval
- **MBPP** (374 problemas) → https://huggingface.co/datasets/mbpp
- **StarCoderData** (sample de 10k archivos) → HuggingFace

**Tokenizers a probar:**
- `cl100k_base` (GPT-4)
- `llama` tokenizer
- `codegen` tokenizer

**Tabla de resultados que necesitas:**

| Dataset | Tokenizer | Python tokens | Kern tokens | Reducción % |
|---------|-----------|--------------|-------------|-------------|
| HumanEval | GPT-4 | X | Y | Z% |
| MBPP | GPT-4 | X | Y | Z% |
| ... | ... | ... | ... | ... |

**Output esperado:** Tabla con resultados reales. Si promedias >20% de reducción, tienes paper.

---

### Paso 9 — Preparar el Dataset de Entrenamiento (1 semana)

Este es el corazón del loop bootstrapped. La idea es simple: usas código Python que ya existe, lo pasas por el transpilador, y tienes miles de ejemplos Kern sin escribir nada a mano.

**¿De dónde sacas el Python?**

```bash
pip install datasets
```

```python
from datasets import load_dataset
from kern_transpiler import transpile

# StarCoderData — millones de archivos Python reales
dataset = load_dataset("bigcode/starcoderdata", 
                        data_files="python/train-00000.parquet",
                        split="train[:10000]")  # empieza con 10k

kern_examples = []
for item in dataset:
    python_code = item["content"]
    try:
        kern_code = transpile(python_code)
        kern_examples.append({
            "python": python_code,
            "kern": kern_code
        })
    except:
        pass  # ignora los que fallen

print(f"Ejemplos generados: {len(kern_examples)}")
# Output esperado: ~8,000-9,000 ejemplos válidos
```

**Formato del dataset para entrenamiento:**

Para que el LLM aprenda a generar Kern dado un problema en lenguaje natural usas HumanEval + MBPP que ya tienen descripción en texto + solución en Python. Transpiles la solución y tienes el par instrucción → Kern.

```json
{
  "instruction": "Write a function that adds two numbers",
  "output": "fn add(a,b)=a+b"
}
```

**Output esperado:** Archivo `kern_dataset.jsonl` con mínimo 5,000 ejemplos limpios.

---

### Paso 10 — Fine-tuning del LLM con QLoRA (2-3 semanas)

**¿Por qué QLoRA y no fine-tuning normal?**

Fine-tuning completo de un modelo 7B requiere 80GB de VRAM y cuesta cientos de dólares. QLoRA congela el modelo base y solo entrena una capa pequeña encima. Necesitas 16-24GB de VRAM y cuesta $15-30 USD por experimento en RunPod.

**Setup completo:**

```bash
pip install transformers peft trl datasets bitsandbytes accelerate
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
import torch

# 1. Cargar modelo base en 4-bit (ahorra memoria)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    "codellama/CodeLlama-7b-hf",   # gratuito en HuggingFace
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")

# 2. Config LoRA — solo entrenas ~0.06% de los parámetros
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920 || ~0.06%

# 3. Dataset
from datasets import load_dataset
dataset = load_dataset("json", data_files="kern_dataset.jsonl", split="train")

def format_example(example):
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    }
dataset = dataset.map(format_example)

# 4. Entrenar
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="./kern-codellama-7b",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=50,
    ),
    dataset_text_field="text",
    max_seq_length=512,
)
trainer.train()
trainer.save_model("./kern-codellama-7b-final")
```

**¿Dónde corres esto? — RunPod**

```
1. runpod.io → renta RTX 4090 (24GB VRAM) → ~$0.44/hr
2. Imagen: runpod/pytorch:2.1.0-py3.10-cuda11.8.0
3. Sube tu dataset y corre el script
4. 3 epochs con 5k ejemplos → ~4-6 horas → ~$3 USD
```

**Output esperado:** Modelo `kern-codellama-7b-final` que genera Kern nativo dado un prompt en lenguaje natural.

---

### Paso 11 — Evaluar si el modelo aprendió Kern (3-4 días)

De nada sirve entrenar si no mides. Tienes que demostrar dos cosas para el paper:

**Prueba 1 — ¿Genera Kern sintácticamente válido?**

```python
from transformers import pipeline
from kern_compiler import compile_kern

generator = pipeline("text-generation", model="./kern-codellama-7b-final")

prompt = "### Instruction:\nWrite a function that reverses a string\n\n### Response:\n"
output = generator(prompt, max_new_tokens=100)[0]["generated_text"]
kern_code = output.split("### Response:\n")[1].strip()

try:
    python_code = compile_kern(kern_code)  # Kern → Python
    print(f"VÁLIDO: {kern_code}")
except:
    print("INVÁLIDO")
```

**Prueba 2 — Pass@k en HumanEval (el benchmark estándar del campo)**

```bash
pip install human-eval
```

```python
from human_eval.data import write_jsonl, read_problems

problems = read_problems()
samples = []

for task_id, problem in problems.items():
    prompt = f"### Instruction:\n{problem['prompt']}\n\n### Response:\n"
    output = generator(prompt, max_new_tokens=200)[0]["generated_text"]
    kern_code = output.split("### Response:\n")[1]
    
    try:
        python_code = compile_kern(kern_code)
    except:
        python_code = "def solution(): pass"
    
    samples.append({"task_id": task_id, "completion": python_code})

write_jsonl("kern_samples.jsonl", samples)
# evaluate_functional_correctness kern_samples.jsonl
```

**La tabla que necesitas para el paper:**

| Modelo | Pass@1 | Tokens promedio | Reducción |
|--------|--------|-----------------|-----------|
| CodeLlama-7B base (Python) | X% | X | — |
| CodeLlama-7B fine-tuned Kern | X% | X | -Y% |

Si Pass@1 de Kern es comparable al de Python Y los tokens son menores → tienes el resultado central del paper. Eso es todo lo que necesitas demostrar.

**Output esperado:** Tabla con números reales comparando Python vs Kern.

---

## FASE 3 — EL PAPER

### Paso 10 — Estructura del paper (1 semana de escritura)

**Estructura estándar para venues como MSR / SANER:**

```
1. Abstract (250 palabras)
   - Problema, propuesta, resultados clave

2. Introduction (1 página)
   - Motivación, problema, contribuciones

3. Background & Related Work (1 página)
   - SimPy, Token Sugar, trabajo previo
   - Cómo Kern se diferencia

4. Kern Language Design (1.5 páginas)
   - Gramática, principios de diseño
   - Ejemplos de sintaxis

5. Transpiler Architecture (1 página)
   - AST pipeline, casos especiales

6. Evaluation (2 páginas)
   - RQ1: ¿Cuánto reduce tokens?
   - RQ2: ¿Puede un LLM aprender Kern?
   - RQ3: ¿El loop bootstrapped mejora el modelo?

7. Discussion & Threats to Validity (0.5 página)

8. Conclusion & Future Work (0.5 página)
```

**Output esperado:** Draft completo del paper en LaTeX u Overleaf.

---

### Paso 11 — Subir a arXiv (1 día)

- Crea cuenta en https://arxiv.org
- Categoría: `cs.PL` (Programming Languages) o `cs.LG` (Machine Learning)
- Sube el PDF
- Tienes timestamp público de tu idea

Esto lo haces ANTES de submittear a congreso. Clava la bandera.

---

### Paso 12 — Submittear a congreso

**Venues recomendados en orden de dificultad:**

| Venue | Tipo | Deadline aprox | Dificultad |
|-------|------|---------------|------------|
| MSR 2026 | Mining Software Repositories | Enero | Media |
| SANER 2026 | Software Analysis & Evolution | Octubre | Media |
| ICSME 2026 | Software Maintenance | Abril | Media |
| ISSTA 2026 | Software Testing & Analysis | Enero | Alta |

---

## RESUMEN DE TIEMPOS

| Fase | Pasos | Tiempo estimado | Horas |
|------|-------|----------------|-------|
| Pre-código | 1-3 | 1-2 semanas | ~20 hrs |
| Transpilador | 4-7 | 3-4 semanas | ~60 hrs |
| Benchmark | 8-9 | 3-5 semanas | ~60 hrs |
| Paper | 10-12 | 2-3 semanas | ~40 hrs |
| **Total** | | **~4 meses** | **~180 hrs** |

Con **10 hrs/semana** → 4-5 meses  
Con **20 hrs/semana** → 2-2.5 meses

---

## STACK TECNOLÓGICO

```
Lenguaje principal:  Python 3.11+
AST parsing:         ast (stdlib)
Kern parser:         lark
Token counting:      tiktoken
Fine-tuning:         transformers + peft + trl
Datasets:            HumanEval, MBPP, StarCoderData
Compute:             RunPod / Modal (barato)
Paper:               Overleaf (LaTeX)
Repo:                GitHub público desde el día 1
Preprint:            arXiv cs.PL
```

---

## PRIMER PASO HOY MISMO

Antes de cualquier otra cosa, corre esto:

```python
import ast

code = """
def add(a, b):
    return a + b

for i in range(10):
    print(add(i, i+1))
"""

tree = ast.parse(code)
print(ast.dump(tree, indent=2))
```

Si entiendes el output de ese comando, ya puedes empezar el Paso 4.  
Si no lo entiendes, me lo mandas y lo vemos juntos.

---

*"The best time to start was yesterday. The second best time is now."*  
— Oscar Hdz, Kern 2025
