# Pre-entrega 2: Pipeline de extracción de entidades técnicas

Recibe un párrafo sin procesar (un log de error, una descripción de
arquitectura...) y devuelve un **objeto validado con Pydantic**:

```json
{
  "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
  "nivel_de_criticidad": "media",
  "resumen_tecnico": "La API desarrollada en FastAPI con caché en Redis y persistencia en PostgreSQL experimenta agotamiento del pool de conexiones a la base de datos durante horas pico, causando timeouts en peticiones concurrentes."
}
```

*(Salida real de `python prueba_pipeline.py` con `claude-sonnet-4-5`.)*

Está construido sobre el proyecto del Módulo 1 (`AsyncLLMManager`): reutiliza
su configuración (`.env`, claves, proveedor principal y de respaldo,
reintentos y timeout) en lugar de duplicarla.

## Estructura del proyecto

```text
schemas.py          -> QUE datos entran y salen: ExtraccionTecnica (Pre-entrega 2),
                       GenerationParams y LLMError (Modulo 1)
chain.py            -> Pre-entrega 2: prompt + cadena LCEL + reintentos + process_text()
prueba_pipeline.py  -> Pre-entrega 2: mini-script de prueba asincrono
test_chain.py       -> Pre-entrega 2: pruebas automaticas sin gastar tokens
providers.py        -> COMO se habla con cada API (Modulo 1 + create_chat_model())
llm_manager.py      -> Modulo 1: QUE proveedor usar y QUE hacer si falla
main.py             -> Modulo 1: script de streaming interactivo
test_fallback.py    -> Modulo 1: pruebas del respaldo
```

## Mapa de la consigna

| Consigna | Dónde está |
|---|---|
| Esquema Pydantic: `tecnologias` (lista), `nivel_de_criticidad` (enum baja/media/alta), `resumen_tecnico` | `schemas.py` → `ExtraccionTecnica`, `NivelCriticidad` |
| Restricciones del contrato (lista no vacía, etc.) | `schemas.py` → validadores `limpiar_tecnologias` y `validar_resumen` |
| `ChatOpenAI` o `ChatAnthropic` reutilizando el Módulo 1 | `providers.py` → `create_chat_model()`; `chain.py` → `build_chat_models()` |
| `ChatPromptTemplate` con texto de entrada + instrucciones de formato | `chain.py` → `build_prompt()` |
| Cadena LCEL `prompt \| model.with_structured_output(Schema)` | `chain.py` → `build_chain()` |
| Reintento automático con `.with_retry()` ante JSON mal formado o incompleto | `chain.py` → `build_chain()` + `ERRORES_REINTENTABLES` |
| Función async `process_text(text: str)` con `.ainvoke()` | `chain.py` → `process_text()` |
| Logs de validación y reintentos | `chain.py` → logger `pipeline` + `LogDeIntentos` |
| No ignorar el `finish_reason` | `chain.py` → `validar_respuesta()` |
| Sin f-strings en el prompt | `chain.py` → variables `{texto}` e `{instrucciones_formato}` del template |
| Prueba de estrés con texto ambiguo | `prueba_pipeline.py` (3.er ejemplo) y `test_chain.py` |
| Mini-script de prueba asíncrono | `prueba_pipeline.py` |

## Instalación

Requisitos: Python 3.12 o superior y una clave de OpenAI o Anthropic.

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> **Si usás Python 3.15:** `langsmith` (una dependencia interna de
> `langchain-core`, no pedida por la consigna) requiere `zstandard`, que
> todavía no publica una versión precompilada para 3.15 y falla al compilar
> sin Visual C++. `zstandard` solo se usa para comprimir trazas de LangSmith,
> que este proyecto no usa. Con Python 3.12–3.14 la instalación anterior
> funciona directo. En 3.15 se puede instalar así:
>
> ```powershell
> $env:PYYAML_FORCE_LIBYAML = "0"   # PyYAML en Python puro (sin compilar)
> python -m pip install pyyaml openai anthropic pydantic python-dotenv httpx2
> python -m pip install langsmith --no-deps
> python -m pip install anyio distro orjson requests requests-toolbelt sniffio "uuid-utils<1.0" websockets xxhash packaging
> python -m pip install certifi httpx jsonpatch langchain-protocol tenacity tiktoken
> python -m pip install --no-deps langchain-core langchain-openai langchain-anthropic
> ```
>
> `pip check` va a avisar que falta `zstandard`; es esperable y no afecta al pipeline.

> **Error "la ejecución de scripts está deshabilitada en este sistema"** al
> activar el entorno: ejecutá `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
> y volvé a activar, o usá directamente `.venv\Scripts\python.exe prueba_pipeline.py`.

## Configuración

Copiá `.env.example` como `.env` y completá las claves (`Copy-Item .env.example .env`).
Son las mismas variables del Módulo 1, más dos opcionales del pipeline:

| Variable | Para qué | Por defecto |
|---|---|---|
| `LLM_PROVIDER` | Proveedor principal: `anthropic` u `openai` | `anthropic` |
| `LLM_FALLBACK_PROVIDER` | Proveedor de respaldo (vacío = sin respaldo) | — |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | Claves de API | — |
| `ANTHROPIC_MODEL` / `OPENAI_MODEL` | Modelos | `claude-sonnet-4-5` / `gpt-4o-mini` |
| `LLM_MAX_RETRIES` | Reintentos del SDK ante 429, 5xx o red | `2` |
| `LLM_TIMEOUT` | Segundos máximos por petición | `60` |
| `PIPELINE_MAX_TOKENS` | Tokens máximos de la respuesta del pipeline | `1024` |
| `PIPELINE_MAX_INTENTOS` | Intentos totales de la cadena ante salida inválida (mínimo 2) | `3` |

Si hay respaldo configurado se necesitan las dos claves. `.env` está en
`.gitignore`: las claves nunca se suben al repositorio.

## Ejecución

```powershell
python prueba_pipeline.py                      # corre los 3 ejemplos incluidos
python prueba_pipeline.py "texto a analizar"   # analiza un texto propio
python prueba_pipeline.py --debug              # logs detallados de LangChain y HTTP
```

Uso desde código:

```python
import asyncio
from chain import process_text

resultado = asyncio.run(process_text("ERROR: Redis timeout en el worker de Celery"))
print(resultado.tecnologias, resultado.nivel_de_criticidad.value)
print(resultado.model_dump_json(indent=2))
```

### Ejemplo de logs

Caso normal (válido al primer intento):

```text
10:22:17 INFO    [pipeline] Modelos configurados: anthropic:claude-sonnet-4-5 -> openai:gpt-4o-mini
10:22:17 INFO    [pipeline] Procesando texto (252 caracteres)...
10:22:17 INFO    [pipeline] Intento 1 -> llamando a anthropic (claude-sonnet-4-5)
10:22:23 INFO    [pipeline] Intento 1 <- respuesta recibida (motivo de fin=tool_use, tokens in/out=1328/136)
10:22:23 INFO    [pipeline] Validacion Pydantic OK (motivo de fin=tool_use).
10:22:23 INFO    [pipeline] Extraccion valida en 1 intento(s) y 5.5 s.
```

Prueba de estrés: texto ambiguo, sin tecnologías
(`"Desde ayer algo anda raro, a veces tarda y otras no. Nadie toco nada."`):

```text
10:23:37 INFO    [pipeline] Intento 1 -> llamando a anthropic (claude-sonnet-4-5)
10:23:40 WARNING [pipeline] Salida INVALIDA (ValidationError): tecnologias: Value error, La lista de tecnologias no puede estar vacia.
10:23:42 INFO    [pipeline] Intento 2 -> llamando a anthropic (claude-sonnet-4-5)
10:23:45 WARNING [pipeline] Salida INVALIDA (ValidationError): tecnologias: Value error, La lista de tecnologias no puede estar vacia.
10:23:48 INFO    [pipeline] Intento 3 -> llamando a anthropic (claude-sonnet-4-5)
10:23:50 WARNING [pipeline] Salida INVALIDA (ValidationError): tecnologias: Value error, La lista de tecnologias no puede estar vacia.
10:23:50 ERROR   [pipeline] Sin salida valida despues de 3 intento(s): tecnologias: Value error, La lista de tecnologias no puede estar vacia.
[SIN RESULTADO VALIDO] El modelo no devolvio un objeto valido en 3 intento(s): ...
```

Respuesta cortada por tokens (`PIPELINE_MAX_TOKENS=40`):

```text
10:23:04 INFO    [pipeline] Intento 1 <- respuesta recibida (motivo de fin=max_tokens, tokens in/out=1297/40)
10:23:04 WARNING [pipeline] Respuesta CORTADA por limite de tokens (motivo=max_tokens).
10:23:05 INFO    [pipeline] Intento 2 -> llamando a anthropic (claude-sonnet-4-5)
```

## Cómo funciona

```text
{"texto": ...}
   │
   ▼
ChatPromptTemplate ── system: rol + criterios de criticidad + {instrucciones_formato}
   │                  human:  {texto}
   ▼
model.with_structured_output(ExtraccionTecnica, include_raw=True)
   │   el LLM responde siguiendo el JSON Schema generado desde Pydantic
   ▼
validar_respuesta ── 1) ¿finish_reason = length / max_tokens?   -> RespuestaIncompletaError
   │                 2) ¿JSON mal formado o no cumple el esquema? -> OutputParserException / ValidationError
   │                 3) ¿no hubo datos estructurados?             -> OutputParserException
   ▼
ExtraccionTecnica (validado)

Toda la cadena va envuelta en .with_retry(): ante los errores 1-3 se repite
completa (con espera exponencial). Si hay respaldo, .with_fallbacks() pasa al
otro proveedor cuando el principal se cae (rate limit, red, 5xx).
```

### Decisiones de diseño

- **`include_raw=True`.** `with_structured_output` solo devuelve el objeto, y
  así no se puede ver el `finish_reason`. Con `include_raw=True` devuelve también
  el mensaje crudo, y `validar_respuesta` revisa primero si la respuesta vino
  cortada y solo después usa el objeto (ver *Errores comunes a evitar*).
- **Qué se reintenta y qué no.** `.with_retry()` reintenta solo cuando el
  modelo respondió mal (JSON inválido, esquema incumplido, respuesta cortada).
  Las caídas de red, 429 y 5xx ya las reintenta el SDK (`LLM_MAX_RETRIES`, como
  en el Módulo 1) y, si persisten, entra el proveedor de respaldo. Una clave
  inválida no se reintenta ni activa el respaldo: es un error de configuración.
- **Lista de tecnologías no vacía, validada por Pydantic.** La regla está
  en el validador `limpiar_tecnologias` y no en el JSON Schema
  (`Field(min_length=1)`). En las pruebas, al exigirlo en el esquema, el modelo
  rellenaba la lista ante un texto ambiguo (`"indeterminado"`,
  `"Sin tecnologías específicas mencionadas"`). Ahora el modelo puede responder
  con honestidad (`[]`) y el validador lo rechaza; el pipeline reintenta y, si
  no se recupera, termina en un `ExtraccionError` controlado en lugar de
  devolver datos inventados. Además se descartan duplicados y comodines como
  `"desconocido"` o `"n/a"`.
- **Instrucciones de formato generadas desde el esquema.** `instrucciones_formato()`
  arma el texto a partir de las `description` de `ExtraccionTecnica`; si
  cambia el esquema, el prompt se actualiza solo. Se fijan con `.partial()`,
  así que al invocar solo hace falta `{"texto": ...}`.
- **Temperatura 0** (solo OpenAI; el SDK de Anthropic instalado no la acepta):
  extraer datos no es una tarea creativa.
- **Límite conocido.** Si la respuesta se corta por tokens, el reintento usa el
  mismo `PIPELINE_MAX_TOKENS`, así que es probable que vuelva a cortarse. El
  pipeline lo informa en los logs y termina en un error controlado; la solución
  es subir `PIPELINE_MAX_TOKENS`.

### Errores que devuelve `process_text()`

| Situación | Resultado |
|---|---|
| Salida válida (en cualquier intento) | `ExtraccionTecnica` |
| Texto vacío | `ValueError`, sin llamar al modelo |
| Reintentos agotados (JSON inválido, esquema incumplido, respuesta cortada) | `ExtraccionError` |
| Falla de la API (clave inválida, o caída del principal y del respaldo) | `ExtraccionError` |
| Falta una clave o `LLM_PROVIDER` inválido | `ValueError` al crear el pipeline |

## Pruebas sin gastar tokens

```powershell
python -m unittest -v test_chain test_fallback
```

`test_chain.py` reemplaza el LLM por un `ModeloFalso` con respuestas guionadas
y prueba la cadena real (prompt, `with_structured_output`, validación,
`with_retry` y `with_fallbacks`):

- Salida válida al primer intento → objeto `ExtraccionTecnica`.
- Lista vacía, solo comodines, criticidad fuera del enum, campo faltante o
  inventado → reintento y recuperación.
- `finish_reason` `length` (OpenAI) / `max_tokens` (Anthropic) → reintento.
- Reintentos agotados → `ExtraccionError` tras exactamente N llamadas.
- Caída del principal → respaldo; clave inválida → sin respaldo.
- Los logs muestran cada intento.

---

## Base reutilizada: Módulo 1 (`AsyncLLMManager`)

El Módulo 1 sigue funcionando igual (`python main.py`, streaming
interactivo). En esta entrega solo se le agregó a `providers.py` el método
`create_chat_model()`, que crea el `ChatOpenAI` / `ChatAnthropic` con la misma
clave, modelo, reintentos y timeout, y la propiedad `outage_errors`, que
agrupa los errores que activan el respaldo.

### Respaldo automatico (Anthropic -> OpenAI)

Si el proveedor principal se cae, primero el SDK reintenta solo
(`LLM_MAX_RETRIES` veces, esperando un poco más cada vez). Si sigue sin
responder, `generate()` pasa al proveedor de respaldo y muestra un aviso:

```text
[AVISO] anthropic no disponible (SERVER). Reintentando con openai...
```

| Situacion | Que pasa |
|---|---|
| Limite de tasa (429) | Usa el respaldo |
| Error de red o timeout | Usa el respaldo |
| Error del servidor 5xx / saturado (529) | Usa el respaldo |
| Clave invalida, peticion mal armada | `[ERROR:PROVIDER]`, sin respaldo (es un error de configuracion) |
| Se cae a mitad de la respuesta | `[ERROR:...]`, sin respaldo (evita texto duplicado) |
| El respaldo tambien falla | `[ERROR:RATE_LIMIT]`, `[ERROR:NETWORK]` o `[ERROR:SERVER]` |

Cuando entra OpenAI se usan `temperature` y `top_p` tal como se validaron en
`GenerationParams`.

### Guia rapida del codigo del Modulo 1

**`schemas.py`**

- `GenerationParams`: estructura Pydantic para validar `prompt`, `temperature`,
  `max_tokens`, `top_p` y `system`.
- `LLMError`: error controlado con `kind` (`RATE_LIMIT`, `NETWORK`, `SERVER`,
  `PROVIDER`), `provider` y `message`.

**`providers.py`**

- `BaseProvider`: clase base abstracta. Define lo que todo proveedor tiene que
  saber hacer: `create_client()`, `stream()` y `classify_error()`. También lee
  la clave y el modelo desde `.env`.
- `OpenAIProvider` y `AnthropicProvider`: implementan esos métodos con su SDK.
  `stream()` entrega fragmentos con `yield` y no captura errores: los deja subir
  al manager.
- `PROVIDERS`: diccionario `{"openai": OpenAIProvider, ...}`. Es el registro de
  proveedores que se pueden elegir desde `.env`.

**`llm_manager.py`**

- `AsyncLLMManager.__init__`: lee `LLM_PROVIDER`, `LLM_FALLBACK_PROVIDER`,
  `LLM_MAX_RETRIES` y `LLM_TIMEOUT`, busca cada proveedor en `PROVIDERS` y lo
  crea. No tiene ningún `if openai / else anthropic`.
- `AsyncLLMManager.generate`: recorre los proveedores en orden, reenvía los
  fragmentos mediante `yield` y decide si usar el respaldo. Entrega textos
  (`str`) y, si algo falla, un único `LLMError` al final.

**`main.py`**: pide el prompt, crea los parámetros, inicia el gestor y muestra
cada fragmento sin esperar la respuesta completa.

### Como agregar un proveedor nuevo

1. En `providers.py`, crear una clase que herede de `BaseProvider` (con
   `name`, `api_key_env`, `model_env`, `default_model`, las tuplas de errores,
   `create_client()`, `stream()` y, para usarlo en el pipeline,
   `create_chat_model()`).
2. Registrarla en el diccionario `PROVIDERS` (una línea).
3. Usarla desde `.env`: `LLM_PROVIDER=mistral` o `LLM_FALLBACK_PROVIDER=mistral`.

No hace falta tocar `llm_manager.py`, `main.py` ni `chain.py`.
`test_fallback.py` lo demuestra con `EchoProvider`, un proveedor de juguete.

### Que errores se controlan en el Modulo 1

| Error del SDK | `LLMError.kind` |
|---|---|
| `RateLimitError` | `RATE_LIMIT` |
| `APIConnectionError` / `APITimeoutError` | `NETWORK` |
| `InternalServerError` (incluye `OverloadedError` de Anthropic) | `SERVER` |
| Cualquier otro `APIError` (clave inválida, petición mal armada) | `PROVIDER` |

Los errores de configuración (clave faltante, `LLM_PROVIDER` inválido) se
detectan al iniciar con un `ValueError`, y los bugs de nuestro propio código se
propagan con su traceback en lugar de disfrazarse de error de la API.

### Nota sobre Anthropic

La version de Anthropic instalada (2.5.0) no acepta `temperature` ni `top_p`.
Esos valores se validan igual con Pydantic porque los usa OpenAI cuando actua
como respaldo.
