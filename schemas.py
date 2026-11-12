"""
schemas.py - Esquemas de datos del proyecto (modelos Pydantic).

Aca viven SOLO las estructuras de datos: que entra al modelo y que sale
(tanto la respuesta valida como los errores). No hay logica de red ni de
proveedores; por eso este archivo no importa openai, anthropic ni langchain.

Mapa de la consigna dentro de este archivo:
    Modulo 1 / PUNTO 2 -> GenerationParams (validacion de parametros de entrada).
    Modulo 1 / PUNTO 5 -> LLMError (error controlado devuelto como objeto).
    Pre-entrega 2      -> NivelCriticidad + ExtraccionTecnica (contrato de salida
                          del pipeline de extraccion, ver chain.py).
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# Tipos de error posibles. Se define una sola vez y se reutiliza en
# LLMError y en providers.py (classify_error), asi no hay que repetirlo.
ErrorKind = Literal["RATE_LIMIT", "NETWORK", "SERVER", "PROVIDER"]


# =============================================================================
# PUNTO 2 - Esquema Pydantic para validar los parametros de entrada del modelo.
# =============================================================================
class GenerationParams(BaseModel):
    # Pydantic valida los datos cuando se crea GenerationParams(...).
    # Si un dato no cumple, se lanza una ValidationError ANTES de llamar al
    # proveedor: no se gasta una peticion (ni dinero) con datos invalidos.
    #
    # Field(...) declara las reglas de cada campo:
    #   ge = mayor o igual   | gt = mayor estricto
    #   le = menor o igual   | min_length = largo minimo del texto

    # Texto que escribe el usuario. Obligatorio (no tiene default).
    prompt: str = Field(min_length=1, description="Texto enviado al modelo")

    # Instruccion de sistema: define el "rol" o comportamiento del modelo.
    system: str | None = Field(default=None, description="Instruccion de sistema opcional")

    # PUNTO 2 - "temperatura de 0 a 2": ge=0.0 y le=2.0 cumplen exactamente eso.
    # Valores bajos = respuestas mas deterministas; altos = mas creativas.
    # IMPORTANTE: solo la usa OpenAI. El SDK de Anthropic instalado no acepta
    # temperature, pero se valida igual porque si Anthropic se cae, el
    # respaldo (OpenAI) la necesita.
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    # PUNTO 2 - "max_tokens": limite de largo de la respuesta. Debe ser > 0.
    # Este si lo usan ambos proveedores.
    max_tokens: int = Field(default=500, gt=0, le=100000)

    # Parametro extra (el "etc." de la consigna): nucleus sampling, entre 0 y 1.
    # Igual que temperature: solo lo usa OpenAI.
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)

    # Validadores personalizados: corren DESPUES de las reglas de Field.
    # min_length=1 acepta "   " (tres espacios), por eso hace falta este
    # validador extra que limpia espacios y rechaza el texto vacio.
    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("El prompt no puede estar vacio.")
        return value

    # Si system llega vacio o con solo espacios, se normaliza a None
    # para que mas adelante no se envie un mensaje de sistema vacio.
    @field_validator("system")
    @classmethod
    def validate_system(cls, value: str | None) -> str | None:
        return value.strip() if value and value.strip() else None


# =============================================================================
# PUNTO 5 - Error controlado: un OBJETO, no un texto suelto.
# =============================================================================
class LLMError(BaseModel):
    # generate() entrega textos (str) o, si algo falla, UN LLMError.
    # Quien consume pregunta: isinstance(fragmento, LLMError), sin tener que
    # leer el texto para adivinar si es un error.
    #
    # Para un AGENTE esto es clave: puede decidir que hacer segun el tipo de
    # error (esperar, avisar, cambiar de herramienta...) en vez de solo mostrarlo.
    #
    # Alcance: LLMError cubre los errores que vienen de la API. Los errores de
    # configuracion (clave faltante) y los bugs de nuestro codigo NO se
    # convierten en LLMError (ver README, "Que errores se controlan").
    kind: ErrorKind
    provider: str
    message: str

    def __str__(self) -> str:
        # Formato legible para mostrar en la terminal.
        return f"[ERROR:{self.kind}] {self.provider}: {self.message}"


# =============================================================================
# PRE-ENTREGA 2 - Contrato de salida del pipeline de extraccion tecnica.
# =============================================================================
class NivelCriticidad(str, Enum):
    # Enum pedido por la consigna: solo estos tres valores son validos.
    # Hereda de str para que en el JSON aparezca "alta" y no "NivelCriticidad.ALTA".
    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"


# Valores de relleno que NO cuentan como tecnologia (ver limpiar_tecnologias).
_COMODINES = {
    "indeterminado", "indeterminada", "desconocido", "desconocida", "ninguno",
    "ninguna", "no especificado", "no especificada", "n/a", "na", "none",
    "unknown", "otro", "otros", "-",
}


class ExtraccionTecnica(BaseModel):
    """Entidades tecnicas extraidas de un texto (log de error, descripcion de arquitectura...)."""

    # with_structured_output() convierte esta clase en un JSON Schema y se lo
    # envia al modelo: los "description" de cada campo son INSTRUCCIONES que el
    # LLM lee. Despues, la respuesta se valida con estas mismas reglas.
    #
    # extra="forbid": si el modelo inventa un campo que no existe, es un error.
    model_config = ConfigDict(extra="forbid")

    # Restriccion pedida en "Define tu contrato": la lista no puede estar vacia.
    # A proposito NO se usa Field(min_length=1): eso viajaria al JSON Schema y
    # el modelo, obligado a poner algo, inventaria un relleno ("indeterminado")
    # ante un texto ambiguo. Asi el modelo puede ser honesto (devolver []) y es
    # el validador de abajo el que rechaza la lista vacia -> reintento -> error
    # controlado. Ver la "prueba de estres" en prueba_pipeline.py.
    tecnologias: list[str] = Field(
        description=(
            "Tecnologias, lenguajes, frameworks, bases de datos o servicios "
            "mencionados en el texto. Nombres propios, sin repetir. Lista vacia "
            "si el texto no menciona ninguna."
        ),
    )
    nivel_de_criticidad: NivelCriticidad = Field(
        description="Gravedad tecnica del problema o riesgo descrito: baja, media o alta."
    )
    resumen_tecnico: str = Field(
        description="Resumen tecnico de una o dos oraciones, en espanol."
    )

    # --- Validadores: corren DESPUES de que el modelo responde ----------------
    # Si alguno falla, with_structured_output reporta un error de parseo y la
    # logica de reintentos de chain.py vuelve a pedirle la respuesta al LLM.

    @field_validator("tecnologias")
    @classmethod
    def limpiar_tecnologias(cls, values: list[str]) -> list[str]:
        # Quita espacios, descarta vacios y duplicados (sin importar mayusculas),
        # respetando el orden en que aparecieron.
        limpias: list[str] = []
        vistas: set[str] = set()
        for value in values:
            value = value.strip()
            # Ante un texto ambiguo, el modelo tiende a "rellenar" la lista con
            # comodines como "indeterminado" para cumplir la regla de no vacia.
            # Eso no es una tecnologia: se descarta, asi la regla se cumple de verdad.
            if value.lower() in _COMODINES:
                continue
            if value and value.lower() not in vistas:
                vistas.add(value.lower())
                limpias.append(value)
        # Se comprueba DESPUES de limpiar: ["  "] o ["desconocido"] tambien
        # cuentan como lista vacia.
        if not limpias:
            raise ValueError("La lista de tecnologias no puede estar vacia.")
        return limpias

    @field_validator("resumen_tecnico")
    @classmethod
    def validar_resumen(cls, value: str) -> str:
        # Se valida aca (y no con min_length en Field) para no agregar al JSON
        # Schema restricciones que algunos proveedores no aceptan.
        value = value.strip()
        if len(value) < 10:
            raise ValueError("El resumen tecnico esta vacio o es demasiado corto.")
        return value
