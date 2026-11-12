"""
test_chain.py - Pruebas del pipeline SIN llamar a las APIs (no gasta tokens).

Ejecutar:
    python -m unittest -v test_chain

Idea central: ModeloFalso es un modelo de chat de LangChain que, en vez de
llamar a una API, devuelve respuestas guionadas (una por llamada): una
correcta, una con la lista vacia, una cortada por tokens, una caida de red...
Asi se comprueba que la cadena REAL (prompt | with_structured_output |
validacion, con with_retry y with_fallbacks) se comporta como debe.
"""

import os
import unittest
from typing import Any
from unittest.mock import patch

import anthropic
import httpx2
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from chain import (
    ExtraccionError,
    build_chain,
    build_chat_models,
    build_prompt,
    process_text,
)
from schemas import ExtraccionTecnica, NivelCriticidad

VALIDO = {
    "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
    "nivel_de_criticidad": "alta",
    "resumen_tecnico": "API con cache en Redis y persistencia en PostgreSQL.",
}
LISTA_VACIA = {**VALIDO, "tecnologias": []}
TEXTO = "Una API en FastAPI con Redis y PostgreSQL que se cae en horas pico."

_REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def caida_de_red() -> Exception:
    return anthropic.APIConnectionError(request=_REQUEST)


def clave_invalida() -> Exception:
    response = httpx2.Response(401, request=_REQUEST)
    return anthropic.AuthenticationError("clave invalida", response=response, body=None)


class ModeloFalso(BaseChatModel):
    """Modelo de chat de juguete con respuestas guionadas.

    Cada elemento de `guion` es la respuesta a UNA llamada:
        dict                -> tool call con esos argumentos (respuesta normal)
        (dict, motivo)      -> idem, con ese finish_reason / stop_reason
        Exception           -> la lanza (simula un error de la API)
    Si hay mas llamadas que elementos, se repite el ultimo.
    """

    guion: list[Any]
    llamadas: int = 0

    @property
    def _llm_type(self) -> str:
        return "falso"

    def bind_tools(self, tools, **kwargs) -> "ModeloFalso":
        # with_structured_output() llama a bind_tools(); el falso no lo necesita.
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        paso = self.guion[min(self.llamadas, len(self.guion) - 1)]
        self.llamadas += 1
        if isinstance(paso, BaseException):
            raise paso
        args, motivo = paso if isinstance(paso, tuple) else (paso, "tool_use")
        mensaje = AIMessage(
            content="",
            tool_calls=[{"name": "ExtraccionTecnica", "args": args, "id": f"call_{self.llamadas}"}],
            response_metadata={"stop_reason": motivo},
        )
        return ChatResult(generations=[ChatGeneration(message=mensaje)])


def cadena(*modelos, errores_de_caida=(), max_intentos=3):
    # espera_exponencial=False: los reintentos son inmediatos (tests rapidos).
    return build_chain(
        list(modelos), errores_de_caida, max_intentos=max_intentos, espera_exponencial=False
    )


class PipelineTests(unittest.IsolatedAsyncioTestCase):

    # --- Caso feliz -----------------------------------------------------------

    async def test_devuelve_objeto_validado(self):
        modelo = ModeloFalso(guion=[VALIDO])
        resultado = await process_text(TEXTO, chain=cadena(modelo))
        self.assertIsInstance(resultado, ExtraccionTecnica)
        self.assertEqual(resultado.nivel_de_criticidad, NivelCriticidad.ALTA)
        self.assertEqual(resultado.model_dump(mode="json"), VALIDO)
        self.assertEqual(modelo.llamadas, 1)

    # --- with_retry: el modelo se recupera ----------------------------------

    async def test_reintenta_si_la_salida_no_cumple_el_esquema(self):
        malos = {
            "lista vacia": LISTA_VACIA,
            "solo comodines": {**VALIDO, "tecnologias": ["desconocido", "  "]},
            "criticidad fuera del enum": {**VALIDO, "nivel_de_criticidad": "urgente"},
            "falta un campo": {"tecnologias": ["Redis"], "nivel_de_criticidad": "baja"},
            "campo inventado": {**VALIDO, "extra": 1},
        }
        for nombre, malo in malos.items():
            with self.subTest(caso=nombre):
                modelo = ModeloFalso(guion=[malo, VALIDO])
                resultado = await process_text(TEXTO, chain=cadena(modelo))
                self.assertEqual(resultado.tecnologias, VALIDO["tecnologias"])
                self.assertEqual(modelo.llamadas, 2)

    async def test_finish_reason_cortado_se_reintenta_aunque_el_json_parezca_completo(self):
        # Anthropic dice "max_tokens"; OpenAI dice "length". Se revisa ANTES de
        # usar el objeto, aunque haya pasado la validacion.
        for motivo in ("max_tokens", "length"):
            with self.subTest(motivo=motivo):
                modelo = ModeloFalso(guion=[(VALIDO, motivo), VALIDO])
                await process_text(TEXTO, chain=cadena(modelo))
                self.assertEqual(modelo.llamadas, 2)

    async def test_los_logs_muestran_cada_intento(self):
        modelo = ModeloFalso(guion=[LISTA_VACIA, VALIDO])
        with self.assertLogs("pipeline", level="INFO") as logs:
            await process_text(TEXTO, chain=cadena(modelo))
        salida = "\n".join(logs.output)
        self.assertIn("Intento 1", salida)
        self.assertIn("Salida INVALIDA", salida)
        self.assertIn("Intento 2", salida)
        self.assertIn("Extraccion valida en 2 intento(s)", salida)

    # --- Prueba de estres: el modelo NO se recupera ---------------------------

    async def test_agotados_los_reintentos_lanza_error_controlado(self):
        modelo = ModeloFalso(guion=[LISTA_VACIA])
        with self.assertRaises(ExtraccionError) as ctx:
            await process_text(TEXTO, chain=cadena(modelo, max_intentos=3))
        self.assertEqual(modelo.llamadas, 3)
        self.assertIn("3 intento(s)", str(ctx.exception))

    async def test_texto_vacio_no_llama_al_modelo(self):
        modelo = ModeloFalso(guion=[VALIDO])
        for texto in ("", "   \n "):
            with self.assertRaises(ValueError):
                await process_text(texto, chain=cadena(modelo))
        self.assertEqual(modelo.llamadas, 0)

    # --- Respaldo (reutiliza el criterio del Modulo 1) ------------------------

    async def test_caida_del_principal_usa_el_respaldo(self):
        principal = ModeloFalso(guion=[caida_de_red()])
        respaldo = ModeloFalso(guion=[VALIDO])
        c = cadena(principal, respaldo, errores_de_caida=(anthropic.APIConnectionError,))
        resultado = await process_text(TEXTO, chain=c)
        self.assertEqual(resultado.tecnologias, VALIDO["tecnologias"])
        # Una caida NO se reintenta en la cadena (ya la reintento el SDK).
        self.assertEqual(principal.llamadas, 1)
        self.assertEqual(respaldo.llamadas, 1)

    async def test_clave_invalida_no_usa_el_respaldo(self):
        principal = ModeloFalso(guion=[clave_invalida()])
        respaldo = ModeloFalso(guion=[VALIDO])
        c = cadena(principal, respaldo, errores_de_caida=(anthropic.APIConnectionError,))
        with self.assertRaises(ExtraccionError):
            await process_text(TEXTO, chain=c)
        self.assertEqual(respaldo.llamadas, 0)


class PromptYConfigTests(unittest.TestCase):

    def test_prompt_solo_pide_el_texto(self):
        # {instrucciones_formato} ya viene fijado con partial().
        prompt = build_prompt()
        self.assertEqual(prompt.input_variables, ["texto"])
        system, human = prompt.format_messages(texto="log {con llaves}")
        self.assertIn("nivel_de_criticidad", system.content)
        self.assertIn("baja, media, alta", system.content)
        self.assertIn("log {con llaves}", human.content)

    def test_se_exige_al_menos_un_reintento(self):
        with self.assertRaises(ValueError):
            cadena(ModeloFalso(guion=[VALIDO]), max_intentos=1)

    @patch.dict(
        os.environ,
        {"ANTHROPIC_API_KEY": "fake-anthropic", "OPENAI_API_KEY": "fake-openai"},
    )
    def test_reutiliza_la_configuracion_del_modulo_1(self):
        modelos, errores = build_chat_models(provider="anthropic", fallback_provider="openai")
        self.assertIsInstance(modelos[0], ChatAnthropic)
        self.assertIsInstance(modelos[1], ChatOpenAI)
        self.assertIsNone(modelos[0].temperature)  # Anthropic no recibe temperature
        self.assertEqual(modelos[1].temperature, 0.0)
        self.assertIn(anthropic.RateLimitError, errores)
        self.assertIn(anthropic.APIConnectionError, errores)
        self.assertNotIn(anthropic.AuthenticationError, errores)


class EsquemaTests(unittest.TestCase):

    def test_limpia_tecnologias(self):
        datos = {**VALIDO, "tecnologias": [" Redis ", "redis", "", "n/a", "PostgreSQL"]}
        self.assertEqual(ExtraccionTecnica(**datos).tecnologias, ["Redis", "PostgreSQL"])

    def test_rechaza_salidas_invalidas(self):
        for datos in (LISTA_VACIA, {**VALIDO, "resumen_tecnico": "  corto "}):
            with self.subTest(datos=datos):
                with self.assertRaises(ValidationError):
                    ExtraccionTecnica(**datos)


if __name__ == "__main__":
    unittest.main(verbosity=2)
