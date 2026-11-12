"""
prueba_pipeline.py - Mini-script de prueba asincrono del pipeline (Pre-entrega 2).

Uso:
    python prueba_pipeline.py                     -> corre los 3 ejemplos de abajo
    python prueba_pipeline.py "texto a analizar"  -> analiza el texto indicado
    python prueba_pipeline.py --debug             -> logs mas detallados (de LangChain y HTTP)

Los ejemplos cubren:
    1) Una descripcion de arquitectura (el ejemplo de la consigna).
    2) Un log de error real.
    3) PRUEBA DE ESTRES: un texto ambiguo, sin tecnologias concretas. Sirve
       para ver si el validador lanza excepciones (lista vacia -> reintento) o
       si el modelo se recupera.
"""

import asyncio
import json
import logging
import sys

from chain import ExtraccionError, process_text

EJEMPLOS = {
    "arquitectura": (
        "Tenemos una API hecha con FastAPI que cachea las consultas frecuentes en "
        "Redis y persiste todo en PostgreSQL. Desde el lanzamiento, en horas pico "
        "el pool de conexiones a la base se agota y las peticiones concurrentes "
        "quedan esperando hasta dar timeout."
    ),
    "log de error": (
        "2026-11-12 03:14:07 ERROR [worker-3] celery.app.trace: Task "
        "orders.sync_stock raised unexpected: OperationalError('could not connect "
        "to server: Connection refused. Is the server running on host "
        "\"db-primary\" and accepting TCP/IP connections on port 5432?') -- "
        "kombu.exceptions: RabbitMQ channel closed, 1.243 mensajes en cola sin procesar."
    ),
    "texto ambiguo (estres)": (
        "Desde ayer algo anda raro, a veces tarda y otras no. Nadie toco nada."
    ),
}


def configurar_logs(debug: bool) -> None:
    # El formato muestra la hora y QUIEN escribe el log ("pipeline" = chain.py).
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.WARNING,
        format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    # Los logs del pipeline siempre en INFO: validacion, intentos, reintentos.
    logging.getLogger("pipeline").setLevel(logging.DEBUG if debug else logging.INFO)
    if not debug:
        # Cuando la respuesta viene cortada, el parser de LangChain imprime su
        # propio traceback. El pipeline ya lo informa en una linea; el detalle
        # completo queda disponible con --debug.
        logging.getLogger("langchain_core.output_parsers").setLevel(logging.CRITICAL)


async def analizar(nombre: str, texto: str) -> bool:
    print(f"\n=== {nombre} ===", flush=True)
    try:
        resultado = await process_text(texto)
    except ExtraccionError as exc:
        # Error CONTROLADO: el pipeline no se cae, informa que no pudo validar.
        print(f"[SIN RESULTADO VALIDO] {exc}", flush=True)
        return False
    # mode="json" convierte el Enum en su valor ("alta") para poder serializar.
    print(json.dumps(resultado.model_dump(mode="json"), ensure_ascii=False, indent=2), flush=True)
    return True


async def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--debug"]
    configurar_logs(debug="--debug" in sys.argv)

    try:
        if args:
            ok = await analizar("texto de entrada", " ".join(args))
            return 0 if ok else 1
        # Uno detras de otro para que los logs de cada ejemplo no se mezclen.
        # (process_text es async: tambien se podrian lanzar juntos con asyncio.gather.)
        for nombre, texto in EJEMPLOS.items():
            await analizar(nombre, texto)
        return 0
    except ValueError as exc:
        # Texto vacio o configuracion invalida (.env), igual que en main.py.
        print(f"Error de entrada o configuracion: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
