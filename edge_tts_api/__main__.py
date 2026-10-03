"""Run the edge-tts batch synthesis service.

Usage: python -m edge_tts_api [--host HOST] [--port PORT]
"""

import argparse
import logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=None, help="Bind host")
    parser.add_argument("--port", type=int, default=None, help="Bind port")
    parser.add_argument("--concurrency", type=int, default=None, help="Synthesis workers")
    args = parser.parse_args()

    from . import config

    host = args.host or config.HOST
    port = args.port or config.PORT
    if args.concurrency:
        config.CONCURRENCY = args.concurrency

    logging.basicConfig(level=logging.INFO)

    import uvicorn

    uvicorn.run(
        "edge_tts_api.app:app",
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
