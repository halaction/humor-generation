import structlog


structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name).bind(logger=name)
