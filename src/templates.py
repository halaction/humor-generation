from jinja2 import Environment, FileSystemLoader, StrictUndefined

from src.paths import TEMPLATES_DIR

environment = Environment(
    loader=FileSystemLoader(TEMPLATES_DIR),
    autoescape=False,
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)
