import argparse
import logging
import sys
from dataclasses import asdict

from agent.agents import get_agents
from agent.config import get_config
from agent.logging import RunLogger, configure_logging
from agent.paths import BASE_PATH
from agent.pipeline import run_pipeline
from agent.settings import settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Humor generation agent")
    parser.add_argument("--model", type=str, default=None, help="Model name to use.")
    parser.add_argument("--input", type=str, default=None, help="Input topic or constraint.")
    parser.add_argument("--k", type=int, default=4, help="Number of associations/jokes to generate.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = get_config(args)

    run_path = BASE_PATH / "runs" / "latest"
    run_path.mkdir(parents=True, exist_ok=True)

    configure_logging(run_path)
    logger = logging.getLogger(__name__)
    logger.info("run directory=%s", run_path)

    agents = get_agents(config.model)

    run_logger = RunLogger(run_path)
    run_logger.log_step("settings", {"base_url": settings.base_url, "model_default": settings.model})
    run_logger.log_step("run_config", asdict(config))

    summary = run_pipeline(agents, config, run_logger)

    sys.stdout.write(f"{summary.selection.best_text}\n")


if __name__ == "__main__":
    main()
