import argparse
import shutil

from src.config import config
from src.logging import get_logger
from src.paths import DATA_DIR
from src.pipelines.embeddings import EmbeddingsPipeline
from src.pipelines.jokes import JokesPipeline
from src.pipelines.keywords import KeywordsPipeline
from src.pipelines.references import ReferencesPipeline

logger = get_logger(__name__)


class DataPipeline:
    @staticmethod
    def _clear_derived_artifacts() -> None:
        paths = [
            DATA_DIR / config.jokes.hf_config_name,
            DATA_DIR / config.embeddings.hf_config_name,
            DATA_DIR / config.keywords.hf_config_name,
            DATA_DIR / config.references.hf_config_name,
            DATA_DIR / config.references.index_dirname,
        ]
        for path in paths:
            if not path.exists():
                continue
            shutil.rmtree(path)
            logger.info("artifact_cache.cleared", path=str(path))

    def build(self, resume: bool = False):
        embeddings_jokes_split = config.embeddings.jokes_split
        keywords_jokes_split = config.keywords.jokes_split
        embeddings_split = config.keywords.embeddings_split
        references_jokes_split = config.references.jokes_split
        references_embeddings_split = config.references.embeddings_split
        keywords_split = config.references.keywords_split

        if not resume:
            self._clear_derived_artifacts()
        JokesPipeline().build()
        EmbeddingsPipeline().build(
            jokes_split=embeddings_jokes_split,
            resume=resume,
        )
        KeywordsPipeline().build(
            jokes_split=keywords_jokes_split,
            embeddings_split=embeddings_split,
            resume=resume,
        )
        ReferencesPipeline().build(
            jokes_split=references_jokes_split,
            embeddings_split=references_embeddings_split,
            keywords_split=keywords_split,
            resume=resume,
        )
        logger.info(
            "build.done",
            embeddings_jokes_split=embeddings_jokes_split,
            keywords_jokes_split=keywords_jokes_split,
            references_jokes_split=references_jokes_split,
            embeddings_split=embeddings_split,
            references_embeddings_split=references_embeddings_split,
            keywords_split=keywords_split,
            resume=resume,
        )

    def publish(self, private: bool = False) -> None:
        JokesPipeline().publish(config_name=config.jokes.hf_config_name, private=private)
        EmbeddingsPipeline().publish(config_name=config.embeddings.hf_config_name, private=private)
        KeywordsPipeline().publish(config_name=config.keywords.hf_config_name, private=private)
        ReferencesPipeline().publish(config_name=config.references.hf_config_name, private=private)
        logger.info("publish.done", private=private)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full data pipeline for finalized training references.")
    parser.add_argument("--build", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--publish", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    pipeline = DataPipeline()
    if args.build:
        pipeline.build(resume=args.resume)
    if args.publish:
        pipeline.publish(private=args.private)


if __name__ == "__main__":
    main()
