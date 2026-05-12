import argparse
import shutil
from dataclasses import dataclass

from src.config import config
from src.logging import get_logger
from src.paths import DATA_DIR
from src.pipelines.embeddings import EmbeddingsPipeline
from src.pipelines.jokes import JokesPipeline
from src.pipelines.keywords import KeywordsPipeline
from src.pipelines.references import ReferencesPipeline

logger = get_logger(__name__)


@dataclass
class DataBuildResult:
    jokes_split: str
    embeddings_split: str
    keywords_split: str


class DataPipeline:
    @staticmethod
    def _clear_reference_index_cache() -> None:
        index_dir = DATA_DIR / config.references.index_dirname
        if not index_dir.exists():
            return
        shutil.rmtree(index_dir)
        logger.info("index_cache.cleared", path=str(index_dir))

    def build(
        self,
        jokes_split: str = "train",
        embeddings_split: str = "train",
        keywords_split: str = "train",
        resume: bool = False,
    ) -> DataBuildResult:
        JokesPipeline().build()
        if not resume:
            self._clear_reference_index_cache()
        EmbeddingsPipeline().build(
            split=embeddings_split,
            resume=resume,
        )
        KeywordsPipeline().build(
            jokes_split=jokes_split,
            embeddings_split=embeddings_split,
            resume=resume,
        )
        ReferencesPipeline().build(
            jokes_split=jokes_split,
            embeddings_split=embeddings_split,
            keywords_split=keywords_split,
            resume=resume,
        )
        logger.info(
            "build.done",
            jokes_split=jokes_split,
            embeddings_split=embeddings_split,
            keywords_split=keywords_split,
            resume=resume,
        )
        return DataBuildResult(
            jokes_split=jokes_split,
            embeddings_split=embeddings_split,
            keywords_split=keywords_split,
        )

    def publish(self, private: bool = False) -> None:
        JokesPipeline().publish(config_name=config.jokes.hf_config_name, private=private)
        EmbeddingsPipeline().publish(config_name=config.embeddings.hf_config_name, private=private)
        KeywordsPipeline().publish(config_name=config.keywords.hf_config_name, private=private)
        ReferencesPipeline().publish(config_name=config.references.hf_config_name, private=private)
        logger.info("publish.done", private=private)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full data pipeline for finalized training references.")
    parser.add_argument("--jokes-split", default="train")
    parser.add_argument("--embeddings-split", default="train")
    parser.add_argument("--keywords-split", default="train")
    parser.add_argument("--publish", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    pipeline = DataPipeline()
    pipeline.build(
        jokes_split=args.jokes_split,
        embeddings_split=args.embeddings_split,
        keywords_split=args.keywords_split,
        resume=False,
    )
    if args.publish:
        pipeline.publish(private=args.private)


if __name__ == "__main__":
    main()
