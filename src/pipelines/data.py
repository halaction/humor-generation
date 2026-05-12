import argparse
from dataclasses import dataclass

from src.config import config
from src.logging import get_logger
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
    references_split: str


class DataPipeline:
    def build(
        self,
        jokes_split: str = "train",
        embeddings_split: str = "train",
        keywords_split: str = "train",
        references_split: str = "train",
        resume: bool = True,
    ) -> DataBuildResult:
        JokesPipeline().build()
        EmbeddingsPipeline().build(split=embeddings_split, resume=resume)
        KeywordsPipeline().build(jokes_split=jokes_split, embeddings_split=embeddings_split, resume=resume)
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
            references_split=references_split,
            resume=resume,
        )
        return DataBuildResult(
            jokes_split=jokes_split,
            embeddings_split=embeddings_split,
            keywords_split=keywords_split,
            references_split=references_split,
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
    parser.add_argument("--references-split", default="train")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--publish", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    pipeline = DataPipeline()
    pipeline.build(
        jokes_split=args.jokes_split,
        embeddings_split=args.embeddings_split,
        keywords_split=args.keywords_split,
        references_split=args.references_split,
        resume=args.resume,
    )
    if args.publish:
        pipeline.publish(private=args.private)


if __name__ == "__main__":
    main()
