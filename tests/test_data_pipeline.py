from src.pipelines.data import DataPipeline


def test_data_pipeline_build_and_publish_orchestration(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class _FakeJokesPipeline:
        def build(self) -> None:
            calls.append(("jokes.build", {}))

        def publish(self, **kwargs) -> None:
            calls.append(("jokes.publish", kwargs))

    class _FakeEmbeddingsPipeline:
        def build(self, split: str, resume: bool) -> None:
            calls.append(("embeddings.build", {"split": split, "resume": resume}))

        def publish(self, **kwargs) -> None:
            calls.append(("embeddings.publish", kwargs))

    class _FakeKeywordsPipeline:
        def build(self, jokes_split: str, embeddings_split: str, resume: bool) -> None:
            calls.append(
                (
                    "keywords.build",
                    {"jokes_split": jokes_split, "embeddings_split": embeddings_split, "resume": resume},
                )
            )

        def publish(self, **kwargs) -> None:
            calls.append(("keywords.publish", kwargs))

    class _FakeReferencesPipeline:
        def build(
            self,
            jokes_split: str,
            embeddings_split: str,
            keywords_split: str,
            resume: bool,
        ) -> None:
            calls.append(
                (
                    "references.build",
                    {
                        "jokes_split": jokes_split,
                        "embeddings_split": embeddings_split,
                        "keywords_split": keywords_split,
                        "resume": resume,
                    },
                )
            )

        def publish(self, **kwargs) -> None:
            calls.append(("references.publish", kwargs))

    monkeypatch.setattr("src.pipelines.data.JokesPipeline", _FakeJokesPipeline)
    monkeypatch.setattr("src.pipelines.data.EmbeddingsPipeline", _FakeEmbeddingsPipeline)
    monkeypatch.setattr("src.pipelines.data.KeywordsPipeline", _FakeKeywordsPipeline)
    monkeypatch.setattr("src.pipelines.data.ReferencesPipeline", _FakeReferencesPipeline)

    pipeline = DataPipeline()
    result = pipeline.build(
        jokes_split="train",
        embeddings_split="train[:1000]",
        keywords_split="train",
        references_split="train",
        resume=False,
    )
    pipeline.publish(private=True)

    assert result.embeddings_split == "train[:1000]"
    assert [name for name, _ in calls] == [
        "jokes.build",
        "embeddings.build",
        "keywords.build",
        "references.build",
        "jokes.publish",
        "embeddings.publish",
        "keywords.publish",
        "references.publish",
    ]
    assert calls[1][1] == {"split": "train[:1000]", "resume": False}
    assert calls[2][1] == {"jokes_split": "train", "embeddings_split": "train[:1000]", "resume": False}
    assert calls[3][1] == {
        "jokes_split": "train",
        "embeddings_split": "train[:1000]",
        "keywords_split": "train",
        "resume": False,
    }
    assert all(payload.get("private") is True for _, payload in calls[4:])
