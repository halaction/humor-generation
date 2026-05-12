from src.pipelines.data import DataPipeline


def test_data_pipeline_build_and_publish_orchestration(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class _FakeJokesPipeline:
        def build(self) -> None:
            calls.append(("jokes.build", {}))

        def publish(self, **kwargs) -> None:
            calls.append(("jokes.publish", kwargs))

    class _FakeEmbeddingsPipeline:
        def build(self, jokes_split: str, resume: bool) -> None:
            calls.append(("embeddings.build", {"jokes_split": jokes_split, "resume": resume}))

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
    monkeypatch.setattr("src.pipelines.data.DataPipeline._clear_derived_artifacts", lambda self: calls.append(("clear", {})))

    pipeline = DataPipeline()
    result = pipeline.build(resume=False)
    pipeline.publish(private=True)

    assert result.embeddings_split == "train"
    assert result.keywords_jokes_split == "train[:1500]"
    assert result.references_jokes_split == "train"
    assert [name for name, _ in calls] == [
        "clear",
        "jokes.build",
        "embeddings.build",
        "keywords.build",
        "references.build",
        "jokes.publish",
        "embeddings.publish",
        "keywords.publish",
        "references.publish",
    ]
    assert calls[2][1] == {"jokes_split": "train", "resume": False}
    assert calls[3][1] == {"jokes_split": "train[:1500]", "embeddings_split": "train", "resume": False}
    assert calls[4][1] == {
        "jokes_split": "train",
        "embeddings_split": "train",
        "keywords_split": "train",
        "resume": False,
    }
    assert all(payload.get("private") is True for _, payload in calls[5:])


def test_data_pipeline_resume_true_does_not_clear_index(monkeypatch) -> None:
    calls: list[str] = []

    class _FakeJokesPipeline:
        def build(self) -> None:
            calls.append("jokes")

    class _FakeEmbeddingsPipeline:
        def build(self, jokes_split: str, resume: bool) -> None:
            calls.append(f"embeddings:{jokes_split}:{resume}")

    class _FakeKeywordsPipeline:
        def build(self, jokes_split: str, embeddings_split: str, resume: bool) -> None:
            calls.append(f"keywords:{resume}")

    class _FakeReferencesPipeline:
        def build(self, jokes_split: str, embeddings_split: str, keywords_split: str, resume: bool) -> None:
            calls.append(f"references:{resume}")

    monkeypatch.setattr("src.pipelines.data.JokesPipeline", _FakeJokesPipeline)
    monkeypatch.setattr("src.pipelines.data.EmbeddingsPipeline", _FakeEmbeddingsPipeline)
    monkeypatch.setattr("src.pipelines.data.KeywordsPipeline", _FakeKeywordsPipeline)
    monkeypatch.setattr("src.pipelines.data.ReferencesPipeline", _FakeReferencesPipeline)
    monkeypatch.setattr("src.pipelines.data.DataPipeline._clear_derived_artifacts", lambda self: calls.append("clear"))

    DataPipeline().build(resume=True)
    assert calls == ["jokes", "embeddings:train:True", "keywords:True", "references:True"]
