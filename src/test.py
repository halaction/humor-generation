from src.dataset.extraction import ExtractionPipeline

if __name__ == "__main__":
    pipeline = ExtractionPipeline()
    result = pipeline.run("I asked my dog what's two minus two. He said nothing.")
    print(result.model_dump())
