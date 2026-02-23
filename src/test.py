from src.dataset.extraction import ExtractionPipeline

if __name__ == "__main__":
    pipeline = ExtractionPipeline()

    jokes = [
        "I asked my dog what's two minus two. He said nothing.",
        "How does a crackhead lose weight? Diet Coke.",
        "He was a real gentlemen and always opened the fridge door for me.",
        "What happened to everyone’s money in 2008? Someone Madoff with it.",
        "So United Airlines just bought the naming rights to the NFL stadium in LA... Immediately a delay was announced, no word on the first beating and dragging through the aisles...",
        "What do you call a french racist? A beget! ",
        "What would a lesbian pirate say? Scissor me timbers!",
        "What do you call a fake noodle? An Impasta.",
        "What kind of wine goes in a broken glass? Chardonnay",
        "Was Einstein's theory good? Relatively.",
    ]

    for joke in jokes:
        result = pipeline.run(joke)

        print(joke)
        print(result.model_dump())
        print()
