from src.dataset.jokes import JokesDataset

if __name__ == "__main__":
    dataset = JokesDataset()

    dataset._collect_data()
