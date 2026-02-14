from src.dataset import Dataset

if __name__ == "__main__":
    dataset = Dataset()

    dataset._download_short_jokes()
    dataset._download_r_jokes()
