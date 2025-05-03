from src.preprocessing.preprocess import preprocess
from src.ml.train import train
from src.visualization.plot import plot

def main():
    preprocess()
    train()
    plot()

if __name__ == "__main__":
    main()