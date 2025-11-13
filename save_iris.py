from sklearn.datasets import load_iris
import csv

def save_iris_to_csv(path="iris.csv"):
    iris = load_iris()
    X = iris.data
    y = iris.target

    header = ["sepal_length", "sepal_width", "petal_length", "petal_width", "target"]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for xi, yi in zip(X, y):
            writer.writerow(list(xi) + [yi])

    print(f"Saved Iris dataset to {path}")

if __name__ == "__main__":
    save_iris_to_csv("iris.csv")
