import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from CARTDecisionTree import bDecision  Tree

# Параметры для модели и дерева решений
THRESHOLD = 0.1
X_DECIMALS = 5

# Пути к файлам
TRAIN_PATH = "/Users/denis/Documents/Python/Maschinelles Lernen/Trainingsset.csv"
TEST_PATH = "/Users/denis/Documents/Python/Maschinelles Lernen/Testset.csv"
ALL_DATA_PATH = "/Users/denis/Documents/Python/Maschinelles Lernen/AllData.csv"

def load_data():
    """Загружает тренировочный, тестовый и полный набор данных."""т157куж-=
    train_data = pd.read_csv(TRAIN_PATH)
    test_data = pd.read_csv(TEST_PATH)
    all_data = pd.read_csv(ALL_DATA_PATH)
    return train_data, test_data, all_data

def split_features_labels(data):
    """Разделяет данные на признаки и метки."""
    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values
    return X, y

def optimize_tree(X_train, y_train, X_test, y_test):
    """Оптимизирует модель CART по minLeafNodeSize."""
    best_accuracy = 0
    best_leaf_size = 0
    for minLeafNodeSize in range(1, 51):
        model = bDecisionTree(threshold=THRESHOLD, xDecimals=X_DECIMALS, minLeafNodeSize=minLeafNodeSize)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        errors = np.sum(y_pred != y_test)
        accuracy = np.mean(y_pred == y_test) * 100
        print(f'minLeafNodeSize = {minLeafNodeSize}, Ошибки = {errors}, Точность = {accuracy:.2f}%')
    print(f'Лучшая minLeafNodeSize: {best_leaf_size} с точностью: {best_accuracy:.2f}%')
    return best_leaf_size

def plot_feature_relationships(data, labels):
    """Построение scatter-плотов для анализа данных."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Scatterplots различных признаков для оценки группировки")

    feature_pairs = [
        (7, 12, "Nonflavanoid Phenols", "Proline"),
        (6, 9, "Flavanoids", "Color Intensity"),
        (0, 6, "Alcohol", "Flavanoids"),
        (0, 9, "Alcohol", "Color Intensity")
    ]
    
    colors = {1: 'red', 2: 'green', 3: 'blue'}
    markers = {1: 'o', 2: '^', 3: '*'}

    for i, (feat_x, feat_y, x_label, y_label) in enumerate(feature_pairs):
        ax = axs[i // 2, i % 2]
        for wine_type in np.unique(labels):
            subset = data[labels == wine_type]
            ax.scatter(subset.iloc[:, feat_x], subset.iloc[:, feat_y], c=colors[wine_type], marker=markers[wine_type], s=60, alpha=0.6, label=f'Weinsorte {wine_type}')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{x_label} vs. {y_label}")
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('/Users/denis/Documents/Python/Maschinelles Lernen/Aufgabe1.pdf')
    plt.show()

def main():
    train_data, test_data, all_data = load_data()
    X_train, y_train = split_features_labels(train_data)
    X_test, y_test = split_features_labels(test_data)
    X_all, y_all = split_features_labels(all_data)

    # Оптимизация модели
    best_leaf_size = optimize_tree(X_train, y_train, X_test, y_test)

    # Построение графиков
    plot_feature_relationships(all_data.iloc[:, 1:], y_all)

if __name__ == "__main__":
    main()
