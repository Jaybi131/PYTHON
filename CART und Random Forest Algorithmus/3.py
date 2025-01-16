import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from CARTDecisionTree import bDecisionTree

# Пути к данным
TRAIN_PATH = "/Users/denis/Documents/Python/Maschinelles Lernen/Trainingsset.csv"
TEST_PATH = "/Users/denis/Documents/Python/Maschinelles Lernen/Testset.csv"

# Параметры для дерева решений
THRESHOLD = 0.1
X_DECIMALS = 5

def load_data():
    """Загружает тренировочные и тестовые данные."""
    train_data = pd.read_csv(TRAIN_PATH)
    test_data = pd.read_csv(TEST_PATH)
    return train_data, test_data

def split_features_labels(data):
    """Разделяет данные на признаки и метки."""
    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values
    return X, y

def train_forest(X_train, y_train, X_test, y_test, n_trees=50):
    """Обучает случайный лес с количеством деревьев от 1 до n_trees и возвращает список ошибок."""
    errors = []
    for n in range(1, n_trees + 1):
        predictions = []
        
        # Построение случайного леса
        for _ in range(n):
            # Строим дерево с выборкой признаков и данных
            bootstrap_indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
            X_bootstrap = X_train[bootstrap_indices]
            y_bootstrap = y_train[bootstrap_indices]
            
            # Создание и обучение модели
            tree = bDecisionTree(threshold=THRESHOLD, xDecimals=X_DECIMALS, minLeafNodeSize=5)
            tree.fit(X_bootstrap, y_bootstrap)
            predictions.append(tree.predict(X_test))
        
        # Мажоритарное голосование для получения итогового предсказания
        y_pred = np.array(predictions).T
        final_pred = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=1, arr=y_pred)
        
        # Подсчёт ошибок
        error_count = np.sum(final_pred != y_test)
        errors.append(error_count)
        print(f"Количество деревьев: {n}, Ошибки: {error_count}")
    
    return errors

def plot_errors(errors):
    """Построение графика ошибок в зависимости от количества деревьев."""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(errors) + 1), errors, marker='o', color='b', linestyle='-')
    plt.xlabel('Количество деревьев')
    plt.ylabel('Количество ошибок')
    plt.title('Ошибки как функция количества деревьев в случайном лесу')
    plt.grid(True)
    plt.savefig('/Users/denis/Documents/Python/Maschinelles Lernen/Aufgabe3.pdf')
    plt.show()

def main():
    # Загрузка данных и разделение на признаки и метки
    train_data, test_data = load_data()
    X_train, y_train = split_features_labels(train_data)
    X_test, y_test = split_features_labels(test_data)
    
    # Обучение случайного леса и подсчёт ошибок
    errors = train_forest(X_train, y_train, X_test, y_test, n_trees=50)
    
    # Построение графика
    plot_errors(errors)

if __name__ == "__main__":
    main()
