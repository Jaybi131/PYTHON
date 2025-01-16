import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap  # Добавлен корректный импорт
from CARTDecisionTree import bDecisionTree

# Пути к данным
TRAIN_PATH = "/Users/denis/Documents/Python/Maschinelles Lernen/Trainingsset.csv"
TEST_PATH = "/Users/denis/Documents/Python/Maschinelles Lernen/Testset.csv"
ALL_DATA_PATH = "/Users/denis/Documents/Python/Maschinelles Lernen/AllData.csv"

def load_dataset(path, features):
    """Загружает данные из CSV и выделяет выбранные признаки и метки."""
    data = pd.read_csv(path)
    X = data.iloc[:, features].values
    y = data.iloc[:, 0].values
    return X, y

def train_model(X, y, threshold=0.1, x_decimals=5, min_leaf_size=5):
    """Создаёт и обучает модель CART."""
    model = bDecisionTree(threshold=threshold, xDecimals=x_decimals, minLeafNodeSize=min_leaf_size)
    model.fit(X, y)
    return model

def evaluate_model(model, X_test, y_test):
    """Оценивает модель и возвращает ошибки и точность."""
    y_pred = model.predict(X_test)
    errors = np.sum(y_pred != y_test)
    accuracy = np.mean(y_pred == y_test) * 100
    error_points = X_test[y_pred != y_test]
    return errors, accuracy, error_points

def create_meshgrid(X, step=0.005):
    """Создаёт сетку для визуализации классификационных областей."""
    x_min, x_max = X.min(), X.max()
    XX, YY = np.mgrid[x_min:x_max:step, x_min:x_max:step]
    X_grid = np.c_[XX.ravel(), YY.ravel()]
    return XX, YY, X_grid

def plot_decision_boundary(model, X_test, y_test, XX, YY, Z, error_points):
    """Визуализирует области классификации и тестовые точки."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Новая цветовая схема
    mesh = ax.pcolormesh(XX, YY, Z, cmap='coolwarm', alpha=0.8)  # Изменённая цветовая карта

    # Цвета для классов
    class_colors = ['purple', 'orange', 'cyan']
    scatter1 = ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=ListedColormap(class_colors), edgecolor='black', s=50, alpha=1, marker='o')

    # Ошибочные точки выделены чёрными треугольниками
    scatter2 = ax.scatter(error_points[:, 0], error_points[:, 1], color="black",edgecolor='black', s=50, alpha=1, marker='^')

    plt.colorbar(mesh)
    plt.title('Классификационные области с тестовыми данными')
    plt.xlabel('Признак 1: Flavanoids')
    plt.ylabel('Признак 2: Color Intensity')
    plt.savefig('/Users/denis/Documents/Python/Maschinelles Lernen/Aufgabe2.pdf')
    plt.show()

def main():
    # Определяем лучшие признаки
    best_features = [7, 10]

    # Загружаем обучающую и тестовую выборки
    X_train, y_train = load_dataset(TRAIN_PATH, best_features)
    X_test, y_test = load_dataset(TEST_PATH, best_features)

    # Обучаем модель
    model = train_model(X_train, y_train)

    # Оцениваем модель
    errors, accuracy, error_points = evaluate_model(model, X_test, y_test)
    print(f"Ошибок с двумя лучшими признаками: {errors}")
    print(f'Точность с двумя лучшими признаками: {accuracy:.2f}%')

    # Создаём сетку и предсказываем для визуализации
    XX, YY, X_grid = create_meshgrid(X_train)
    Z = model.predict(X_grid).reshape(XX.shape)

    # Визуализация
    plot_decision_boundary(model, X_test, y_test, XX, YY, Z, error_points)

if __name__ == "__main__":
    main()
