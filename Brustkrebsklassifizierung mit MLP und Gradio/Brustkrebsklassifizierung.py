import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers
from sklearn.datasets import load_breast_cancer
import gradio as gr

# 1. Данные
# Загрузка Breast Cancer Dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Перемешивание данных: Voreingenommenheit im Training verhindern, 
# Сбалансированное разбиение на train/val/test, Улучшение стабильности обучения, Минимизация последовательности зависимостей
#Kontrolle der Reproduzierbarkeit (Воспроизводимость)

##Перемешивание данных улучшает качественное (qualitativ) разбиение, 
#предотвращает смещение и гарантирует обобщающую способность модели, что делает обучение и валидацию более надёжными
np.random.seed(42)
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X, y = X[indices], y[indices]

# Разделение данных (70:15:15)
n_train = int(0.7 * len(X))
n_val = int(0.15 * len(X))

X_train, y_train = X[:n_train], y[:n_train]                                     # Первые 70%
X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]        # Следующие 15%
X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]           # Остаток, последние 15%

# Нормализация данных вручную
#Нормализация данных в данном коде означает приведение всех признаков (фич) к единому масштабу (einzige Skala zu bringen), 
# что позволяет улучшить качество и стабильность обучения нейронной сети. 

#Ускоряет и стабилизирует обучение
#Предотвращает доминирование признаков
#Обеспечивает корректную работу регуляризации
#Лучшее обобщение
X_train_mean, X_train_std = X_train.mean(axis=0), X_train.std(axis=0)
X_train = (X_train - X_train_mean) / X_train_std
X_val = (X_val - X_train_mean) / X_train_std
X_test = (X_test - X_train_mean) / X_train_std

# 2. Модель MLP с L2-регуляризацией и Early Stopping 
#early stop останавливает переобучение и возвращает лучшую валидационную производительность *(Leistung)
#MLP используется для классификации и регрессии, позволяя моделировать сложные зависимости между входными и выходными данными



#Это метод, который добавляет штраф к функции потерь за слишком большие веса модели.
#L2
#Уменьшить переобучение (overfitting),
#Сделать модель более устойчивой, ограничивая рост весов и улучшая обобщающую способность (generalization) на новых данных.
def create_model(l2_value, learning_rate, optimizer_name):
    model = models.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        #ReLU: Способствует быстрой сходимости обучения.
        # Не вызывает проблему затухающих(verblassender) градиентов, как сигмоид или tanh на скрытых слоях.
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.L2(l2_value)), #Вход
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.L2(l2_value)),  #Вход
        layers.Dense(1, activation='sigmoid') #Выход
        #Зачем используется sigmoid?
#Подходит для задач бинарной классификации, где нужно предсказать вероятность принадлежности к одному из двух классов 
# (например, класс 0 или 1).
    ])
    
    #64 и 32 – разумный компромисс между мощностью модели (способностью учиться сложным зависимостям) и переобучением.
    #Модель решает бинарную классификацию (классы 0 и 1).
    # Один нейрон на выходе с активацией sigmoid выдаёт вероятность принадлежности к классу 1.


    if optimizer_name == "Adam":
        optimizer = optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == "SGD":
        optimizer = optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_name == "RMSprop":
        optimizer = optimizers.RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError("Unsupported optimizer name")

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 3. Gradio интерфейс
# Функция для обучения модели и тестирования
def train_and_evaluate(l2_value, learning_rate, optimizer_name):
    model = create_model(l2_value, learning_rate, optimizer_name)

    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Обучение модели
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=50, batch_size=32, callbacks=[early_stopping], verbose=0)

    # Оценка на тренировочных данных
    train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)

    # Оценка на валидационных данных
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)

    # Оценка на тестовых данных
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    # Рассчитываем тестовую погрешность
    test_error = 100 - (test_accuracy * 100)

    if test_accuracy < 0.98:
        result_message = f"⚠️ Тестовая точность ниже 98% ({test_accuracy * 100:.2f}%). Попробуйте изменить параметры."
    else:
        result_message = f"✅ Тестовая точность достигнута: {test_accuracy * 100:.2f}%."

    return {
        "Тренировочная точность": f"{train_accuracy * 100:.2f}%",
        "Валидационная точность": f"{val_accuracy * 100:.2f}%",
        "Тестовая точность": f"{test_accuracy * 100:.2f}%",
        "Тестовый Loss": f"{test_loss:.4f}",
        "Тестовая погрешность": f"{test_error:.2f}%",
        "Результат": result_message
    }

# Функция для предсказаний
# 0 Gutartig (Доброкачественные)
# # Maligne *(Злокачественные)
model_cache = None
def predict_example(l2_value, learning_rate, optimizer_name, example_index):
    global model_cache

    if model_cache is None:
        model_cache = create_model(l2_value, learning_rate, optimizer_name)
        model_cache.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, verbose=0)

    example = X_val[example_index:example_index + 1]
    prediction = model_cache.predict(example)[0][0]
    predicted_class = 1 if prediction >= 0.5 else 0

    return f"Вероятность класса 1: {prediction:.2f}, Предсказанный класс: {predicted_class}, Истинный класс: {y_val[example_index]}"

# Создание Gradio интерфейса
with gr.Blocks() as demo:
    gr.Markdown("### Breast Cancer Classification with MLP")

    l2_value = gr.Slider(0.001, 0.1, step=0.001, value=0.01, label="L2 Регуляризация") #При низком - никакого эффекта(переобучение), при высоком - Недообучение
    learning_rate = gr.Slider(0.0001, 0.01, step=0.0001, value=0.001, label="Learning Rate") #При низком - медленный процесс обучения, при высоком - Нестабильность
    optimizer_name = gr.Dropdown(["Adam", "SGD", "RMSprop"], value="Adam", label="Оптимизатор")

    train_button = gr.Button("Обучить модель")
    results = gr.JSON(label="Результаты")

    train_button.click(train_and_evaluate, [l2_value, learning_rate, optimizer_name], results)

    gr.Markdown("### Предсказание для отдельного примера")
    example_index = gr.Slider(0, len(X_val) - 1, step=1, label="Индекс примера")
    predict_button = gr.Button("Предсказать")
    prediction_output = gr.Text(label="Результат предсказания")

    predict_button.click(predict_example, [l2_value, learning_rate, optimizer_name, example_index], prediction_output)

# Запуск приложения
demo.launch()
