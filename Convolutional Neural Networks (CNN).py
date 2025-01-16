import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os  # Для проверки существования файла

# Загрузка данных Fashion-MNIST
(XTrain, yTrain), (XTest, yTest) = tf.keras.datasets.fashion_mnist.load_data()
print("x_train shape:", XTrain.shape, "y_train shape:", yTrain.shape)

# Приведение данных в формат для CNN
img_rows, img_cols = 28, 28
XTrain = XTrain.reshape(XTrain.shape[0], img_rows, img_cols, 1)
XTest = XTest.reshape(XTest.shape[0], img_rows, img_cols, 1)
XTrain = XTrain.astype("float32") / 255.0
XTest = XTest.astype("float32") / 255.0

# Параметры модели
input_shape = (img_rows, img_cols, 1)
num_classes = 10

# Визуализация данных
noOfClasses = yTrain.max() + 1
im = []
fig = plt.figure()
for i in range(noOfClasses):
    ax = fig.add_subplot(2, 5, i + 1, xticks=[], yticks=[])
    first = np.flatnonzero(yTrain == i)[0]
    im.append(XTrain[first, :, :, 0])
    ax.set_title(i)
    ax.imshow(im[i], cmap="gray")
plt.tight_layout()
plt.show()

# Создание модели CNN
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.35),
    tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.35),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Компиляция модели
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Путь к файлу с весами
weights_path = "/Users/denis/Documents/Python/Maschinelles Lernen/P3Shegay/Praktikum2.weights.h5"

# Проверка наличия файла с весами
if os.path.exists(weights_path):
    print(f"Weights file found: {weights_path}. Loading weights...")
    model.load_weights(weights_path)
else:
    print(f"Weights file not found: {weights_path}. Training the model...")
    # Обучение модели, если веса не найдены
    history = model.fit(XTrain, yTrain, batch_size=128, epochs=10, validation_split=0.2, verbose=1)
    model.save_weights(weights_path)
    print(f"Weights saved to {weights_path}")

# Оценка модели на тестовых данных
test_loss, test_accuracy = model.evaluate(XTest, yTest, verbose=0)
print(f"Test Accuracy (evaluate): {test_accuracy * 100:.2f}%")

# Предсказания
predictions = model.predict(XTest)
predicted_classes = np.argmax(predictions, axis=1)

# Построение конфузионной матрицы
confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
for true_label, predicted_label in zip(yTest, predicted_classes):
    confusion_matrix[true_label, predicted_label] += 1

# Вывод конфузионной матрицы
print("\nConfusion Matrix:")
print(confusion_matrix)

# Визуализация конфузионной матрицы
fig, ax = plt.subplots(figsize=(8, 8))
ax.matshow(confusion_matrix, cmap=plt.cm.Blues, alpha=0.7)
for i in range(num_classes):
    for j in range(num_classes):
        ax.text(x=j, y=i, s=confusion_matrix[i, j], va='center', ha='center')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Интерпретация конфузионной матрицы
print("\nConfusion Matrix Interpretation:")
for i in range(num_classes):
    sorted_indices = np.argsort(confusion_matrix[i, :])[::-1]  # Сортируем в порядке убывания
    top_confused = sorted_indices[1]
    print(f"Class {i} is often confused with Class {top_confused} (True: {confusion_matrix[i, i]}, Confused: {confusion_matrix[i, top_confused]})")
