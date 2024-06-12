import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist         # MNIST - датасет рукописных цифр
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
#%matplotlib inline - для Google Colab

'''Курсовая работа по проектированию нейронной сети, распознающей
   рукописные цифры и числа.'''

#region Получение данных для выборки.
# Загрузка обучающей и тестовой выборок.
# train_images - изображения цифр обучающей выборрки
# train_labels - вектор значений, соответствующих определенной цифре
# test_images, test_labels - изображения и вектор тестовой выборки
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Нормализация входных данных, чтобы они были вещественными числами
# в диапазоне от 0 до 1.
train_images = train_images / 255.0
test_images = test_images / 255.0

# Преобразование выходных значений в векторы по категориям.
# На выходе каждая распознаваемая цифра будет занимать
# определенную позицию в векторе.
test_labels_orig = test_labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
#endregion

# Отображение первых 25-ти изображений из обучающей выборки.
plt.figure(figsize=(10,5))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images[i], cmap = plt.cm.binary)
plt.show()

#region Создание модели.
# Формирование нейросетевой модели и вывод структуры модели в терминал.
model = Sequential([
    Flatten(input_shape=(28, 28)),          # размер изображений - 28x28 pixels
    Dense(128, activation='relu'),          # скрытый слой из 128 нейронов с ф. активации = relu
    Dense(64, activation='relu'),           # скрытый слой из 64 нейронов с ф. активации = relu
    Dense(10, activation='softmax')         # выходной слой из 10 нейронов (по 1 нейрону на каждую цифру) и ф. активации = softmax
])
# Вывод структуры нейросети в терминал.
print(model.summary())

# Компиляция нейросети с оптимизацией по Adam
# и критерием "категориальная кросс-энтропия"
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Запуск процесса обучения.
model.fit(train_images, # данные для обучения
          train_labels, # 
          epochs=10,    # кол-во эпох обучения
          validation_data=(test_images, test_labels)) # данные для проверки
# Сохранить модель в файле с расширением .keras.
model.save('mnist_model.keras')

# После обучения нейросетевой модели подать на вход тестовую выборку.
model.evaluate(test_images, test_labels)
#endregion

#region Тестирование модели.
# Проверка распознавания цифр.
# n - индекс изображения цифры, которая подается на вход для распознавания.
n = 0
x = np.expand_dims(test_images[n], axis=0) # трехмерный тензор
res = model.predict(x)
print(res)
print(f"Распознанная цифра: {np.argmax(res)}")
plt.imshow(test_images[n], cmap=plt.cm.binary)
plt.show()

# Распознавание всей тестовой выборки.
pred = model.predict(test_images)
pred = np.argmax(pred, axis=1)
print(pred.shape)
# Сравнение распознанных цифр с цифрами из выборки.
print(pred[:20])
print(test_labels_orig[:20])

# Выделение неверных вариантов.
mask = pred == test_labels_orig
print(mask[:10])
x_false = test_images[~mask]
p_false = pred[~mask]
print(x_false.shape)

# Вывод первых 5 неверных результатов
for i in range(5):
    print("Значение нейросети: " + str(test_labels_orig[i]))
    plt.imshow(x_false[i], cmap=plt.cm.binary)
    plt.show()
#endregion