"""
 TensorFlow
 はじめてのニューラルネットワーク：分類問題の初歩
 Basic classification: Classify images of clothing
    https://www.tensorflow.org/tutorials/keras/classification
 GitHub
    https://github.com/tomnt/tensorflow_tutorials
"""
from __future__ import absolute_import, division, print_function, unicode_literals
# TensorFlow と tf.keras のインポート / TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# ヘルパーライブラリのインポート / Helper libraries
import numpy as np
import matplotlib.pyplot as plt


def print_coordinates(image_coordinates: np.ndarray):
    """
    Print given image coordinates
    @:param: numpy.ndarray image_coordinates: Image coordinates to print
    """
    for raw in image_coordinates.tolist():
        column_no = 0
        print('[', end='')
        for depth in raw:
            if column_no < len(raw) - 1:
                print(str(depth).rjust(3), end=' ')
            else:
                print(str(depth).rjust(3) + ']')
            column_no += 1


print('tf.__version__: ', tf.__version__)

"""
 ファッションMNISTデータセットのロード
 Import the Fashion MNIST dataset
"""
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['0:T-shirt/top', '1:Trouser**', '2:Pullover**', '3:Drss', '4:Coat',
               '5:Sandal', '6:Shirt', '7:Sneaker', '8:Bag', '9:Ankle bot']
print('len(train_images):', len(train_images))
print('len(test_labels) :', len(test_labels))
print('train_labels:', len(train_labels), train_labels.tolist())
print('test_labels :', len(test_labels), test_labels.tolist())

"""
 データの前処理
 Preprocess the data
"""
# 単一画像の表示 / Show a single image
print('train_images[0].tolist()', train_images[0].tolist())  # Print image as list of integers

print_coordinates(train_images[0])

plt.figure('A single image')
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# 訓練用データセットの最初の25枚の画像を、クラス名付きで表示 / Display the first 25 images from the training set and display the class name
plt.figure('First 25 images from the training set and display the class name', figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

"""
 Build the model
 モデルの構築 / Set up the layers
"""
# 層の設定
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# モデルのコンパイル / Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# モデルの訓練 / Train the model
model.fit(train_images, train_labels, epochs=5)

"""
 正解率の評価
 Evaluate accuracy
"""
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

"""
 予測する
 Make predictions
"""
predictions = model.predict(test_images)
np.argmax(predictions[0])


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# 0番目の画像と、予測、予測配列を見てみましょう。/ Let's look at the 0th image, predictions, and prediction array.
i = 0
plt.figure(str(i) + 'th image, predictions, and prediction array', figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)
plt.show()

# X個のテスト画像、予測されたラベル、正解ラベルを表示します。 / Plot the first X test images, their predicted labels, and the true labels.
# 正しい予測は青で、間違った予測は赤で表示しています。 / Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure('Plot the first X test images, their predicted labels, and the true labels.',
           figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions, test_labels)
plt.show()

# テスト用データセットから画像を1枚取り出す / Grab an image from the test dataset.
img = test_images[0]
print('img.shape', img.shape)

# 画像を1枚だけのバッチのメンバーにする / Add the image to a batch where it's the only member.
img = (np.expand_dims(img, 0))
print('img.shape', img.shape)

# そして、予測を行います。/ Now predict the correct label for this image:
predictions_single = model.predict(img)
print('predictions_single', predictions_single)
plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()
