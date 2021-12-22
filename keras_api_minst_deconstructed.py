import keras as k
import tensorflow as tf
import matplotlib.pyplot as plt

(x_train, y_train), (x_val, y_val) = k.datasets.fashion_mnist.load_data()
print(x_val[0])

def preprocess(x, y):
    x = tf.cast(x, tf.float32) / 255.0 # cast converts datatype of x to tf.float32
    y = tf.cast(y, tf.int64)
    return x, y


def create_dataset(xs, ys, n_classes=10):
    ys = tf.one_hot(ys, depth=n_classes)
    return tf.data.Dataset.from_tensor_slices((xs, ys)) \
        .map(preprocess) \
        .shuffle(len(ys)) \
        .batch(128)


train_dataset = create_dataset(x_train, y_train)
val_dataset = create_dataset(x_val, y_val)

plt.imshow(tf.keras.layers.Conv2D(x_val[0], kernel_size=2))
plt.show()

# print(y_val[0])
#
# model = k.Sequential([
#     k.layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
#     k.layers.Dense(units=256, activation='relu'),
#     k.layers.Dense(units=192, activation='relu'),
#     k.layers.Dense(units=128, activation='relu'),
#     k.layers.Dense(units=10, activation='softmax')
# ])
#
# model.compile(optimizer='adam',
#               loss=tf.losses.CategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
#
# history = model.fit(
#     train_dataset.repeat(),
#     epochs=10,
#     steps_per_epoch=500,
#     validation_data=val_dataset.repeat(),
#     validation_steps=2
# )

predictions = model.predict(val_dataset)
print(predictions)
