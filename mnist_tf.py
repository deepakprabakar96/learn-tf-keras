import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
mnist = tf.keras.datasets.mnist

#load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#normalise data
x_train, x_test = x_train / 255.0, x_test / 255.0

#construct model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
#compile, fit and evaluate model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)

#output image and predictions
test_ind = 2435
first_imag = x_test[test_ind]
first_img = np.array(first_imag, dtype = 'float')
pixels = first_img.reshape((28,28))
plt.imshow(pixels, cmap='gray')
plt.show()
first_imag=first_imag.reshape(1, first_imag.shape[0], first_imag.shape[1])
y_pred = model.predict(first_imag) #returns array
y_pred_val = np.argmax(y_pred) #returns prediction value (index with max probability)
print(y_pred, y_test[test_ind])