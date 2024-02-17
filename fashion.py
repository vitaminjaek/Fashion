import tensorflow as tf

## dataset that contains images in the form ((trainX, trainY), (testX, testY))
(trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()

## reshape trainX so we can use Conv2D later on
trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))

## simplifying data so its 0~1 instead of 0~255
trainX = trainX / 255.0
testX = testX / 255.0

## first deep learning model (not very accurate)
"""
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'), ## layer with 128 nodes
    tf.keras.layers.Dense(64, activation='relu'), ## layer with 64 nodes
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax'), ## final layer 0~9
]) ## sigmoid is used for binary predictions
   ## softmax is used for categorical predictions """

## problem with tf.keras.layers.Flatten is that it converts a 2d image into a 1d image, which doesn't work very well
## instead, we can use convolutional layers to extract features from the image
## we can use different kernels to extract specific layers
## on top of that, we can use pooling to simplify the image to further improve predictions
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(28,28,1)), ## convolutional layer
                                                                                                 ## use relu so there are no negative values
    tf.keras.layers.MaxPooling2D((2,2)),                                                                     
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'), ## layer with 64 nodes
    tf.keras.layers.Dense(10, activation='softmax'), ## final layer 0~9
]) ## sigmoid is used for binary predictions
   ## softmax is used for categorical predictions

## instead of using sequential models that can't have complex connected layers, we can also use functional apis
## insert the name of the layer before each layer to design the model
""" input1 = tf.keras.layers.Input(shape=[28, 28])
flatten1 = tf.keras.layers.Flatten()(input1)
dense1 = tf.keras.layers.Dense(28*28, activation='relu')(flatten1)
reshape1 = tf.keras.layers.Reshape((28, 28))(dense1)
concat1 = tf.keras.layers.Concatenate()([input1, reshape1])
flatten2 = tf.keras.layers.Flatten()(concat1)
output = tf.keras.layers.Dense(10, activation='softmax')(flatten2)
model = tf.keras.Model(input1, output) """

## use cross entropy when you are dealing with categorical data
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

## fitting model
model.fit(trainX, trainY, validation_data=(testX, testY), epochs=5)

## evaluating the accuracy of my model
score = model.evaluate(testX, testY)
print(score)

## --------------------------------------------------------------------------------------

## saving/loading entire models
""" model.save('C:\\Users\\dusjc\\Desktop\\CS\\codingapple\\tensorFlow\\fashion\\model')
getModel = tf.keras.models.load_model('C:\\Users\\dusjc\\Desktop\\CS\\codingapple\\tensorFlow\\fashion\\model') """

## saving checkpoints
""" callbackFunction = tf.keras.callbacks.ModelCheckpoint(
    filepath='C:\\Users\\dusjc\\Desktop\\CS\\codingapple\\tensorFlow\\fashion\\checkpoint\\mnist',
    monitor='val_acc', ## only saving the highest validation accuracies
    mode='max',
    save_weights_only=True, ## only saving weights
    save_freq='epoch' ## saving after every epoch
)
model.fit(trainX, trainY, validation_data=(testX, testY), epochs=5, callbacks=[callbackFunction]) """
## use model.load_weights to use the saved weights

## --------------------------------------------------------------------------------------

## using tensorboard to visualize fitting process
""" tensorboard = tf.keras.callbacks.TensorBoard(log_dir='fashion/logs/{}'.format("model" + str(int(time.time()))))
model.fit(trainX, trainY, validation_data=(testX, testY), epochs=5, callbacks=[tensorboard]) """

## early stopping (stop fitting automatically)
""" es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min') ## if val_loss doesn't change after 3 epochs, stop """

## --------------------------------------------------------------------------------------