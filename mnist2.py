import keras
from keras import backend as K


class mnist():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.input_shape = (self.img_rows, self.img_cols, 1)
        (xtr, ytr), (xte, yte) = keras.datasets.mnist.load_data()
        xtr, xte = xtr / 255.0, xte / 255.0
        self.xtrain = xtr.reshape(xtr.shape[0], self.img_rows, self.img_cols, 1)
        self.xtest = xte.reshape(xte.shape[0], self.img_rows, self.img_cols, 1)
#        one_hot = lambda x: [int(i == x) for i in range(1,10)]
#        self.ytrain = list(map(one_hot, ytr))
#        self.ytest = list(map(one_hot, yte))
        self.ytrain = keras.utils.to_categorical(ytr, 10)
        self.ytest = keras.utils.to_categorical(yte, 10)
        print(self.xtrain.shape)
        print(self.ytrain.shape)
        print(self.xtest.shape)
        print(self.ytest.shape)

    def trainTutorial(self, epochs=25, batch_size=100, use_cb=False):
        model = keras.models.Sequential()
        cb = [keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.003)] if use_cb else []
        model.add(keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=self.input_shape))
        model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(10, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy',
                metrics=['accuracy'])
        model.summary()
        model.fit(self.xtrain, self.ytrain, epochs=epochs, callbacks=cb, validation_data=(self.xtest, self.ytest))
        model.evaluate(self.xtest, self.ytest)




if __name__ == "__main__":
    mnist = mnist()
    mnist.trainTutorial()

        

