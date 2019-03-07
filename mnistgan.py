import keras
from keras.datasets import mnist
from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

class mnist():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.latent_dim = 100
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.input_shape = (self.img_rows, self.img_cols, 1)
        (self.xtrain, _), _ = keras.datasets.mnist.load_data()
        self.xtrain = self.xtrain / 127.5 - 1
        self.xtrain = np.expand_dims(self.xtrain, axis=3) 

        self.D = self.build_D()
        self.D.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.G = self.build_G()
        z = Input(shape=(self.latent_dim,))
        img = self.G(z)

        self.D.trainable = False
        valid = self.D(img)

        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer='adam')


    def build_G(self):
        noise_shape = (100,)
        model = Sequential()

        model.add(Dense(256, input_shape=noise_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        #BatchNormalization(momentum=0.8)
        model.add(LeakyReLU(alpha=0.2))
        model.add(UpSampling1D())
        model.add(Dense(512))
        #BatchNormalization(momentum=0.8)
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1024))
        #BatchNormalization(momentum=0.8)
        model.add(LeakyReLU(alpha=0.2))
        model.add(UpSampling1D())
        model.add(Dense(np.prod(self.img_shape)))
        model.add(Activation("tanh"))
        model.add(Reshape(self.img_shape))
        
        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_D(self):
        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same", input_shape=self.img_shape))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        #model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Dropout(0.25))
        model.add(ZeroPadding2D((3,3)))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        #model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        #model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation="sigmoid"))
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)
        
        

    def train(self, epochs=25, batch_size=100, use_cb=False):
        model = keras.models.Sequential()
        cb = [keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.003)] if use_cb else []
        
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1)) 

        for epoch in range(epochs):
            idx = np.random.randint(0, self.xtrain.shape[0], batch_size)
            imgs = self.xtrain[idx]
            
            noise = np.random.normal(0,1,(batch_size, self.latent_dim))
            gen_imgs = self.G.predict(noise)

            d_loss_real = self.D.train_on_batch(imgs, valid)
            d_loss_fake = self.D.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_real)

            g_loss = self.combined.train_on_batch(noise, valid)
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]"
                     % (epoch, d_loss[0], 100*d_loss[1], g_loss))
        
        self.save_imgs(epochs)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.G.predict(noise)

        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/mnist_%d.png" % epoch)
        plt.close()


if __name__ == "__main__":
    mnist = mnist()
    mnist.train()

        

