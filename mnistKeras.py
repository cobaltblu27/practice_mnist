import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from keras import backend as k
from keras.layers import Dense
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy as accuracy

sess = tf.Session()
k.set_session(sess)

img = tf.placeholder(tf.float32, shape = [None, 784]) # "shape =" 생략해도 됨.

# fully-connected layer with 128 units and ReLU activation
x = Dense(128, activation = 'relu')(img)
x = Dense(128, activation = 'relu')(x)

preds = Dense(10, activation = 'softmax')(x)
labels = tf.placeholder(tf.float32, [None, 10])

loss = tf.reduce_mean(categorical_crossentropy(labels, preds))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

acc = accuracy(labels, preds)

sess.run(tf.global_variables_initializer())

mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)
with sess.as_default():
    for i in range(1000):
        batch = mnist_data.train.next_batch(50)
        train_step.run(feed_dict={img:batch[0], labels:batch[1]})
    
    correct_prediction = acc.eval(
            feed_dict={img:mnist_data.test.images, labels:mnist_data.test.labels})    

print(sess.run(tf.reduce_mean(correct_prediction)))
