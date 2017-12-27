import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from random import randint
import numpy as np
from sklearn.metrics import f1_score,recall_score,precision_score

logs_path = 'log_mnist_softmax'
batch_size = 128
learning_rate = 0.2
training_epochs = 100
mnist = input_data.read_data_sets("data", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784], name="input")
Y_ = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
XX = tf.reshape(X, [-1, 784])

Y = tf.nn.softmax(tf.matmul(XX, W) + b, name="output")
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=Y))
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

loss = -tf.reduce_sum(Y_*tf.log(Y))


train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)

tf.summary.scalar("cost", cross_entropy)
tf.summary.scalar("accuracy", accuracy)
summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(logs_path,  graph=tf.get_default_graph())
    for epoch in range(training_epochs):
        batch_count = int(mnist.train.num_examples / batch_size)
        for i in range(batch_count):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, summary = sess.run([train_step, summary_op],feed_dict={X: batch_x,Y_: batch_y})
            writer.add_summary(summary, epoch * batch_count + i)

        # show the loss of every epoch
        _, c = sess.run([train_step, loss], feed_dict={X: mnist.train.images, Y_: mnist.train.labels})
        print("Epoch {}: Training Loss = {}, Training Accuracy = {}".format(
                epoch, c, sess.run(accuracy, feed_dict={X: mnist.train.images, Y_: mnist.train.labels})))

    # show the performance of the model on test data
    print("the accuracy of test data is following:")
    print("Accuracy: ", accuracy.eval(feed_dict={X: mnist.test.images, Y_: mnist.test.labels}))
    y_p = tf.argmax(Y, 1)
    y_true = np.argmax(mnist.test.labels, 1)
    final_acc, y_pred = sess.run([accuracy, y_p], feed_dict={X: mnist.test.images, Y_: mnist.test.labels})
    print("Testing Accuracy: {}".format(final_acc))

    temp_y_true = np.unique(y_true)
    temp_y_pred = np.unique(y_pred)
    np.save("y_true", y_true)
    np.save("y_pred", y_pred)
    print("temp_y_true", temp_y_true)
    print("temp_y_pred", temp_y_pred)
    # calculate metrics for the model
    print("Precision", precision_score(y_true.tolist(), y_pred.tolist(), average='weighted'))
    print("Recall", recall_score(y_true, y_pred, average='weighted'))
    print("f1_score", f1_score(y_true, y_pred, average='weighted'))

    print("test done")

    num = randint(0, mnist.test.images.shape[0])

    img = mnist.test.images[num]

    classification = sess.run(tf.argmax(Y, 1), feed_dict={X: [img]})
    print('Neural Network predicted', classification[0])
    print('Real label is:', np.argmax(mnist.test.labels[num]))

    saver = tf.train.Saver()
    save_path = saver.save(sess, "data/saved_mnist_cnn.ckpt")
    print("Model saved to %s" % save_path)