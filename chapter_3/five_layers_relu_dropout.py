from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import math

logs_path = 'log_simple_stats_5_layers_relu_dropout_softmax'
batch_size = 100
learning_rate = 0.5
training_epochs = 10

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784])
Y_ = tf.placeholder(tf.float32, [None, 10])
lr = tf.placeholder(tf.float32)
dropout_ratio = tf.placeholder(tf.float32)

L = 200
M = 100
N = 60
O = 30
with tf.name_scope("inputlayer") as scope:
    XX = tf.reshape(X, [-1, 784])


#  for layer in [layer1,layer4 ], use the Relu function

with tf.name_scope("layer1") as scope:
    W1 = tf.Variable(tf.truncated_normal([784, L], stddev=0.1),name="W1")
    B1 = tf.Variable(tf.ones([L]) / 10,name="B1")
    Y1 = tf.nn.relu(tf.matmul(XX, W1) + B1)
    Y1d = tf.nn.dropout(Y1, dropout_ratio,name="OUTPUT1")

with tf.name_scope("layer2") as scope:
    W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1),name="W2")
    B2 = tf.Variable(tf.ones([M]) / 10,name="B2")
    Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
    Y2d = tf.nn.dropout(Y2, dropout_ratio, name="OUTPUT2")

with tf.name_scope("layer3") as scope:
    W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1),name="W3")
    B3 = tf.Variable(tf.ones([N]) / 10,name="B3")
    Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
    Y3d = tf.nn.dropout(Y3, dropout_ratio, name="OUTPUT3")

with tf.name_scope("layer4") as scope:
    W4 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1),name="W4")
    B4 = tf.Variable(tf.ones([O]) / 10,name="B4")
    Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)
    Y4d = tf.nn.dropout(Y4, dropout_ratio, name="OUTPUT4")


with tf.name_scope("outputlayer") as scope:
    W5 = tf.Variable(tf.truncated_normal([O, 10], stddev=0.1),name="W5")
    B5 = tf.Variable(tf.zeros([10]),name="B5")
    Ylogits = tf.matmul(Y4, W5) + B5
    Y = tf.nn.softmax(Ylogits,name="FINAL_OUTPUT")

with tf.name_scope("cross_entropy") as scope:
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
    cross_entropy = tf.multiply(tf.reduce_mean(cross_entropy),100,name="cross_entropy")

with tf.name_scope("correct_predition") as scope:
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1),name="correct_predition")

with tf.name_scope("accuracy") as scope:
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name="accuracy")

with tf.name_scope("min_errors") as scope:
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

tf.summary.scalar("cost", cross_entropy)
tf.summary.scalar("accuracy", accuracy)
summary_op = tf.summary.merge_all()

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    for epoch in range(training_epochs):
        batch_count = int(mnist.train.num_examples / batch_size)
        for i in range(batch_count):
            batch_x, batch_y = mnist.train.next_batch(batch_size)

            #define a learning_rate
            max_learning_rate = 0.003
            min_learning_rate = 0.0001
            decay_speed = 2000

            learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i / decay_speed)
            _, summary = sess.run([train_step, summary_op],  {X: batch_x, Y_: batch_y,
                                                              lr: learning_rate,dropout_ratio:0.75})
            writer.add_summary(summary,epoch * batch_count + i)
        # if epoch % 2 == 0:
        print("Epoch: ", epoch)

    print("Accuracy: ", accuracy.eval(feed_dict={X: mnist.test.images, Y_: mnist.test.labels,dropout_ratio:1}))
    print("done")