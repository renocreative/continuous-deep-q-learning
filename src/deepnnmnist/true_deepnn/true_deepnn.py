'''

Training a deep ANN on MNIST with Tensorflow

Loss is the cross-entropy function

'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# Parameters
n_epoch = 1200
n_features = 784
n_examples = None
n_hidden_units_1 = 10
n_hidden_units_2 = 5
n_outputs = 10
learning_rate = .5
mini_batch_size = 100

# Fetch the mnist data
def fetch():
    return input_data.read_data_sets('MNIST_data', one_hot = True)


# Define the model
# Model inputs & outputs definition
xx = tf.placeholder(tf.float32, shape=(n_examples, n_features), name = "MyInputs")
yy = tf.placeholder(tf.float32, shape=(n_examples, n_outputs), name = "MyLabels")


# Model hypothesis
ww_1 = tf.Variable(tf.truncated_normal(shape=(n_features, n_hidden_units_1), stddev = .1, dtype=tf.float32), name = "MyWeights_1")
bb_1 = tf.Variable(tf.zeros([n_hidden_units_1], dtype=tf.float32), name = "MyBias_1")
ww_2 = tf.Variable(tf.truncated_normal(shape=(n_hidden_units_1, n_hidden_units_2), stddev = .1, dtype=tf.float32), name = "MyWeights_2")
bb_2 = tf.Variable(tf.zeros([n_hidden_units_2], dtype=tf.float32), name = "MyBias_2")
ww_3 = tf.Variable(tf.truncated_normal(shape=(n_hidden_units_2, n_outputs), stddev = .1, dtype=tf.float32), name = "MyWeights_3")
bb_3 = tf.Variable(tf.zeros([n_outputs], dtype=tf.float32), name = "MyBias_3")


aa_1 = tf.nn.sigmoid(tf.matmul(xx, ww_1) + bb_1)
aa_2 = tf.nn.sigmoid(tf.matmul(aa_1, ww_2) + bb_2)
predict_yy = tf.nn.softmax(tf.matmul(aa_2, ww_3) + bb_3)


# Evaluate the loss
loss = tf.reduce_mean(-tf.reduce_sum(yy * tf.log(predict_yy), reduction_indices=[1]))
    

# Train the model / Apply gradient updates (One Step)
# Calculate gradient of the loss for each weight
# + Update each weight
opt = tf.train.GradientDescentOptimizer(learning_rate)
minimizer = opt.minimize(loss)

    
# Evaluate the model against the test data. Test the model
def eval(inputs):
    return tf.matmul(inputs, ww)

# Init variables
init = tf.initialize_all_variables()

tf.scalar_summary("Loss", tf.reduce_mean(loss))
tf.scalar_summary("Weight1", tf.reduce_mean(ww_1))
tf.scalar_summary("Weight2", tf.reduce_mean(ww_2))
tf.scalar_summary("Weight3", tf.reduce_mean(ww_3))
merged = tf.merge_all_summaries()

def main():
    print "Running %s" % __file__
    mnist = fetch()
    #tf.is_variable_initialized(ww)
    with tf.Session() as sess:
        # Create a summary writer, add the 'graph' to the event file.
        writer = tf.train.SummaryWriter(".", sess.graph)
        init.run()
        for epoch in range(n_epoch):
            batch = mnist.train.next_batch(mini_batch_size)
            summaries, _, loss_val =sess.run([merged, minimizer, loss], feed_dict={xx: batch[0], yy: batch[1]})
            
            
            print "run epoch {:d}: loss value is {:f}".format(epoch, loss_val) 
            #print summaries
            writer.add_summary(summaries,epoch)
        
        correct_prediction = tf.equal(tf.argmax(yy,1), tf.argmax(predict_yy,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy_val = accuracy.eval(feed_dict={xx: mnist.test.images, yy: mnist.test.labels})
        print "\naccuracy is {:f}".format(accuracy_val*100)
    # print eval(test_data)
        
    
if __name__ == '__main__': main()
    