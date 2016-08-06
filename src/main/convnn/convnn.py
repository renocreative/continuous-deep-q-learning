from __future__ import absolute_import 
from __future__ import division 
#from __future__ import print_function 

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''

    Classic ConvNets Architecture: Conv -> Pooling -> Conv -> Pooling -> Fully Connected -> Classifier

'''


# Parameters
n_epoch = 100
#n_features = 784
n_examples = None
#n_hidden_units_1 = 10
#n_hidden_units_2 = 5
n_outputs = 10
learning_rate = 1e-4
mini_batch_size = 100

class MyConv2D:
    
    #x, xx, yy, predicted_y
    #w1, w2, w3, w4, b1, b2, b3, b4
    
    x = tf.placeholder(tf.float32)
    yy = tf.placeholder(tf.float32)
    
    # Fetch the mnist data 
    def fetch(self): 
        return input_data.read_data_sets('MNIST_data', one_hot = True) 
    
    def __init__(self, n_outputs):
        self.setup()
        self.loss()
        self.minimizer()
        self.n_outputs = n_outputs
    
    def createWeight(self, shape):
        return tf.Variable (tf.truncated_normal(shape=shape, stddev=.1))
    
    def createBias(self, shape):
        return tf.Variable (tf.constant(.1, shape=shape))
    
    def createConv2D(self, input, filter):
        return tf.nn.conv2d(input, filter, strides=[1,1,1,1], padding='SAME')
        
    def applyMaxPooling2x2(self, convLayer):
        return tf.nn.max_pool(convLayer, ksize = [1,2,2,1], strides = [1,2,2,1], padding='SAME') 
    
    def setup(self):
        # Randomly initialize      
        self.xx = tf.reshape(self.x, [-1,28,28,1])

        # Conv layer 1
        self.w1 = self.createWeight([5,5,1,32])
        self.b1 = self.createBias([32])
        conv1 = self.createConv2D(self.xx, self.w1)
        activated_conv1 = tf.nn.relu(conv1 + self.b1)
        pooled_conv1 = self.applyMaxPooling2x2(activated_conv1)

        # Conv layer 2
        self.w2 = self.createWeight([5,5,32,64])
        self.b2 = self.createBias([64])
        conv2 = self.createConv2D(pooled_conv1, self.w2)
        activated_conv2 = tf.nn.relu(conv2 + self.b2)
        pooled_conv2 = self.applyMaxPooling2x2(activated_conv2) 

        # NN Layer
        self.w3 = self.createWeight([7*7*64,1024])
        self.b3 = self.createBias([1024])
        pooled_conv2_flat = tf.reshape(pooled_conv2, [-1, 7*7*64])
        nnlayer3 = tf.nn.relu (tf.matmul(pooled_conv2_flat, self.w3) + self.b3)
        
        # Readout Layer
        self.w4 = self.createWeight([1024, self.n_outputs])
        self.b4 = self.createBias([self.n_outputs])
        
        self.predicted_y =tf.nn.softmax(tf.matmul(nnlayer3, self.w4) + self.b4)

        return self.predicted_y
    
    def loss(self): #cross-entropy
        return tf.reduce_mean(-tf.reduce_sum(self.yy * tf.log(self.predicted_y), reduction_indices=[1]))
    
    def minimizer(self):
        return tf.train.AdamOptimizer(learning_rate).minimize(self.loss())
 
'''
tf.scalar_summary("Loss", tf.reduce_mean(loss))
tf.scalar_summary("Weight1", tf.reduce_mean(ww_1))
tf.scalar_summary("Weight2", tf.reduce_mean(ww_2))
tf.scalar_summary("Weight3", tf.reduce_mean(ww_3))'''
#merged = tf.merge_all_summaries()
    
def main():
    print "Running {:s}".format(__file__)
    
    #tf.is_variable_initialized(ww)
    with tf.Session() as sess:
        convNN = MyConv2D()
        minimizer = convNN.minimizer()
        loss = convNN.loss()
        mnist = convNN.fetch()
        
        # Create a summary writer, add the 'graph' to the event file.
        writer = tf.train.SummaryWriter(".", sess.graph)
        
        # Init variables
        sess.run(tf.initialize_all_variables())
        
        for epoch in range(n_epoch):
            batch = mnist.train.next_batch(mini_batch_size)
            _, loss_val =sess.run([minimizer, loss], feed_dict={convNN.x: batch[0], convNN.yy: batch[1]})
            
            print "run epoch {:d}: loss value is {:f}".format(epoch, loss_val) 
            #print summaries
            #writer.add_summary(summaries,epoch)
        
        correct_prediction = tf.equal(tf.argmax(convNN.yy,1), tf.argmax(convNN.predicted_y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy_val = accuracy.eval(feed_dict={convNN.x: mnist.test.images, convNN.yy: mnist.test.labels})
        print "\naccuracy is {:f}".format(accuracy_val*100)
    # print eval(test_data)
    
if __name__ == '__main__': main()
    
        
