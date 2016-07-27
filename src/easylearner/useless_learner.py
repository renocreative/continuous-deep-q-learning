'''

    Training a trivial parametric monomial function "wx" (with no bias parameter) 
    to approximate the true hypothesis f(x)= 2x given 3 datapoints of observation (1, 2),(2, 4),(3, 6)
    This learner has no practical usage (hence, its name).
    
    We are using non-stochastic gradient descent and running weight updates for 30 epochs.
    
    A rudimentary squared difference is used for the loss function:
        
        From data, we get:
        L(w) = (2-w)(2-w) + (4-2w)(4-2w) + (6-3w)(6-3w)
             = (4 - 4w + w^2) + (16 - 16w + 4w^2) + (36 - 36w + 9w^2)
        L(w) = 56 - 56w + 14w^2
        
        L'(w) = -56 + 28w
             
        Solving this analytically gives us w = 2
        
        But for the sake of this exercise, we apply gradient descent with w starting at 0 <=> w_t = 0 for t=0 where t is epoch 
        
        w_t+1 = w_t - learning_rate * L'(w_t)
    
    
    Training effectively overfits the data as the setup is completely hypothetical (e.g. there is no test data)
    The point of the exercise is solely to get familiar with operating Google TensorFlow framework

'''

import tensorflow as tf

# Parameters
n_epoch = 30
n_features = 1
n_examples = 3
n_outputs = 1
learning_rate = .01


# Fetch the data
def fetch():
    return {xx:[[1],[2],[3]], yy:[[2],[4],[6]]}


# Define the model
# Model inputs & outputs definitions
xx = tf.placeholder(tf.float32, shape=(n_examples, n_features), name = "MyInputs")
yy = tf.placeholder(tf.float32, shape=(n_examples, n_outputs), name = "MyLabels")

# Model hypothesis
ww = tf.Variable(tf.zeros(dtype=tf.float32, shape=(n_features, 1)), name = "MyWeights", trainable=True) 
predict_yy = tf.matmul(xx, ww)


# Evaluate the loss
loss = tf.reduce_sum(tf.squared_difference(predict_yy, yy), name = "MyLoss")
    

# Train the model / Apply gradient updates (One Step)
# Calculate gradient of the loss for each weight
# + Update each weight
opt = tf.train.GradientDescentOptimizer(learning_rate= learning_rate)
minimizer = opt.minimize(loss, var_list=[ww])

    
# Evaluate the model against the test data. Test the model
def eval(inputs):
    return tf.matmul(inputs, ww)

# Init variables
init = tf.initialize_all_variables()

tf.scalar_summary("Loss", tf.reduce_mean(loss))
tf.scalar_summary("Weight", tf.reduce_mean(ww))
merged = tf.merge_all_summaries()

def main():
    print "Running %s" % __file__
    #tf.is_variable_initialized(ww)
    with tf.Session() as sess:
        # Create a summary writer, add the 'graph' to the event file.
        writer = tf.train.SummaryWriter(".", sess.graph)
        init.run()
        for epoch in range(n_epoch):
            summaries, _, loss_value,_ =sess.run([merged, minimizer, loss, ww], feed_dict = fetch())
            
            
            print 'epoch {:d}: loss is {:f}'.format(epoch, loss_value)
            writer.add_summary(summaries, epoch)
        
    # eval(test_data)
        
    
if __name__ == '__main__': main()
    