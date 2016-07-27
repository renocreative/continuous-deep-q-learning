import tensorflow as tf

# Parameters
n_epoch = 50
n_features = 1
n_examples = 3
n_hidden_1 = 30
n_hidden_2 = 10
n_outputs = 1
learning_rate = .01


# Fetch the data
def fetch():
    return 0


# Define the model
# Model inputs & outputs definitions
xx = tf.placeholder(tf.float32, shape=(n_examples, n_features), name = "MyInputs")
yy = tf.placeholder(tf.float32, shape=(n_examples, n_outputs), name = "MyLabels")

# Model hypothesis
ww = tf.Variable(tf.zeros(dtype=tf.float32, shape=(n_features, 1)), name = "MyWeights", trainable=True) 
predict_yy = tf.matmul(xx, ww)


# Evaluate the loss
loss = tf.squared_difference(predict_yy, yy, "MyLoss")
    

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
    fetch()
    #tf.is_variable_initialized(ww)
    with tf.Session() as sess:
        # Create a summary writer, add the 'graph' to the event file.
        writer = tf.train.SummaryWriter(".", sess.graph)
        init.run()
        for epoch in range(n_epoch):
            summaries, _, _,_ =sess.run([merged, minimizer, loss, ww], feed_dict={xx:[[1],[2],[3]], yy:[[2],[4],[6]]})
            
            
            print summaries
            writer.add_summary(summaries,epoch)
        
    # print eval(test_data)
        
    
if __name__ == '__main__': main()
    