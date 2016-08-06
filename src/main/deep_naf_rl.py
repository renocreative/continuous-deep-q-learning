'''
    Deep Reinforcement Learning using Normalized Advantage Functions for Continuous Action Spaces
    Adapted for OpenAi Gym's Atari Environments
'''

import tensorflow as tf
from convnn.convnn import conv2D

n_epochs = 40
e = .2
reward_discount =.9
k = 0.5 # Parameter for updating the target Q network
R = [] # Initialize replay buffer R <- empty.
m = 10 # minibatch size of transitions
bias_size = 64
total_reward = 0
best_score = -999999999
T = 1000
I = 16
learningrate = .5

#Create environment

# Randomly initialize normalized Q network Q(x, u|W_Q).
Qu = myConv2D(env.action_size)
# Qu = Au + V
# the advantage for a given state-action pair under a policy pi is 
# its q-value under that policy minus the best q-value that can be outputed under that policy
# Au = Qu - V => Au = Qu(s)[u] - tf.reduce_max( Qu(s) )


# Empty!
target_Qu = myConv2D(env.action_size)
# target_Qu = target_Au + target_V

# Initialize target network Q' with weight W_Q' <- W_Q.
target_Qu.w1.assign(Qu.w1)
target_Qu.w2.assign(Qu.w2)
target_Qu.w3.assign(Qu.w3)
target_Qu.w4.assign(Qu.w4)
target_Qu.b1.assign(Qu.b1)
target_Qu.b2.assign(Qu.b2)
target_Qu.b3.assign(Qu.b3)
target_Qu.b4.assign(Qu.b4)

training_batch = tf.placeholder(none)

# Set y_i = r_i + discount * V W(x_i+1|W_Q')
# y = r + reward_discount * target_Qu(next_x)
y = training_batch.r + reward_discount * target_Qu(training_batch.next_x) # to be modified to use state value instead of q-value
    
# The loss function: L = 1/N Sum_i 
#(y_i - Q(x_i, u_i | W_Q))^2
loss = 1/m * tf.squared_difference(y, Qu) # reshape into one-hot vectors

minimizer = tf.train.GradientDescentOptimizer(learningrate).minimize(loss)



def main():
    print "Running %s" % __file__
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        for epoch in range(n_epochs):
            # Initialize a random process N for action exploration
            # Nothing to do
            def getAction(x):
                e_val = tf.random_uniform(shape=[], maxval=1)
                if e_val <= e:
                    #random selection
                    u = tf.random_uniform(shape=[], maxval=env.action_size, dtype=tf.int32)
                else:
                    u = tf.argmax( Qu(x) )
                return u


            # Receive initial observation state x_1 ∼ p(x_1)
            x = env.reset()

            for t in range(T)
                # Select action ut = u*(x_t|W_u*) + N_t
                u = getAction(x)

                # Execute ut and observe r_t and x_t+1
                next_x, r, done = env.act(u)
                x = next_x

                # Store transition (x_t, u_t, r_t, x_t+1) in R
                transition = {}
                transition.x = x
                transition.u = u
                transition.r = r
                transition.next_x = next_x
                R.append(transition)

                # Track statistics
                total_reward += r

                for iteration in range(I)
                    # Initialize training batch
                    training_batch = []

                    # Sample a random minibatch of m transitions from R
                    # Sequentially sample each transition
                    for count in range (m):
                        i = tf.random_uniform(maxval=len(R), dtype=tf.int32)
                        training_batch.add(R[i])

                    # Update W_Q by minimizing the loss
                    # reshape batch.r into a one-hot vector of size = env.action_size
                    _, loss_val = sess.run([minimizer, loss], feed_dict={training_batch: training_batch, Qu.xx: batch.x, Qu.y: batch.r})

                    print 'loss value is {:d}'.format(loss_val)

                    # Update the target network: W_Q' ← kW_Q + (1 − k)W_Q'
                    target_Qu.w1.assign(k * Qu.w1 + (1 - k) * target_Qu.w1)
                    target_Qu.w2.assign(k * Qu.w2 + (1 - k) * target_Qu.w2)
                    target_Qu.w1.assign(k * Qu.w3 + (1 - k) * target_Qu.w3)
                    target_Qu.w2.assign(k * Qu.w4 + (1 - k) * target_Qu.w4)
                    target_Qu.b1.assign(k * Qu.b1 + (1 - k) * target_Qu.b1)
                    target_Qu.b2.assign(k * Qu.b2 + (1 - k) * target_Qu.b2)
                    target_Qu.b1.assign(k * Qu.b3 + (1 - k) * target_Qu.b3)
                    target_Qu.b2.assign(k * Qu.b4 + (1 - k) * target_Qu.b4)
                    
            if done == true:
                if best_score < total_reward
                    best_score = total_reward
                break


    
if __name__ == '__main__': main()
    
