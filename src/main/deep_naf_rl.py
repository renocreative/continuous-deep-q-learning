'''
    Deep Reinforcement Learning using Normalized Advantage Functions for Continuous Action Spaces
'''

import tensorflow as tf

n_epochs = 40
e = .2
reward_discount =.9
actions = []
R = [] # Initialize replay buffer R <- empty.
m = 10 # minibatch size of transitions
bias_size = 64

# Randomly initialize normalized Q network Q(x, u|W_Q).
Qu = myConv2D()
# Qu = Au + V


# Initialize target network Q' with weight W_Q' <- W_Q.
# Empty!
target_Qu = myConv2D()
# target_Qu = target_Au + target_V


training_batch = tf.placeholder(none)

# Set y_i = r_i + discount * V W(x_i+1|W_Q')
# y = r + reward_discount * target_Qu(next_x)
y = training_batch.r + reward_discount * target_Qu(training_batch.next_x) # to be modified to use target network
    
# The loss function: L = 1/N Sum_i 
#(y_i - Q(x_i, u_i | W_Q))^2
loss = 1/n * tf.squared_difference(y, Qu)



def main():
    print "Running %s" % __file__
    for epoch in range(n_epochs):
        # Initialize a random process N for action exploration
        # Nothing to do
        def getAction(x):
            e_val = tf.random()
            if e_val <= e:
                #random selection
                u = tf.random(actions)
            else:
                u = argmax Qu(x)
            return u

        # Receive initial observation state x_1 ∼ p(x_1)
        x = environment.get()


        for t in range(T)
            # Select action ut = u*(x_t|W_u*) + N_t
            u = getAction(x)

            # Execute ut and observe r_t and x_t+1
            next_x, r = environment.put(x, u)
            x = next_x

            # Store transition (x_t, u_t, r_t, x_t+1) in R
            transition.x = x
            transition.u = u
            transition.r = r
            transition.next_x = next_x
            R.push(transition)

            for iteration in range(I)
                # Initialize training batch
                training_batch = []

                # Sample a random minibatch of m transitions from R
                # Sequentially sample each transition
                for count in range (m):
                    i = tf.randomInt(R.size)
                    training_batch.add(R[i])

                # Update W_Q by minimizing the loss
                loss_val = loss.eval(batch)
                
                print 'loss value is {:d}'.format(loss_val)
                
                # Update the target network: W_Q' ← kW_Q + (1 − k)W_Q'
                k = 0.5
                target_w1 = k * w1 + (1 - k) * target_w1
                target_w2 = k * w2 + (1 - k) * target_w2
                target_b1 = k * b1 + (1 - k) * target_b1
                target_b2 = k * b2 + (1 - k) * target_b2


    
if __name__ == '__main__': main()
    