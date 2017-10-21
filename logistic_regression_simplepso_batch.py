import tensorflow as tf
import tools 
import random

import SimplePSO
import numpy as np
from functools import partial


# Parameters
learning_rate = 0.01
training_epochs = 5
batch_size = 30
display_step = 1
FeatureNo = 4
LabelNo = 3
SampleNo = 120

# tf Graph Input
x = tf.placeholder(tf.float32, [None, FeatureNo]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, LabelNo]) # 0-9 digits recognition => 10 classes

# Set model weights
# W = tf.Variable(tf.zeros([FeatureNo, LabelNo]))
# b = tf.Variable(tf.zeros([LabelNo]))
# we need to feed the paramters tuned by PSO in to the model. So the weights and bias use the placeholder insteaded
W = tf.placeholder(tf.float32, [FeatureNo, LabelNo])
b = tf.placeholder(tf.float32, [LabelNo])
# The pyswarm need to define the upper bounds and lower bounds
bounds=[(-1,1) for i in range(FeatureNo * LabelNo + LabelNo)]

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# # Gradient Descent
# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# pyswarm need to give a subroutine, that's why this capsule is made
def fitness(dummy_x, sess, bbatch_xs, bbatch_ys, fFeatureNo, lLabelNo):
    WW = np.array(dummy_x[0:fFeatureNo*lLabelNo])
    WW = np.reshape(WW, [fFeatureNo, lLabelNo])
    bb = np.array(dummy_x[fFeatureNo*lLabelNo:])
    return(
    sess.run([cost], feed_dict={x: bbatch_xs,
                                y: bbatch_ys,
                                W: WW,
                                b: bb
                                })
    )
pass

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

Tr_features, Tr_labels, Ts_features, Ts_labels, label_set = tools.csv_train_test_shuffer_split('iris.csv', 0.8)
gen = tools.training_data_generator(Tr_features, Tr_labels)

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)


    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(SampleNo/batch_size)
        PSO_init = [random.random() for i in range(FeatureNo * LabelNo + LabelNo)]
        # Loop over all batches
        for i in range(total_batch):
            
            #batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs , batch_ys = [], []
            for i in range(batch_size):
                j,k = gen.__next__()
                onehotk, label_set = tools.trans2onehot(k, label_set)
                batch_xs.append(j)
                batch_ys.append(onehotk)
            pass
            
            
            
            # # Run optimization op (backprop) and cost op (to get loss value)
            # _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          # y: batch_ys})
            
            # use the PSO as optimizer
            # opt_weights = SimplePSO.PSO(fitness, PSO_init, bounds, 50, 100,
                                        # sess, batch_xs, batch_ys, FeatureNo, LabelNo
                                        # ).pos_best_g
            [c, opt_weights] = SimplePSO.PSO(fitness, PSO_init, bounds, 50, 100,
                                                sess, batch_xs, batch_ys, FeatureNo, LabelNo
                                                ).results()
            PSO_init = opt_weights[:]
            # Compute average loss
            avg_cost += c / total_batch
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
            #print("Epoch:", '%04d' % (epoch+1))
            #print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    onehot_ts = []
    for i in Ts_labels:
        onehotk, label_set = tools.trans2onehot(i, label_set)
        onehot_ts.append(onehotk)
    pass
    
    WW = np.array(opt_weights[0:FeatureNo*LabelNo])
    WW = np.reshape(WW, [FeatureNo, LabelNo])
    bb = np.array(opt_weights[FeatureNo*LabelNo:])
    print("Accuracy:", accuracy.eval({x: Ts_features, 
                                      y: onehot_ts,
                                      W: WW,
                                      b: bb
                                      }))
