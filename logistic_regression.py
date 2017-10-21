''' Ogirginal license
The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

All contributions by Aymeric Damien:
Copyright (c) 2015, Aymeric Damien.
All rights reserved.

'''

import tensorflow as tf
import tools 

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
W = tf.Variable(tf.zeros([FeatureNo, LabelNo]))
b = tf.Variable(tf.zeros([LabelNo]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

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
            
            # Run optimization op (backprop) and cost op (to get loss value)
            for i in range(5000):
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                              y: batch_ys})
            
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
            # print(sess.run(W))
            # print(sess.run(b))

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
    print("Accuracy:", accuracy.eval({x: Ts_features, 
                                      y: onehot_ts}))
