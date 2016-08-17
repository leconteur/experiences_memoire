from __future__ import print_function, division

import numpy
import sklearn
import tensorflow as tf

import datasets


def build_model(initial_training_rate):
    n_features = X.shape[-1]
    inputtensor = tf.placeholder(tf.float32, shape=[None, numsteps, n_features], name='X')
    dropoutprobtensor = tf.placeholder(tf.float32, name='dropoutProb')
    target = tf.placeholder(tf.int32, shape=[None, numsteps], name='target')
    globalsteptensor = tf.Variable(0, name='global_step', trainable=False)
    learning_rate = tf.train.exponential_decay(initial_training_rate, globalsteptensor,
                                               300, 0.96, staircase=False)
    cells = []
    for i in range(nlayers):
        if i == 0:
            input_size = n_features
        else:
            input_size = NUM_UNITS
        initializer = tf.contrib.layers.xavier_initializer(seed=1000)
        cell = tf.nn.rnn_cell.LSTMCell(NUM_UNITS, input_size=input_size, initializer=initializer,
                                       cell_clip=1.0, forget_bias=0.)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, dropoutprobtensor, dropoutprobtensor)
        cells.append(cell)
    cells = tf.nn.rnn_cell.MultiRNNCell(cells)
    input = tf.nn.dropout(inputtensor, dropoutprobtensor, seed=1000)
    inputtensorlist = [tf.squeeze(i, [1]) for i in tf.split(1, numsteps, input)]
    output, _ = tf.nn.rnn(cells, inputs=inputtensorlist, dtype=tf.float32)
    output = [tf.concat(1, [o, i]) for o, i in zip(inputtensorlist, output)]
    output = tf.reshape(tf.concat(0, output), [-1, NUM_UNITS + n_features])
    bias_init = tf.constant_initializer(0.7)
    l2reg = tf.contrib.layers.l2_regularizer(0.2)
    h = tf.contrib.layers.fully_connected(output, NUM_UNITS, tf.nn.elu)
    logits = tf.contrib.layers.fully_connected(h, 2, bias_init=bias_init, weight_regularizer=l2reg)
    pred = tf.arg_max(logits, 1)
    targettensor = tf.reshape(target, [-1], )

    #mean = tf.reduce_mean(tf.to_float(targettensor))
    #weights_class_1 = tf.ones_like(targettensor, dtype=tf.float32) * (mean)
    #weights_class_2 = tf.ones_like(targettensor, dtype=tf.float32) * (1 - mean)
    #weights = tf.select(tf.equal(targettensor, 0), weights_class_1, weights_class_2)

    weights = tf.ones_like(targettensor, dtype=tf.float32)

    loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [targettensor], [weights], average_across_timesteps=False)
    loss = tf.reduce_sum(loss) * (1 / numsteps)
    sgd = tf.train.GradientDescentOptimizer(learning_rate)
    minimize = sgd.minimize(loss, global_step=globalsteptensor)

    return inputtensor, target, dropoutprobtensor, globalsteptensor, minimize, loss, pred, targettensor


def training_epoch(X, y, batchsize, dropout_prob, inputtensor, targettensor, dropoutprobtensor):
    X, y = sklearn.utils.shuffle(X, y)
    for i, step in enumerate(range(0, X.shape[0], batchsize)):
        xtrain = X[step:step + batchsize]
        targettrain = y[step:step + batchsize]
        fd = {inputtensor: xtrain, targettensor: targettrain, dropoutprobtensor: [dropout_prob]}
        minimize.run(feed_dict=fd)
    fd = {inputtensor: X, targettensor: y, dropoutprobtensor: [1.]}
    l = loss.eval(fd)
    return l / (X.shape[0])


def valid_epoch(X, y, inputtensor, targettensor, dropoutprobtensor):
    fd = {inputtensor: X, targettensor: y, dropoutprobtensor: [1.]}
    l = loss.eval(fd)
    return l / X.shape[0]

def compute_mcc(X, y, inputtensor, targettensor, dropoutprobtensor, pred, truetargets, session):
    fd = {inputtensor: X, targettensor: y, dropoutprobtensor: [1.]}
    p, t = session.run([pred, truetargets], feed_dict=fd)
    return sklearn.metrics.matthews_corrcoef(t, p)


if __name__ == "__main__":
    LR = 0.0025
    NUM_UNITS = 200
    dropout_prob = 0.75
    nlayers = 2
    numsteps = 30
    batchsize = 20

    X, y, labels = [], [], []
    trainingdata, info = datasets.load_training_data('datasets_begin', 'trainset.csv')
    for g, td in trainingdata.groupby(['participant', 'taskname', 'timeofday', 'workload']):
        Xtd, ytd, labelstd = datasets.process_data(td, info, context=False, shuffle=False)
        Xtd, ytd, labelstd = datasets.create_sequence(Xtd, ytd, labelstd, numsteps, 1)
        X.append(Xtd)
        y.append(ytd)
        labels.append(labelstd)
    X = numpy.concatenate(X)
    y = numpy.concatenate(y)
    labels = numpy.concatenate(labels)
    #X, y, labels = sklearn.utils.shuffle(X, y, labels)
    Xtrain, Xtest, ytrain, ytest, labelstrain, labelstest = sklearn.cross_validation.train_test_split(X, y, labels, test_size = 0.1,random_state=1000)
    m = numpy.mean(Xtrain.reshape((-1, 26)), axis=0)
    std = numpy.std(Xtrain.reshape((-1, 26)), axis=0)
    Xtrain = (Xtrain - m) / std
    Xtest = (Xtest - m) / std
    with tf.Graph().as_default(), tf.Session() as session:
        inputtensor, targettensor, dropoutprobtensor, globalsteptensor, minimize, loss, pred, truetargets = build_model(LR)
        tf.initialize_all_variables().run()
        print("{:5}    {:10}    {:10}    {:10}".format("Epoch", "Training loss", "Valid loss", "MCC"))
        for epoch in range(1, 101):
            trainloss = training_epoch(Xtrain, ytrain, batchsize, dropout_prob, inputtensor, targettensor, dropoutprobtensor)
            validloss = valid_epoch(Xtest, ytest, inputtensor, targettensor, dropoutprobtensor)
            mcc = compute_mcc(Xtest, ytest, inputtensor, targettensor, dropoutprobtensor, pred, truetargets, session)
            epoch = globalsteptensor.eval(session)
            print("{:5}    {:<.4}           {:<.4}        {:<.4}".format(epoch, trainloss, validloss, mcc))
