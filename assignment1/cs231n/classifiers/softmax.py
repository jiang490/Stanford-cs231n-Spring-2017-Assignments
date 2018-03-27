import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    num_class = W.shape[1]
    num_train = X.shape[0]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    for i in range(num_train):
        # loss
        base_loss = np.dot(X[i, :], W)
        base_loss -= np.max(base_loss)
        sum_exp_loss = 0

        for j in range(num_class):
            # loss
            sum_exp_loss += np.exp(base_loss[j])
        # logistic calculate the log MLE estimator, different from the distance
        # measure minimization of the SVM. That's why we need to take the log
        # for adding the terms, same as multiplying the probabilities w/o log.
        loss += -base_loss[y[i]] + np.log(sum_exp_loss)

        # graient
        for j in range(num_class):
            dW[:, j] += X[i, :] * np.exp(base_loss[j]) / sum_exp_loss
        dW[:, y[i]] -= X[i, :]

    # for average loss
    loss /= num_train
    dW /= num_train
    # regularization
    loss += 0.5 * reg * np.sum(W**2)
    dW += reg * W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_class = W.shape[1]
    num_train = X.shape[0]

    # loss
    base = np.dot(X, W)
    base -= np.max(base, axis=1)[:, np.newaxis]
    exp_base = np.exp(base)
    loss = np.sum(-base[np.arange(num_train), y]) + \
        np.sum(np.log(np.sum(exp_base, axis=1)))
    loss /= num_train
    loss += 0.5 * reg * np.sum(W**2)

    # gradient
    mask = np.zeros_like(base)
    mask[np.arange(num_train), y] = 1
    exp_base_sum = np.sum(exp_base, axis=1)[:, np.newaxis]
    dW += -X.T.dot(mask) + X.T.dot(exp_base/exp_base_sum)
    dW /= num_train
    dW += reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
