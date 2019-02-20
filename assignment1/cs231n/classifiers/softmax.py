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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.dot(X, W)
  scores = np.exp(scores - np.max(scores))
  scores = scores / np.sum(scores, axis=1)[:, np.newaxis] # 'scores' is softmax-ed
  log_scores = np.log(scores)
  N = X.shape[0]
  D, C = W.shape
  for i in range(N):
      Li = - log_scores[i, y[i]] # shape: (D, )
      Si = scores[i] # shape: (C, )
      Si[y[i]] -= 1
      dWi = np.dot(X[i].reshape(D, 1), Si.reshape(1, C))
      loss += Li
      dW += dWi
  loss = loss / N + reg * np.sum(W**2)
  dW = dW / N + 2 * reg * W
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
  N = X.shape[0]
  D, C = W.shape
  scores = np.dot(X, W) # shape: (N, C)
  scores = np.exp(scores - np.max(scores))
  scores = scores / (np.sum(scores, axis=1)[:, np.newaxis]) # 'scores' is softmax-ed
  loss = - np.sum(np.log(scores[np.arange(N), y]))
  scores[np.arange(N), y] -= 1
  dW = np.dot(X.T, scores)
  # consider regularization
  loss = loss / N + reg * np.sum(W ** 2)
  dW = dW / N + 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

