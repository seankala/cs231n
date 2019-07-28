import numpy as np
from random import shuffle
from past.builtins import xrange

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
  
  N = X.shape[0]
  C = W.shape[1]
  D = W.shape[0]

  scores = np.zeros((N,C))

  # Compute scores.
  for i in range(N):
    for j in range(C):
      for k in range(D):
        scores[i, j] += np.dot(X[i, k], W[k, j])

  # Formulate output.
  output = scores
  output = np.exp(output)
  for i in range(N):
    output[i, :] /= np.sum(output[i, :])  #  (N, C)
  
  # Compute loss.
  loss -= np.sum(np.log(output[np.arange(N), y])) 
  loss /= N
  loss += reg * np.sum(W * W)
  
  # Compute gradient. Refer to class notes for more.
  # http://cs231n.github.io/neural-networks-case-study/#grad
  output[np.arange(N), y] -= 1   # (N, C)
 
  for i in range(N):
    for j in range(C):
      for k in range(D):
        dW[k, j] += X[i, k] * output[i, j] 

  # add reg term
  dW /= N
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
  
  N = X.shape[0]
  C = W.shape[1]
  D = W.shape[0]

  scores = np.dot(X, W) # (N, C)
  output = np.exp(scores)

  normalizer = np.sum(output, axis=1, keepdims=True) # Summed exponents.
  output /= normalizer

  # Compute loss.
  loss -= np.sum(np.log(output[np.arange(N), y])) 
  loss /= N
  loss += reg * np.sum(W * W)
  
  # Compute gradient.
  output[np.arange(N), y] -= 1   # (N, C)
  dW += np.dot(X.T, output)

  # add reg term
  dW /= N
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

