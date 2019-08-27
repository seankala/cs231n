from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TO-DO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    
    # Input will ultimately be of shape (N, D) with N being the number of samples
    #   and D being the dimension. As stated above, D is simply the dimensions
    #   of the things multiplied.

    N = x.shape[0]

    # Remember that -1 simply means that we're inferring the remaining shape from
    #   the specified shape value. In this case, -1 will refer to d_1 * ... * d_k.
    X = np.reshape(x, (N, -1)) # (N, D)
    out = np.matmul(X, w) # (N, D) x (D, M) = (N, M)
    out += b # (N, M) + (M,) = (N, M)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TO-DO: Implement the affine backward pass.                              #
    ###########################################################################
    
    # Compute nodes of computational graph.
    N = x.shape[0]

    X = np.reshape(x, (N, -1)) # (N, D)
    H = np.matmul(X, w) # (N, D) x (D, M) = (N, M)
    Z = H + b # (N, M) + (M,) = (N, M)

    # Compute backward pass accordingly.
    dZ = dout # (N, M)
    db = np.sum(dZ, axis=0) # (M,) Remember that axis=0 means across the samples.
    dH = dZ # (N, M)
    dw = np.matmul(X.T, dH) # (D, N) x (N, M) = (D, M)
    dX = np.matmul(dH, w.T) # (N, M) x (M, D) = (N, D)

    # Reshape dX to make it dx again.
    dx = np.reshape(dX, x.shape)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TO-DO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    
    out = np.maximum(0, x)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TO-DO: Implement the ReLU backward pass.                                 #
    ###########################################################################

    # If you don't know how to calculate this I strongly recommend reading this
    #   answer:
    # https://math.stackexchange.com/questions/368432/derivative-of-max-function

    dx = dout * (x > 0)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    # Added in by Seankala.
    # These functions represent nodes in 

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TO-DO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #######################################################################

        # Computational graph construction.
        H0 = x # (N, D)
        H1 = np.mean(H0, axis=0) # (D,) Average over all samples (i.e. axis=0).
        H2 = H0 - H1 # (N, D)
        H3 = np.square(H2) # (N, D)
        H4 = np.mean(H3, axis=0) # (D,)
        H5 = H4 + eps # (D,)
        H6 = np.sqrt(H5) # (D,)
        H7 = 1.0 / H6 # (D,)
        H8 = H2 * H7 # (N, D)
        H9 = gamma * H8 # (N, D)
        Z = H9 + beta # (N, D)

        mu = H1
        var = H4
        x_hat = H8
        out = Z

        cache = (x, mu, var, x_hat, gamma, eps)

        # running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        # running_var = momentum * running_var + (1 - momentum) * sample_var

        running_mean = momentum * running_mean + (1 - momentum) * mu
        running_var = momentum * running_var + (1 - momentum) * var

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TO-DO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################

        x_hat = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_hat + beta

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TO-DO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    ###########################################################################

    N, D = dout.shape

    x, mu, var, x_hat, gamma, eps = cache

    # Step 1: dZ -> (N, D)
    dZ = dout

    # Step 2: dbeta -> (D,)
    dZ_beta = 1
    dbeta = dZ * dZ_beta # (N, D)
    dbeta = np.sum(dbeta, axis=0) # (D,)

    # Step 3: dH9 -> (N, D)
    dZ_H9 = 1
    dH9 = dZ * dZ_H9 # (N, D)

    # Step 4: dgamma -> (D,)
    dH9_gamma = x_hat # (N, D)
    dgamma = dH9 * dH9_gamma # (N, D) * (N, D)
    dgamma = np.sum(dgamma, axis=0)

    # Step 5: dH8 -> (N, D)
    dH9_H8 = gamma # (D,)
    dH8 = dH9 * dH9_H8 # (N, D) * (D,)

    # Step 6: dH7 -> (D,)
    dH8_H7 = x - mu # (N, D)
    dH7 = dH8 * dH8_H7 # (N, D) * (N, D)
    dH7 = np.sum(dH7, axis=0) # (D,)

    # Step 7: dH6 -> (D,)
    dH7_H6 = -1.0 / (var + eps) # (D,)
    dH6 = dH7 * dH7_H6 # (D,) * (D,)

    # Step 8: dH5 -> (D,)
    dH6_H5 = 1.0 / (2 * np.sqrt(var + eps)) # (D,)
    dH5 = dH6 * dH6_H5 # (D,) * (D,)

    # Step 9: deps
    dH5_eps = 1 # ()
    deps = np.sum(dH5 * dH5_eps) # ()

    # Step 10: dH4 -> (D,)
    dH5_H4 = 1 # ()
    dH4 = dH5 * dH5_H4 # (D,)

    # Step 11: dH3 -> (N, D)
    dH4_H3 = (1.0 / N) * np.ones(shape=(N, D)) # (N, D)
    dH3 = dH4 * dH4_H3 # (N, D) * (D,)

    # Step 12: dH2 -> (N, D)
    dH8_H2 = 1.0 / np.sqrt(var + eps) # (D,)
    dH3_H2 = 2 * (x - mu) # (N, D)
    dH2 = (dH3 * dH3_H2) + (dH8 * dH8_H2) # (N, D) * (D,) + (N, D) + (D,)

    # Step 13: dH1 -> (D,)
    dH2_H1 = -1.0 # ()
    dH1 = dH2 * dH2_H1 # (N, D)
    dH1 = np.sum(dH1, axis=0) # (D,)

    # Step 14: dH0 -> (N, D)
    dH1_H0 = (1.0 / N) * np.ones(shape=(N, D)) # (N, D)
    dH2_H0 = 1.0 # ()
    dH0 = (dH1 * dH1_H0) + (dH2 * dH2_H0) # (D,) * (N, D) + (N, D) * ()

    dx = dH0

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TO-DO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################

    x, mu, var, x_hat, gamma, eps = cache
    dZ = dout
    N, D = dZ.shape

    dbeta = np.sum(dZ, axis=0)
    dgamma = np.sum(dZ * x_hat, axis=0)

    dH8 = dout * gamma
    frac_var = 1.0 / np.sqrt(var + eps)

    dx = (1.0 / N) * frac_var * (N * dH8 - np.sum(dH8, axis=0) - x_hat * np.sum(dH8 * x_hat, axis=0))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TO-DO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################

        mask = np.random.binomial(1, 1 - p, size=x.shape)
        out = x * mask

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TO-DO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################

        out = x

        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TO-DO: Implement training phase backward pass for inverted dropout   #
        #######################################################################

        # We just pass the gradients of the neurons that weren't "dropped."
        #   Dropout mechanism doesn't exactly drop the weights themselves. Rather,
        #   it drops the activations of the neurons. The mask that is used for the
        #   forward pass is also used for the backward pass.
        dx = dout * mask

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TO-DO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################

    # Go back to Lecture 5 if this is confusing for you.

    N, _, H, W = x.shape
    F, _, HH, WW = w.shape

    pad = conv_param['pad']
    s = conv_param['stride']

    # For each sample, the output is going to be of size (C, H_prime, W_prime).
    H_prime = 1 + (H + 2 * pad - HH) // s
    W_prime = 1 + (W + 2 * pad - WW) // s

    out = np.zeros(shape=(N, F, H_prime, W_prime))

    # We only want to pad the height and width, not the number of samples or channels.
    pad_width = ((0, 0), (0, 0), (pad, pad), (pad, pad))
    x_pad = np.pad(array=x, pad_width=pad_width, mode='constant', constant_values=0)

    for n in range(N):
        sample = x_pad[n, :, :, :]
        for f in range(F):
            filt = w[f, :, :, :]
            bias = b[f]
            for i in range(H_prime):
                for j in range(W_prime):
                    patch = sample[:, (i * s):(i * s + HH), (j * s):(j * s + WW)]
                    prod = patch * filt # Choose which filter we're using and do element-wise mult.
                    output = np.sum(prod) + bias # Sum to complete dot product and add bias term.
                    out[n, f, i, j] = output

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TO-DO: Implement the convolutional backward pass.                        #
    ###########################################################################

    dZ = dout # I prefer dZ over dout.
    x, w, b, conv_param = cache

    N, _, H, W = x.shape
    F, _, HH, WW = w.shape
    _, _, H_prime, W_prime = dZ.shape

    pad = conv_param['pad']
    s = conv_param['stride']

    pad_width = ((0, 0), (0, 0), (pad, pad), (pad, pad))
    x_pad = np.pad(array=x, pad_width=pad_width, mode='constant', constant_values=0)

    dx = np.zeros(shape=x.shape)
    dw = np.zeros(shape=w.shape)
    db = np.zeros(shape=b.shape)
    dx_pad = np.zeros(shape=x_pad.shape)

    for n in range(N):
        sample = x_pad[n, :, :, :]
        for f in range(F):
            filt = w[f, :, :, :]
            db[f] += np.sum(dZ[n, f, :, :])
            for i in range(H_prime):
                for j in range(W_prime):
                    upgrad = dZ[n, f, i, j]
                    patch = sample[:, (i * s):(i * s + HH), (j * s):(j * s + WW)]
                    dw[f, :, :, :] += patch * upgrad
                    dx_pad[n, :, (i * s):(i * s + HH), (j * s):(j * s + WW)] += filt * upgrad

    dx = dx_pad[:, :, pad:(pad + H), pad:(pad + W)] # Take out original data without pads.

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TO-DO: Implement the max pooling forward pass                            #
    ###########################################################################

    N, C, H, W = x.shape

    H_pool = pool_param['pool_height']
    W_pool = pool_param['pool_width']
    s = pool_param['stride']

    H_out = 1 + (H - H_pool) // s
    W_out = 1 + (W - W_pool) // s

    out = np.zeros(shape=(N, C, H_out, W_out))

    for n in range(N):
        sample = x[n, :, :, :]
        for c in range(C):
            channel = sample[c, :, :]
            for h in range(H_out):
                for w in range(W_out):
                    patch = channel[(h * s):(h * s + H_pool), (w * s):(w * s + W_pool)]
                    out[n, c, h, w] = np.max(patch)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TO-DO: Implement the max pooling backward pass                           #
    ###########################################################################

    x, pool_param = cache
    dx = np.zeros(shape=x.shape)
    dZ = dout

    s = pool_param['stride']
    H_pool = pool_param['pool_height']   
    W_pool = pool_param['pool_width']

    N, C, H, W = x.shape
    _, _, H_out, W_out = dZ.shape

    # The np.unravel_index function is what helps us recover the original index of
    #   the max value before conducting pooling.
    # Docs: https://docs.scipy.org/doc/numpy/reference/generated/numpy.unravel_index.html
    # SO Answer: https://stackoverflow.com/questions/48135736/what-is-an-intuitive-explanation-of-np-unravel-index

    for n in range(N):
        sample = x[n, :, :, :]
        for c in range(C):
            channel = sample[c, :, :]
            for h in range(H_out):
                for w in range(W_out):
                    upgrad = dZ[n, c, h, w]
                    patch = channel[(h * s):(h * s + H_pool), (w * s):(w * s + W_pool)]
                    indices = np.argmax(patch)
                    idx = np.unravel_index(indices=indices, shape=patch.shape)
                    dx[n, c, (h * s):(h * s + H_pool), (w * s):(w * s + W_pool)][idx] = upgrad

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TO-DO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TO-DO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
