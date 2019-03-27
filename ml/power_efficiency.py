""" Models FLOP complexity estimation.
"""

def f(x, y, z):
    """Auxiliary function to approximate number of multiplications
        to compute multiply of matrices with sizes of (x, y) and (y, z).

    Args:
        x: An int with first dimension of first matrix.
        y: An int with common dimension of both matrices.
        z: An int with second dimension of second matrix.

    Returns:
        An int with number of multiplications.
    """
    return 2*x*y*z - x*z

def mlp_flops(n_in, L, n):
    """Approximate FLOP complexity for MLP architecture.

    Args:
        n_in: An int with size of input vector (depends on PCA results).
        L: number of fully-connected layers.
        n: number of neurons in each fully-connected layer.

    Returns:
        An int with FLOP complexity.
    """
    return f(1, n_in, n) + f(n_in, n, n) + (L - 2)*f(n, n, n) + f(n, n, 2)

def cnn_flops(L_conv, D, L_fc, n):
    """Approximate FLOP complexity for CNN architecture.

    Args:
        L_conv: number of convolutional layers.
        D: number of filters in each convolutional layer.
        L_fc: number of fully-connected layers.
        n: number of neurons in each fully-connected layer.

    Returns:
        An int with FLOP complexity.
    """
    return 135 + 135*D*(L_conv - 1) + f(1, 15*D, n) + f(15*D, n, n) + \
        (L_fc - 2)*f(n, n, n) + f(n, n, 2)
