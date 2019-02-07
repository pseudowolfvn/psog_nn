""" Models FLOPS complexity estimation.
"""
def f(x, y, z):
    return 2*x*y*z - x*z

def mlp_flops(n_in, L, n):
    return f(1, n_in, n) + f(n_in, n, n) + (L - 2)*f(n, n, n) + f(n, n, 2)

def MLP_flops(i, l, n):
    def calc_flops(m, n, l):
        return 2*m*n*l - m*l
    params = [i] + [n]*l +[2]
    flops = 0
    for p in range(len(params) - 2):
        flops += calc_flops(params[p], params[p + 1], params[p + 2])
    return flops

def cnn_flops(L_conv, D, L_fc, n):
    return 135 + 135*D*(L_conv - 1) + f(1, 15*D, n) + f(15*D, n, n) + \
        (L_fc - 2)*f(n, n, n) + f(n, n, 2)
