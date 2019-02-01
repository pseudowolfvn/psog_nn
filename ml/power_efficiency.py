import os

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


if __name__ == '__main__':
    # data = open('.\models_eval\keras_cnn\keras_cnn_grid_search', 'r')
    # models = []
    # for line in data.readlines():
    #     if line.startswith('('):
    #         info = line[1:-2].split(',')
    #         params_list = [int(x) for x in info[-4: ]]
    #         params = calc_keras_parameters_num(
    #             *params_list
    #         )
    #         acc = float(info[0])
    #         if acc > 1.25:
    #             continue
    #         models.append((acc * params, *params_list))
    # models.sort()
    # print(models)
    print(cnn_flops(4, 4, 4, 20))