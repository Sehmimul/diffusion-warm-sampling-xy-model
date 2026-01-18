import numpy as np

def sum_cosine_bonds(theta):
    """
    theta: numpy array of shape (L, L)
    returns:
        sum_cos_x : sum of cos(angle differences along rows)
        sum_cos_y : sum of cos(angle differences along columns)
    """
    L = theta.shape[0]

    # neighbor in +x direction: (i, j+1)
    theta_right = np.roll(theta, shift=-1, axis=1)
    dtheta_x = theta - theta_right
    sum_cos_x = np.sum(np.cos(dtheta_x))
    sum_sin_x = np.sum(np.sin(dtheta_x))

    # neighbor in +y direction: (i+1, j)
    theta_down = np.roll(theta, shift=-1, axis=0)
    dtheta_y = theta - theta_down
    sum_cos_y = np.sum(np.cos(dtheta_y))
    sum_sin_y = np.sum(np.sin(dtheta_y))

    return sum_cos_x, sum_sin_x, sum_cos_y, sum_sin_y


def ps(samples, beta):
    """
    samples has to be [num_samples, L, L]
    """
    cos_x = []
    sin_x_square = []

    cos_y = []
    sin_y_square = []

    N = samples.shape[1] * samples.shape[2]
    
    for theta in samples:
        sum_cos_x, sum_sin_x, sum_cos_y, sum_sin_y = sum_cosine_bonds(theta)

        cos_x.append(sum_cos_x)
        sin_x_square.append(sum_sin_x ** 2)

        cos_y.append(sum_cos_y)
        sin_y_square.append(sum_sin_y ** 2)

    # now we take the average
    mean_cos_x = np.mean(cos_x)
    mean_sin_x_square = np.mean(sin_x_square)

    mean_cos_y = np.mean(cos_y)
    mean_sin_y_square = np.mean(sin_y_square)


    ps_x = (1/N) * mean_cos_x - beta * (1/N) * mean_sin_x_square
    ps_y = (1/N) * mean_cos_y - beta * (1/N) * mean_sin_y_square

    mean_ps = (ps_x + ps_y) / 2

    return mean_ps, ps_x, ps_y
    
if __name__ == "__main__":
    from rich import print as pprint
    thetas = np.random.uniform(0, 2*np.pi, size=(100, 6,6))
    beta = 0.01
    mean_ps, ps_x, ps_y = ps(samples = thetas, beta = beta)
    print("mean_ps is {}".format(mean_ps))
    print("ps_x is {}".format(ps_x))
    print("ps_y is {}".format(ps_y))
