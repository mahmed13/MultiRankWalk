import argparse
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

import sklearn.metrics.pairwise as pairwise

def read_data(filepath):
    Z = np.loadtxt(filepath)
    y = np.array(Z[:, 0], dtype = np.int)  # labels are in the first column
    X = np.array(Z[:, 1:], dtype = np.float)  # data is in all the others
    return [X, y]

def save_data(filepath, Y):
    np.savetxt(filepath, Y, fmt = "%d")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Assignment 4",
        epilog = "CSCI 4360/6360 Data Science II: Fall 2017",
        add_help = "How to use",
        prog = "python assignment4.py -i <input-data> -o <output-file> [optional args]")

    # Required args.
    parser.add_argument("-i", "--infile", required = True,
        help = "Path to an input text file containing the data.")
    parser.add_argument("-o", "--outfile", required = True,
        help = "Path to the output file where the class predictions are written.")

    # Optional args.
    parser.add_argument("-d", "--damping", default = 0.95, type = float,
        help = "Damping factor in the MRW random walks. [DEFAULT: 0.95]")
    parser.add_argument("-k", "--seeds", default = 1, type = int,
        help = "Number of labeled seeds per class to use in initializing MRW. [DEFAULT: 1]")
    parser.add_argument("-t", "--type", choices = ["random", "degree"], default = "random",
        help = "Whether to choose labeled seeds randomly or by largest degree. [DEFAULT: random]")
    parser.add_argument("-g", "--gamma", default = 0.5, type = float,
        help = "Value of gamma for the RBF kernel in computing affinities. [DEFAULT: 0.5]")
    parser.add_argument("-e", "--epsilon", default = 0.01, type = float,
        help = "Threshold of convergence in the rank vector. [DEFAULT: 0.01]")

    args = vars(parser.parse_args())

    # Read in the variables needed.
    outfile = args['outfile']   # File where output (predictions) will be written.
    d = args['damping']         # Damping factor d in the MRW equation.
    k = args['seeds']           # Number of (labeled) seeds to use per class.
    t = args['type']            # Strategy for choosing seeds.
    gamma = args['gamma']       # Gamma parameter in the RBF kernel
    epsilon = args['epsilon']   # Convergence threshold in the MRW iteration.
    # For RBF, see: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.rbf_kernel.html#sklearn.metrics.pairwise.rbf_kernel

    # Read in the data.
    X, y = read_data(args['infile'])
    # reshape y
    y = y.reshape(-1,1)

    #
    print('X shape:',X.shape)
    print('y shape:',y.shape)
    print('dampening d:',d)
    print('k number of seeds:',k)
    print('seed selection:',t)
    print('gamma:',gamma)
    print('epsilon:',epsilon)

    # remove
    #np.random.seed(1) ###

    # rbf
    A = rbf_kernel(X, gamma=gamma)

    D = np.diag(np.sum(A, axis=0))

    W = np.zeros_like(A)
    for i in range(len(A)):
        for j in range(len(A[i])):
            W[i][j] = A[i][j]/D[i][i]

    C = np.unique([y_i for y_i in y if y_i != -1])
    r_all = []

    for c in C:
        u = np.array([1 if y_i == c else 0 for y_i in y])

        if t == 'random':
            # random shuffle and select k number of instances from each class
            index = np.random.choice(u.shape[0], u.shape[0], replace=False)
            index = [i for i in index if u[i] != 0][:k]


        elif t == 'degree':
            degree = np.sum(A, axis=0)#
            degree = np.array([degree[i] if y[i] == c else 0 for i in range(len(degree))])
            index = degree.argsort()[-k:]
        u = np.array([1 if i in index else 0 for i in range(len(u))])

        u = u.reshape(-1, 1)
        # normalize u
        u = u / np.linalg.norm(u)

        # initial r
        r = u
        rnew = (1-d)*u + (d*np.dot(W,r))

        # iterate until r converges
        while abs(np.sum(rnew) - np.sum(r))> epsilon:
            r = rnew
            rnew = (1 - d) * u + (d * np.dot(W, r))

        r_all.append(rnew)


    r_all = np.array(r_all).transpose()[0]

    y_hat = [C[r_i.argmax()] for r_i in r_all]

    save_data(args['outfile'], y_hat)

    if(args['infile'] == 'Z_hard.txt'):
        y_actual = np.loadtxt('y_hard.txt')
    else:
        y_actual = np.loadtxt('y_easy.txt')

        print(y_actual)

    print('Accuracy: {0}'.format((y_hat == y_actual).sum().astype(float) / len(y_hat)))



