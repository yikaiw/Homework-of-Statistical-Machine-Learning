import matplotlib.pyplot as plt
import numpy as np
import pprint
import util

percent_goal = 0.3
sample = [19, 100]

def plot(X, y, start, label):
    pos = [1, 4, 2, 5, 3, 6]
    for i, num in enumerate(sample):
        plt.subplot(2, 3, pos[start + i])
        plt.axis('off')
        plt.imshow(np.real(X[num]).reshape(28, 28), cmap=plt.cm.gray, interpolation='nearest')
        plt.title(str(int(percent_goal * 100)) + '%: ' + label + str(y[num]))

if __name__ == '__main__':
    # p = 784, X [p * N], U [p * d], Y [d * N], d << p
    train_X, train_y, val_X, val_y = util.load_mnist() 
    # (60000, 784), (60000, ), (10000, 784), (10000, )
    mean = np.mean(train_X, keepdims=True)
    std = np.std(train_X, keepdims=True)
    X_center = ((train_X - mean) / std).T  # (784, 60000)
    X_uncenter = (train_X / std).T 
    plot(train_X, train_y, 0, 'original-')

    for h in range(2):
        X = X_center if h == 0 else X_uncenter
        cov_X = np.cov(X)  # (784, 784)
        eigen_vals, eigen_vecs = np.linalg.eig(cov_X)  # (784,), (784, 784)
        eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
        eigen_pairs.sort(key=lambda x: x[0], reverse=True)
        eigen_vals_sum = np.sum(np.abs(eigen_vals))
        percent_cur = eigen_pairs[0][0] / eigen_vals_sum
        U = eigen_pairs[0][1].reshape(784, 1)
        idx = 1
        while percent_cur < percent_goal:
            percent_cur += eigen_pairs[idx][0] / eigen_vals_sum
            eigen_vec = eigen_pairs[idx][1].reshape(784, 1)
            U = np.concatenate((U, eigen_vec), axis=1)  # (784, d)
            idx += 1
        Y = np.dot(U.T, X)
        if h == 0:
            X_hat = np.dot(U, Y) * std + mean
            label = 'centered-'
        else:
            X_hat = np.dot(U, Y) * std
            label = 'uncentered-'
        plot(X_hat.T, train_y, h * 2 + 2, label)

    plt.show()