import numpy as np
import matplotlib.pyplot as plt

feature_num = 123
tolerance = 1e-1
eps = 1e-6
maxiter = 10

def Sigmoid(X): 
    return 1. / (1 + np.exp(-X))

def IRLS(X_train, y_train, X_test, y_test, lam = 0):
    accuracies = []
    w = np.mat(np.random.normal(0., 0.1, (feature_num, 1)))
    for _ in range(maxiter):
        u = Sigmoid(X_train.T * w)
        r = np.squeeze(np.array(np.multiply(u, 1 - u)))
        R = np.mat(np.diag(r))
        XRXT = X_train * R * X_train.T
        I = np.mat(np.eye(feature_num))
        delta = (lam * I + eps * I + XRXT).I * (-lam * w + X_train * (y_train - u))
        w += delta

        error = 0.
        for i in range(len(y_test)):
            p = np.exp(w.T * X_test[:, i]) / (1 + np.exp(w.T * X_test[:, i]))
            prediction = 1 if p > 0.5 else 0
            if prediction != y_test[i]:
                error += 1 
        accuracy = 1 - error / len(y_test)
        accuracies.append(accuracy)
        print(accuracy)

        # error = 0.
        # for i in range(len(y_train)):
        #     p = np.exp(w.T * X_train[:, i]) / (1 + np.exp(w.T * X_train[:, i]))
        #     prediction = 1 if p > 0.5 else 0
        #     if prediction != y_train[i]:
        #         error += 1 
        # accuracy = 1 - error / len(y_train)
        # accuracies.append(accuracy)
        # print(accuracy)

        if sum(abs(delta)) < tolerance:
            break
            
        norm = np.linalg.norm(np.squeeze(np.array(w)), ord=2)
    return accuracies, norm

if __name__ == '__main__':
    X_train, y_train = [], []
    training_data = open('a9a')
    for line in training_data:
        X_train.append(np.zeros(feature_num))
        terms = line.split()
        y_train.append(int(terms[0]))
        for i in range(1, len(terms)):
            X_train[-1][int(terms[i].split(':')[0]) - 1] = 1
    training_data.close()
    X_train, y_train = np.mat(X_train).T, np.mat(y_train).T
    y_train[y_train < 0] = 0
    # X_train (123, 32561), y_train (32561, 1))

    X_test, y_test = [], []
    testing_data = open('a9a.t')
    for line in testing_data:
        X_test.append(np.zeros(feature_num))
        terms = line.split()
        y_test.append(int(terms[0]))
        for i in range(1, len(terms)):
            X_test[-1][int(terms[i].split(':')[0]) - 1] = 1
    X_test, y_test = np.mat(X_test).T, np.mat(y_test).T
    y_test[y_test < 0] = 0

    accuracies, norm = IRLS(X_train, y_train, X_test, y_test, lam=100)
    
    fig, ax = plt.subplots()
    ax.plot(accuracies, '-o', ms=20, lw=2, alpha=0.7, mfc='orange')
    ax.grid()
    x_ticks = np.arange(0, len(accuracies))
    plt.xticks(x_ticks)
    # ax.set_title("trainig accuracies without regularization")
    ax.set_title("accuracies with lambda=100")
    plt.show()
    print(norm)
