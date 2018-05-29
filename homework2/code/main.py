import numpy as np
import os
import tqdm

data_path = os.path.join("20news-bydate", "matlab", "train.data")
words_path = os.path.join("20news-bydate", "vocabulary.txt")
stop_words_path = os.path.join("20news-bydate", "stop-words-list-more.txt")
if not os.path.exists('outputs'):
    os.mkdir('outputs')

train_iter_num = 30
K_group = [5, 10, 20, 30]
D = 0
W = 0
T = []
n = []
eps = 1e-10

print("Start reading data.")
words = []
with open(words_path, 'r') as words_file:
    for line in words_file.readlines():
        words.append(line[:-1])

stop_words = set()
with open(stop_words_path, 'r') as stop_words_file:
    # stop_words_ = words_file.readline().split()
    # for stop_word_ in stop_words_:
    #     stop_words.add(stop_word_)
    for line in stop_words_file.readlines():
        stop_words.add(line[:-1])

with open(data_path, 'r') as train_data_file:
    for line in train_data_file.readlines():
        line_split = line.split()
        doc_idx = int(line_split[0]) - 1
        word_idx = int(line_split[1]) - 1
        word_num = int(line_split[2])
        if doc_idx + 1 > len(T):
            T.append([])
            n.append(0)
        if words[word_idx] in stop_words:
            continue
        if W < word_idx + 1:
            W = word_idx + 1
        T[doc_idx].append([word_idx, word_num])
        n[doc_idx] += word_num
    D = doc_idx + 1

def log_sum_exp(log_numerator):
    max_a = np.max(log_numerator)
    return max_a + np.log(np.sum(np.exp(log_numerator - max_a)))

for K in K_group:
    print("\nEM for {} topics:".format(K))
    pi = np.random.rand(K)
    pi = pi / np.sum(pi)
    mu = np.random.rand(W, K)
    mu = mu / np.sum(mu, axis=0)
    gamma = np.random.rand(D, K)
    gamma = gamma / np.sum(gamma, axis=1).reshape(-1, 1)

    for iter in tqdm.tqdm(range(train_iter_num)):
    # for iter in range(train_iter_num):
        # E-step
        # print("\tE-step for the {}/{} iteration.".format(iter, train_iter_num))
        for d in range(D):
            log_numerator = np.zeros(K)
            for k in range(K):
                log_numerator[k] = np.log(pi[k] + eps)
                for word_idx_num in T[d]:
                    log_numerator[k] += word_idx_num[1] * np.log(mu[word_idx_num[0]][k] + eps)
            # numerator = np.exp(log_numerator)
            log_denominator = log_sum_exp(log_numerator)
            for k in range(K):
                gamma[d][k] = np.exp(log_numerator[k] - log_denominator)

        # M-step
        # print("\tM-step for the {}/{} iteration.".format(iter, train_iter_num))
        numerator_pi = np.zeros(K)
        for k in range(K):
            for d in range(D):
                numerator_pi[k] += gamma[d][k]
        denominator_pi = np.sum(numerator_pi)
        for k in range(K):
            pi[k] = numerator_pi[k] / (denominator_pi + eps)

        for k in range(K):
            numerator_mu = np.zeros(W)
            for d in range(D):
                for word_idx_num in T[d]:
                    numerator_mu[word_idx_num[0]] += gamma[d][k] * word_idx_num[1]
            denominator_mu = np.sum(numerator_mu)
            for w in range(W):
                mu[w][k] = numerator_mu[w] / (denominator_mu + eps)

    tmp = {}
    k = 0
    
    with open(os.path.join("outputs", "topics-of-K="+str(K)+".txt"), 'w') as f:
        for d in range(D):
            argmax = np.where(gamma[d] == np.max(gamma[d]))
            pos = argmax[0][0]
            if tmp.get(pos) is None:
                k += 1
                tmp[pos] = k
            f.write("Document {}'s topic: {}\n".format(d, tmp[pos]))

    with open(os.path.join("outputs", "most-frequent-words-of-K="+str(K)+".txt"), 'w') as f:
        for k in range(K):
            sort_mu = -np.sort(-mu[:, k])
            f.write("For topic {}:\n".format(k))
            for i in range(5):
                pos = np.where(mu[:, k] == sort_mu[i])
                f.write("\t'{}': {}\n".format(words[pos[0][0]], sort_mu[i]))
