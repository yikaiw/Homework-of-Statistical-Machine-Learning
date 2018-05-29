import numpy as np
import os
from sklearn.metrics.cluster import adjusted_mutual_info_score

dir_path = os.path.join("20news-bydate","matlab")
labels_path = os.path.join(dir_path, "train.label")
K_group = [5, 10, 20, 30]
topics_path = [os.path.join("outputs", "topics-of-K={}.txt".format(K)) for K in K_group]

labels = []
with open(labels_path, 'r') as labels_file:
    for line in labels_file.readlines():
        labels.append(int(line[:-1]))

for i in range(len(K_group)):
    topics = []
    with open(topics_path[i], 'r') as topics_file:
        for line in topics_file.readlines():
            topic = line.split()[-1]
            topics.append(int(topic))

    with open(os.path.join("outputs", "evaluation.txt"), 'a') as f:
        ami = adjusted_mutual_info_score(topics, labels)
        f.write("Adjusted Mutual Information score between K={} and labels: {}\n".format(K_group[i], ami))