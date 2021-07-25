import sys
sys.path.append("../")
# sys.path.append("D:\\Data\\CODE\\AICODE\\prism-4.7\\bin\\rnn2automata-master")
from level1_abstract.state_partion import *
from utils.help_func import *
from utils.constant import *
from level1_abstract.clustering_based import *

def my_make_L1_abs_trace(labels, seq_len, y_pre, important_tree_train):
    start_p = 0
    abs_seqs = []
    START_SYMBOL = 'S'
    for y in y_pre:
        abs_trace = labels[start_p:start_p + seq_len]
        abs_trace = abs_trace[0:important_tree_train[start_p//seq_len]+1]
        # 决策树逆序
        abs_trace.reverse()
        TERM_SYMBOL = 'N' if y==0 else 'P'
        abs_trace = [START_SYMBOL] + abs_trace + [TERM_SYMBOL]
        abs_seqs.append(abs_trace)
        start_p += seq_len
    return abs_seqs

def cluster2path(clusters):
    paths = []
    cfs = []
    for cluster in clusters:
        path = []
        cf = ""
        for i in range(3):
            max_abs_value = 0.0
            max_idx = 0
            positive = 1
            compare = "L"
            for j in range(10):
                idx = i * 10 + j
                if abs(cluster[idx]) > max_abs_value:
                    max_abs_value = abs(cluster[idx])
                    max_idx = idx
                    positive = 1 if cluster[idx] >= 0 else -1
                    compare = "H" if cluster[idx] >= 0 else "L"
            single_path = [0] * 10
            single_path[max_idx % 10] = positive
            path.extend(single_path)
            cf += "{}{}".format(compare, max_idx%10)
        paths.append(path)
        cfs.append(cf)
    return paths, cfs

if __name__ == "__main__":
    for tree_num in GBDT.ADULT_TREE_NUM:
        data = load_pickle(GBDT.ADULT_TRAIN_DECISION_PATH.format(tree_num))
        path_seq_list = data["train_decision_path"]
        print("total get {} path".format(len(path_seq_list)))
        train_pre_y = data["train_pre_y"]
        tree_num = data["tree_num"]
        important_tree_train = data["important_tree_train"]
        for k in GBDT.ADULT_KMEANS_NUM:
            partitioner = Kmeans(k)
            partitioner.fit(path_seq_list)
            labels = partitioner.get_fit_labels()
            # 使用决策过程
            # abs_seqs = my_make_L1_abs_trace(labels, 1, train_pre_y)
            # 使用决策路径
            abs_seqs = my_make_L1_abs_trace(labels, tree_num, train_pre_y, important_tree_train)
            save_level1_traces(abs_seqs, GBDT.ADULT_DP_TRACE.format(tree_num, k))
            save_pickle(GBDT.ADULT_PARTITIONER.format(tree_num, k), partitioner)
            print("Level1 abstract trace with {} decision tree on k={} saved successfully!".format(tree_num, k))
            # cluster_centers_dp = partitioner.get_cluster_centers()
            # paths, cfs = cluster2path(cluster_centers_dp)

