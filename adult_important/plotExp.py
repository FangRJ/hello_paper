#coding=utf-8
import sys
# sys.path.append("../")
# sys.path.append("D:\\Data\\CODE\\gbdt2pfa-master\\")
sys.path.append("D:\\Data\\CODE\\AICODE\\prism-4.7\\bin\\rnn2automata-master")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from utils.help_func import save_pickle, load_pickle
from utils.constant import *
data_path = get_path("data/adult/adult.data")


def tune_tree_num():
    data = load_pickle(GBDT.ADULT_ONEHOT_DATA)
    X_train, X_test, y_train, y_test = train_test_split(data["X"], data["Y"], random_state=0)
    test_score_list = []
    print("Start loading...")
    for tree_num in range(3, 101, 1):
        clf = GradientBoostingClassifier(n_estimators=tree_num, random_state=0)
        clf.fit(X_train, y_train)
        #预测测试样本的类别
        train_pre_y = clf.predict(X_train).tolist()
        test_pre_y = clf.predict(X_test).tolist()
        #在测试数据集上计算模型精确度
        test_score = clf.score(X_test, y_test)
        test_score_list.append([tree_num, test_score])
    
        
    log_path = get_path('adult_onehot\\log_plot\\test_score.txt')
    with open(log_path, 'w') as f:
        for ts in test_score_list:
            dataline = "{}, {}\n".format(ts[0], ts[1])
            f.write(dataline)
    print("test_score file saved done!".format(tree_num))

def plot_score():
    X = []
    Y = []
    log_path = get_path('adult_onehot\\log_plot\\test_score.txt')
    with open(log_path, "r") as f:
        for line in f.readlines():
            data = line.split(",")
            X.append(int(data[0]))
            Y.append(float(data[1]))
    plt.plot(X, Y, c='k')
    plt.title("GBDT prediction accuracy - number of decision trees")
    plt.xlabel('tree number')
    plt.ylabel('GBDT accuracy')
    plt.savefig('GBDT_acc_tree_num_data_full.jpg')
    plt.show()

# 决策树数目固定,fidelity-k
def plot_fidelity_k(data_type, tree_num):
    X = []
    Y = []
    log_path = get_path("adult_onehot/log_2021_5_3/{}_data_tree_{}.txt".format(data_type, tree_num))
    with open(log_path, "r") as f:
        lines = f.readlines()
        for i in range(2, len(lines)-1):
            data = lines[i].split()
            X.append(int(data[0]))
            Y.append(float(data[3]))
    plt.plot(X, Y, c='k')
    # plt.xlim((0, 100))
    plt.ylim((0, 1.10))
    plt.yticks(np.arange(0, 1.10, 0.05))
    plt.title("PFA fidelity with {} decision tree in {} dataset".format(tree_num, data_type))
    plt.xlabel('K-means clusters number')
    plt.ylabel('PFA fidelity')
    plt.savefig(get_path("adult_onehot/log_plot/PFA_fidelity_{}_data_tree_{}.jpg".format(data_type, tree_num)))
    # plt.show()
    # 清空之前的数据
    plt.cla()

# 聚类参数k的平均指标,fidelity-treenum
def plot_fidelity_tree_num_ave():
    X = []
    Y = []
    data_type = "balanced"
    # data_type = "full"
    for tree_num in GBDT.ADULT_TREE_NUM:
        log_path = get_path("adult_important/log_final/{}_data_tree_{}.txt".format(data_type, tree_num))
        with open(log_path, "r") as f:
            lines = f.readlines()
            # 取最后一行的平均值
            avg_fidelity = float(lines[-1].split()[-1])
            # print(avg_fidelity)
            X.append(tree_num)
            Y.append(avg_fidelity)
    plt.plot(X, Y, c='k')
    plt.xlim((10, 100))
    plt.ylim((0.5, 1.0))
    plt.yticks(np.arange(0.5, 1.0, 0.05))
    plt.title("Filter important tree && Reverse method")
    plt.xlabel('Decision Tree Number of GBDT')
    plt.ylabel('PFA Fidelity')
    # plt.show()
    plt.savefig(get_path("adult_important/log_final/PFA_fidelity_Tree_num.jpg"))
    plt.cla()

# 聚类参数k取10,fidelity-treenum
def plot_fidelity_tree_num(k):
    X = []
    Y = []
    data_type = "balanced"
    # data_type = "full"
    for tree_num in GBDT.ADULT_TREE_NUM:
        log_path = get_path("adult_important/log_final/{}_data_tree_{}.txt".format(data_type, tree_num))
        with open(log_path, "r") as f:
            lines = f.readlines()
            # 取k=10的那行
            line = lines[k+2]
            avg_fidelity = float(lines[k+2].split()[-3])
            # print(avg_fidelity)
            X.append(tree_num)
            Y.append(avg_fidelity)
    plt.plot(X, Y, c='k')
    plt.xlim((10, 100))
    plt.ylim((0.5, 1.0))
    plt.yticks(np.arange(0.5, 1.0, 0.05))
    plt.title("Filter important tree && Reverse method k={}".format(GBDT.ADULT_KMEANS_NUM[k]))
    plt.xlabel('Decision Tree Number of GBDT')
    plt.ylabel('PFA Fidelity')
    # plt.show()
    plt.savefig(get_path("adult_important/log_final/PFA_fidelity_Tree_num_{}.jpg".format(GBDT.ADULT_KMEANS_NUM[k])))
    plt.cla()

if __name__ == "__main__":
    # tune_tree_num()
    # plot_score()
    plot_fidelity_tree_num_ave()
    for k in range(len(GBDT.ADULT_KMEANS_NUM)):
        plot_fidelity_tree_num(k)



