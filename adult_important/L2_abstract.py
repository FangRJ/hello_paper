import sys
sys.path.append("../")
# sys.path.append("D:\\Data\\CODE\\AICODE\\prism-4.7\\bin\\rnn2automata-master")
from level2_abstract.aalergia import *
from level2_abstract.read_seq import *
from utils.constant import *


def dp2pfa():
    alpha = 64
    total_symbols = 1000000
    data_source = "train"

    for tree_num in GBDT.ADULT_TREE_NUM:
        for k in GBDT.ADULT_KMEANS_NUM:
            dp_traces_path = GBDT.ADULT_DP_TRACE.format(tree_num, k)
            output_path = GBDT.ADULT_DP_PFA.format(tree_num, k)

            print("***********num_decision_path={}***clusters={}***alpha={}***********".format(tree_num, k, alpha))
            sequence, alphabet = load_trace_data(dp_traces_path, total_symbols)
            print("{}, init".format(current_timestamp()))
            al = AALERGIA(alpha, sequence, alphabet, start_symbol=START_SYMBOL, output_path=output_path,
                        show_merge_info=False)
            print("{}, learing....".format(current_timestamp()))
            dffa = al.learn()
            print("{}, done.".format(current_timestamp()))
            al.output_prism(dffa, data_source)


if __name__ == '__main__':
    dp2pfa()

