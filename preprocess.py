import pandas as pd


def read_sen_pairs(file):
    data = pd.read_csv(file, sep='\t')
    sen_a_list = [str(sen) for sen in data['sen_a']]
    sen_b_list = [str(sen) for sen in data['sen_b']]
    labels = data['label'].tolist()
    print(len(max(sen_a_list, key=len)))
    print(len(max(sen_b_list, key=len)))
    print(len(sen_a_list))
    return sen_a_list, sen_b_list, labels


read_sen_pairs('data/train_data.tsv')