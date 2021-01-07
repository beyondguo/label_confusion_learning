from keras.preprocessing.text import text_to_word_sequence,Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from numpy.random import shuffle
import pandas as pd
import jieba
import re
import time
import copy

# ===================preprocessing:==============================
def load_dataset(name):
    assert name in ['20NG','AG','DBP','FDU','THU'], "name only supports '20NG','AG','DBP','FDU','THU', but your input is %s"%name

    if name == '20NG':
        num_classes = 20
        df1 = pd.read_csv('datasets/20NG/20ng-train-all-terms.csv')
        df2 = pd.read_csv('datasets/20NG/20ng-test-all-terms.csv')
        df = pd.concat([df1,df2])
        comp_group = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
                      'comp.windows.x']
        rec_group = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
        talk_group = ['talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
        sci_group = ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space']
        other_group = ['alt.atheism', 'misc.forsale', 'soc.religion.christian']
        label_groups = [comp_group,rec_group,talk_group]
        return df, num_classes,label_groups
    if name == 'AG':
        num_classes = 4  # DBP:14 AG:4
        df1 = pd.read_csv('../datasets/AG_news/train.csv', header=None, index_col=None)
        df2 = pd.read_csv('../datasets/AG_news/test.csv', header=None, index_col=None)
        df1.columns = ['label', 'title', 'content']
        df2.columns = ['label', 'title', 'content']
        df = pd.concat([df1, df2])
        label_groups = []
        return df,num_classes,label_groups
    if name == 'DBP':
        num_classes = 14
        df1 = pd.read_csv('../datasets/DBPedia/train.csv', header=None, index_col=None)
        df2 = pd.read_csv('../datasets/DBPedia/test.csv', header=None, index_col=None)
        df1.columns = ['label', 'title', 'content']
        df2.columns = ['label', 'title', 'content']
        df = pd.concat([df1, df2])
        label_groups = []
        return df,num_classes,label_groups
    if name == 'FDU':
        num_classes = 20
        df = pd.read_csv('../datasets/fudan_news.csv')
        label_groups = []
        return df, num_classes,label_groups
    if name == 'THU':
        num_classes = 13
        df = pd.read_csv('../datasets/thucnews_subset.csv')
        label_groups = []
        return df, num_classes,label_groups

def create_shuffled_labels(y, label_groups, label2idx, rate):
    np.random.seed(0)
    """
    y: the list of label index
    label_groups: each group is a list of label names to shuffle
    rate: noise rate, note that this isn't the error rate, since the some labels may remain unchanged after shuffling
    """
    y_orig = list(y)[:]
    y = np.array(y)
    count = 0
    for labels in label_groups:
        label_ids = [label2idx[l] for l in labels]
        # find out the indexes of your target labels to add noise
        indexes = [i for i in range(len(y)) if y[i] in label_ids]
        shuffle(indexes)
        partial_indexes = indexes[:int(rate*len(indexes))]
        count += len(partial_indexes)
        shuffled_partial_indexes = partial_indexes[:]
        shuffle(shuffled_partial_indexes)
        y[partial_indexes] = y[shuffled_partial_indexes]
    errors = list(np.array(y_orig) & np.array(y)).count(0)
    shuffle_rate = count/len(y)
    error_rate = errors/len(y)
    return shuffle_rate,error_rate,y


def create_asy_noise_labels(y, label_groups, label2idx, rate):
    np.random.seed(0)
    """
    y: the list of label index
    label_groups: each group is a list of label names to exchange within groups
    rate: noise rate, in this mode, each selected label will change to another random label in the same group
    """
    y_orig = list(y)[:]
    y = np.array(y)
    count = 0
    for labels in label_groups:
        label_ids = [label2idx[l] for l in labels]
        # find out the indexes of your target labels to add noise
        indexes = [i for i in range(len(y)) if y[i] in label_ids]
        shuffle(indexes)
        partial_indexes = indexes[:int(rate * len(indexes))]
        count += len(partial_indexes)
        # find out the indexes of your target labels to add noise
        for idx in partial_indexes:
            if y[idx] in label_ids:
                # randomly change to another label in the same group
                other_label_ids = label_ids[:]
                other_label_ids.remove(y[idx])
                y[idx] = np.random.choice(other_label_ids)
    errors = len(y) - list(np.array(y_orig) - np.array(y)).count(0)
    shuffle_rate = count / len(y)
    error_rate = errors / len(y)
    return shuffle_rate, error_rate, y

def remove_punctuations(text):
    return re.sub('[，。：；’‘“”？！、,.!?\'\"\n\t]','',text)


def fit_corpus(corpus,vocab_size=None):
    """
    corpus 为分好词的语料库
    """
    print("Start fitting the corpus......")
    t = Tokenizer(vocab_size) # 要使得文本向量化时省略掉低频词，就要设置这个参数
    tik = time.time()
    t.fit_on_texts(corpus) # 在所有的评论数据集上训练，得到统计信息
    tok = time.time()
    word_index = t.word_index # 不受vocab_size的影响
    print('all_vocab_size',len(word_index))
    print("Fitting time: ",(tok-tik),'s')
    freq_word_index = {}
    if vocab_size is not None:
        print("Creating freq-word_index...")
        x = list(t.word_counts.items())
        s = sorted(x,key=lambda p:p[1],reverse=True)
        freq_word_index = copy.deepcopy(word_index) # 防止原来的字典也被改变了
        for item in s[vocab_size:]:
            freq_word_index.pop(item[0])
        print("Finished!")
    return t,word_index,freq_word_index


def text2idx(tokenizer,text,maxlen):
    """
    text 是一个列表，每个元素为一个文档的分词
    """
    print("Start vectorizing the sentences.......")
    X = tokenizer.texts_to_sequences(text) # 受vocab_size的影响
    print("Start padding......")
    pad_X = pad_sequences(X,maxlen=maxlen,padding='post')
    print("Finished!")
    return pad_X


def create_embedding_matrix(wvmodel,vocab_size,emb_dim,word_index):
    """
    vocab_size 为词汇表大小，一般为词向量的词汇量
    emb_dim 为词向量维度
    word_index 为词和其index对应的查询词典
    """
    embedding_matrix = np.random.uniform(size=(vocab_size+1,emb_dim)) # +1是要留一个给index=0
    print("Transfering to the embedding matrix......")
    # sorted_small_index = sorted(list(small_word_index.items()),key=lambda x:x[1])
    for word,index in word_index.items():
        try:
            word_vector = wvmodel[word]
            embedding_matrix[index] = word_vector
        except Exception as e:
            print(e,"Use random embedding instead.")
    print("Finished!")
    print("Embedding matrix shape:\n",embedding_matrix.shape)
    return embedding_matrix


def label2idx(label_list):
    label_dict = {}
    unique_labels = list(set(label_list))
    for i,each in enumerate(unique_labels):
        label_dict[each] = i
    new_label_list = []
    for label in label_list:
        new_label_list.append(label_dict[label])
    return new_label_list,label_dict