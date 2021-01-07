import datetime
import sys
import pandas as pd
import numpy as np
from utils import *
from sklearn.utils import shuffle
from models import lstm

# ========== parameters & dataset: ==========
vocab_size = 20000
maxlen = 100
wvdim = 64
hidden_size = 64
# num_filters = 100
# filter_sizes = [3,10,25] # 不能超过maxlen
alpha = 4 # new model的loss中的alpha
batch_size = 512
epochs = 40
emb_type = ''

log_txt_name = '20NG_log'
num_classes = 20
df1 = pd.read_csv('datasets/20NG/20ng-train-all-terms.csv')
df2 = pd.read_csv('datasets/20NG/20ng-test-all-terms.csv')
df = pd.concat([df1,df2])

# log_txt_name = 'DBPedia' # AG_log
# num_classes = 14
# df1 = pd.read_csv('../datasets/DBPedia/train.csv',header=None,index_col=None)
# df2 = pd.read_csv('../datasets/DBPedia/test.csv',header=None,index_col=None)
# df1.columns = ['label','title','content']
# df2.columns = ['label','title','content']
# df = pd.concat([df1,df2])

# log_txt_name = 'Fudan_log'
# num_classes = 20
# # df = pd.read_csv('../datasets/thucnews_subset.csv')
# df = pd.read_csv('../datasets/fudan_news.csv')


df = df.dropna(axis=0,how='any')
df = shuffle(df)[:50000]
print('data size:',len(df))


# ========== data pre-processing: ==========
labels = sorted(list(set(df.label)))
assert len(labels) == num_classes,'wrong num of classes'
label2idx = {name:i for name,i in zip(labels,range(num_classes))}
num_classes = len(label2idx)

corpus = []
X_words = []
Y = []
i = 0
for content,y in zip(list(df.content),list(df.label)):
    i += 1
    if i%1000 == 0:
        print(i)
    # English:
    content_words = content.split(' ')
    # Chinese:
    # content_words = jieba.lcut(content)
    corpus += content_wordscorpus,vocab_size=vocab_size)
X = text2idx(tokenizer,X_words,maxlen=maxlen)
y = np.array(Y)
    X_words.append(content_words)
    Y.append(label2idx[y])

tokenizer, word_index, freq_word_index = fit_corpus(



# ========== model traing: ==========
old_list = []
ls_list = []
lcm_list = []
N = 10
for n in range(N):
    # randomly split train and test each time:
    np.random.seed(n) # 这样保证了每次试验的seed一致
    random_indexs = np.random.permutation(range(len(X)))
    train_size = int(len(X)*0.6)
    val_size = int(len(X)*0.15)
    X_train = X[random_indexs][:train_size]
    X_val = X[random_indexs][train_size:train_size+val_size]
    X_test = X[random_indexs][train_size+val_size:]
    y_train = y[random_indexs][:train_size]
    y_val = y[random_indexs][train_size:train_size+val_size]
    y_test = y[random_indexs][train_size+val_size:]
    data_package = [X_train,y_train,X_val,y_val,X_test,y_test]


    with open('output/%s.txt'%log_txt_name,'a') as f:
        print(str(datetime.datetime.now()),file=f)
        print('--Round',n+1,file=f)

    print('====Original:============')
    basic_model = lstm.LSTM_Basic(maxlen,vocab_size,wvdim,hidden_size,num_classes,None)
    best_val_score,val_score_list,final_test_score,final_train_score = basic_model.train_val(data_package,batch_size,epochs)
    old_list.append(final_test_score)
    with open('output/%s.txt'%log_txt_name,'a') as f:
        print(n,'old:',final_train_score,best_val_score,final_test_score,file=f)

    print('====LS:============')
    ls_model = lstm.LSTM_LS(maxlen,vocab_size,wvdim,hidden_size,num_classes,None)
    best_val_score,val_score_list,final_test_score,final_train_score = ls_model.train_val(data_package,batch_size,epochs)
    ls_list.append(final_test_score)
    with open('output/%s.txt'%log_txt_name,'a') as f:
        print(n,'ls:',final_train_score,best_val_score,final_test_score,file=f)

    print('====LCM:============')
    dy_lcm_model = lstm.LSTM_LCM_dynamic(maxlen,vocab_size,wvdim,hidden_size,num_classes,alpha,None,None)
    best_val_score,val_score_list,final_test_score,final_train_score = dy_lcm_model.train_val(data_package,batch_size,epochs,lcm_stop=30)
    lcm_list.append(final_test_score)
    with open('output/%s.txt'%log_txt_name,'a') as f:
        print(n,'lcm:',final_train_score,best_val_score,final_test_score,file=f)
