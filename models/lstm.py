# checked at 2020.9.14
import numpy as np
import time
import keras
import tensorflow as tf
from keras.models import Sequential,Model
from keras.layers import Input,Dense,LSTM,Embedding,Conv1D,MaxPooling1D
from keras.layers import Flatten,Dropout,Concatenate,Lambda,Multiply,Reshape,Dot,Bidirectional
import keras.backend as K
from keras.utils import to_categorical
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score


class LSTM_Basic:
    """
    input->embedding->lstm->softmax_dense
    """
    def __init__(self,maxlen,vocab_size,wvdim,hidden_size,num_classes,embedding_matrix=None):
        text_input =  Input(shape=(maxlen,),name='text_input')
        if embedding_matrix is None: # 不使用pretrained embedding
            input_emb = Embedding(vocab_size+1,wvdim,input_length=maxlen,name='text_emb')(text_input) #(V,wvdim)
        else:
            input_emb = Embedding(vocab_size+1,wvdim,input_length=maxlen,weights=[embedding_matrix],name='text_emb')(text_input) #(V,wvdim)
        input_vec = LSTM(hidden_size)(input_emb)
        pred_probs = Dense(num_classes,activation='softmax',name='pred_probs')(input_vec)
        self.model = Model(inputs=text_input,outputs=pred_probs)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train_val(self,data_package,batch_size,epochs,save_best=False):
        X_train,y_train,X_val,y_val,X_test,y_test = data_package
        best_val_score = 0
        final_test_score = 0
        final_train_score = 0
        val_socre_list = []
        """实验说明：
        每一轮train完，在val上测试，记录其accuracy，
        每当val-acc达到新高，就立马在test上测试，得到test-acc，
        这样最终保留下来的test-acc就是在val上表现最好的模型在test上的accuracy
        """
        for i in range(epochs):
            t1 = time.time()
            self.model.fit(X_train,to_categorical(y_train),batch_size=batch_size,verbose=0,epochs=1)
            pred_probs = self.model.predict(X_val)
            predictions = np.argmax(pred_probs,axis=1)
            val_score = round(accuracy_score(y_val,predictions),5)
            t2 = time.time()
            print('(Orig)Epoch',i+1,'| time: %.3f s'%(t2-t1),'| current val accuracy:',val_score)
            if val_score>best_val_score:
                best_val_score = val_score
                # 使用当前val上最好模型进行test:
                pred_probs = self.model.predict(X_test)
                predictions = np.argmax(pred_probs,axis=1)
                final_test_score = round(accuracy_score(y_test,predictions),5)
                print('  Current Best model! Test score:',final_test_score)
                # 同时记录一下train上的score：
                pred_probs = self.model.predict(X_train)
                predictions = np.argmax(pred_probs, axis=1)
                final_train_score = round(accuracy_score(y_train, predictions),5)
                print('  Current Best model! Train score:', final_train_score)
                if save_best:
                    self.model.save('best_model_lstm.h5')
                    print('  best model saved!')
            val_socre_list.append(val_score)
        return best_val_score,val_socre_list,final_test_score,final_train_score


class LSTM_LS:
    """
    input->embedding->lstm->softmax_dense
    """
    def __init__(self,maxlen,vocab_size,wvdim,hidden_size,num_classes,embedding_matrix=None):
        def ls_loss(y_true, y_pred, e=0.1):
            loss1 = K.categorical_crossentropy(y_true, y_pred)
            loss2 = K.categorical_crossentropy(K.ones_like(y_pred)/num_classes, y_pred)
            return (1-e)*loss1 + e*loss2
        
        text_input =  Input(shape=(maxlen,),name='text_input')
        if embedding_matrix is None: # 不使用pretrained embedding
            input_emb = Embedding(vocab_size+1,wvdim,input_length=maxlen,name='text_emb')(text_input) #(V,wvdim)
        else:
            input_emb = Embedding(vocab_size+1,wvdim,input_length=maxlen,weights=[embedding_matrix],name='text_emb')(text_input) #(V,wvdim)
        input_vec = LSTM(hidden_size)(input_emb)
        pred_probs = Dense(num_classes,activation='softmax',name='pred_probs')(input_vec)
        self.model = Model(inputs=text_input,outputs=pred_probs)
        self.model.compile(loss=ls_loss, optimizer='adam', metrics=['accuracy'])
    

    def train_val(self,data_package,batch_size,epochs,save_best=False):
        X_train,y_train,X_val,y_val,X_test,y_test = data_package
        best_val_score = 0
        final_test_score = 0
        final_train_score = 0
        val_score_list = []
        for i in range(epochs):
            t1 = time.time()
            self.model.fit(X_train,to_categorical(y_train),batch_size=batch_size,verbose=0,epochs=1)
            # val:
            pred_probs = self.model.predict(X_val)
            predictions = np.argmax(pred_probs,axis=1)
            val_score = round(accuracy_score(y_val,predictions),5)
            t2 = time.time()
            print('(LS)Epoch', i + 1, '| time: %.3f s' % (t2 - t1), '| current val accuracy:', val_score)
            if val_score>best_val_score:
                best_val_score = val_score
                # 使用当前val上最好模型进行test:
                pred_probs = self.model.predict(X_test)
                predictions = np.argmax(pred_probs, axis=1)
                final_test_score = round(accuracy_score(y_test, predictions),5)
                print('  Current Best model! Test score:', final_test_score)
                # 同时记录一下train上的score：
                pred_probs = self.model.predict(X_train)
                predictions = np.argmax(pred_probs, axis=1)
                final_train_score = round(accuracy_score(y_train, predictions),5)
                print('  Current Best model! Train score:', final_train_score)
                if save_best:
                    self.model.save('best_model_ls.h5')
                    print('  best model saved!')
            val_score_list.append(val_score)
        return best_val_score,val_score_list,final_test_score,final_train_score


class LSTM_LCM_dynamic:
    """
    LCM dynamic,跟LCM的主要差别在于：
    1.可以设置early stop，即设置在某一个epoch就停止LCM的作用；
    2.在停止使用LCM之后，可以选择是否使用label smoothing来计算loss。
    """
    def __init__(self,maxlen,vocab_size,wvdim,hidden_size,num_classes,alpha,default_loss='ls',text_embedding_matrix=None,label_embedding_matrix=None):
        self.num_classes = num_classes

        def lcm_loss(y_true,y_pred,alpha=alpha):
            pred_probs = y_pred[:,:num_classes]
            label_sim_dist = y_pred[:,num_classes:]
            simulated_y_true = K.softmax(label_sim_dist+alpha*y_true)
            loss1 = -K.categorical_crossentropy(simulated_y_true,simulated_y_true)
            loss2 = K.categorical_crossentropy(simulated_y_true,pred_probs)
            return loss1+loss2
        
        def ls_loss(y_true, y_pred, e=0.1):
            loss1 = K.categorical_crossentropy(y_true, y_pred)
            loss2 = K.categorical_crossentropy(K.ones_like(y_pred)/num_classes, y_pred)
            return (1-e)*loss1 + e*loss2
        
        # basic_predictor:
        text_input =  Input(shape=(maxlen,),name='text_input')
        if text_embedding_matrix is None:
            input_emb = Embedding(vocab_size+1,wvdim,input_length=maxlen,name='text_emb')(text_input) #(V,wvdim)
        else:
            input_emb = Embedding(vocab_size+1,wvdim,input_length=maxlen,weights=[text_embedding_matrix],name='text_emb')(text_input) #(V,wvdim)
        input_vec = LSTM(hidden_size)(input_emb)
        pred_probs = Dense(num_classes,activation='softmax',name='pred_probs')(input_vec)
        self.basic_predictor = Model(input=text_input,output=pred_probs)
        if default_loss == 'ls':
            self.basic_predictor.compile(loss=ls_loss, optimizer='adam')
        else:
            self.basic_predictor.compile(loss='categorical_crossentropy', optimizer='adam')

        # LCM:
        label_input = Input(shape=(num_classes,),name='label_input')
        label_emb = Embedding(num_classes,wvdim,input_length=num_classes,name='label_emb1')(label_input) # (n,wvdim)
        label_emb = Dense(hidden_size,activation='tanh',name='label_emb2')(label_emb)
        # similarity part:
        doc_product = Dot(axes=(2,1))([label_emb,input_vec]) # (n,d) dot (d,1) --> (n,1)
        label_sim_dict = Dense(num_classes,activation='softmax',name='label_sim_dict')(doc_product)
        # concat output:
        concat_output = Concatenate()([pred_probs,label_sim_dict])
        # compile；
        self.model = Model(inputs=[text_input,label_input],outputs=concat_output)
        self.model.compile(loss=lcm_loss, optimizer='adam')

    def my_evaluator(self,model,inputs,label_list):
        outputs = model.predict(inputs)
        pred_probs = outputs[:,:self.num_classes]
        predictions = np.argmax(pred_probs,axis=1)
        acc = round(accuracy_score(label_list,predictions),5)
        # recall = recall_score(label_list,predictions,average='weighted')
        # f1 = f1_score(label_list,predictions,average='weighted')
        return acc

    def train_val(self,data_package,batch_size,epochs,lcm_stop=50,save_best=False):
        X_train,y_train,X_val,y_val,X_test,y_test = data_package
        L_train = np.array([np.array(range(self.num_classes)) for i in range(len(X_train))])
        L_val = np.array([np.array(range(self.num_classes)) for i in range(len(X_val))])
        L_test = np.array([np.array(range(self.num_classes)) for i in range(len(X_test))])
        best_val_score = 0
        final_test_score = 0
        final_train_score = 0
        val_score_list = []
        for i in range(epochs):
            if i < lcm_stop:
                t1 = time.time()
                self.model.fit([X_train,L_train],to_categorical(y_train),batch_size=batch_size,verbose=0,epochs=1)
                val_score = self.my_evaluator(self.model,[X_val,L_val],y_val)
                t2 = time.time()
                print('(LCM)Epoch', i + 1, '| time: %.3f s' % (t2 - t1), '| current val accuracy:', val_score)
                if val_score>best_val_score:
                    best_val_score = val_score
                    # test:
                    final_test_score = self.my_evaluator(self.model,[X_test,L_test],y_test)
                    print('  Current Best model! Test score:',final_test_score)
                    # train:
                    final_train_score = self.my_evaluator(self.model, [X_train, L_train], y_train)
                    print('  Current Best model! Train score:', final_train_score)
                    if save_best:
                        self.model.save('best_model.h5')
                        print('best model saved!')
                val_score_list.append(val_score)
            else: # 停止LCM的作用
                t1 = time.time()
                self.basic_predictor.fit(X_train,to_categorical(y_train),batch_size=batch_size,verbose=0,epochs=1)
                pred_probs = self.basic_predictor.predict(X_val)
                predictions = np.argmax(pred_probs,axis=1)
                val_score = round(accuracy_score(y_val,predictions),5)
                t2 = time.time()
                print('(LCM-stop)Epoch', i + 1, '| time: %.3f s' % (t2 - t1), '| current val accuracy:', val_score)
                if val_score>best_val_score:
                    best_val_score = val_score
                    # test:
                    final_test_score = self.my_evaluator(self.model, [X_test, L_test], y_test)
                    print('  Current Best model! Test score:', final_test_score)
                    # train:
                    final_train_score = self.my_evaluator(self.model, [X_train, L_train], y_train)
                    print('  Current Best model! Train score:', final_train_score)
                    if save_best:
                        self.model.save('best_model_lcm.h5')
                        print('  best model saved!')
                val_score_list.append(val_score)
        return best_val_score,val_score_list,final_test_score,final_train_score




# =================================
# class LSTM_LCM:
#
#     def __init__(self, maxlen, vocab_size, wvdim, hidden_size, num_classes, alpha, text_embedding_matrix=None,
#                  label_embedding_matrix=None):
#         self.num_classes = num_classes
#
#         def lcm_loss(y_true, y_pred, alpha=alpha):
#             pred_probs = y_pred[:, :num_classes]
#             label_sim_dist = y_pred[:, num_classes:]
#             simulated_y_true = K.softmax(label_sim_dist + alpha * y_true)
#             loss1 = -K.categorical_crossentropy(simulated_y_true, simulated_y_true)
#             loss2 = K.categorical_crossentropy(simulated_y_true, pred_probs)
#             return loss1 + loss2
#
#         # text_encoder:
#         text_input = Input(shape=(maxlen,), name='text_input')
#         if text_embedding_matrix is None:
#             input_emb = Embedding(vocab_size + 1, wvdim, input_length=maxlen, name='text_emb')(text_input)  # (V,wvdim)
#         else:
#             input_emb = Embedding(vocab_size + 1, wvdim, input_length=maxlen, weights=[text_embedding_matrix],
#                                   name='text_emb')(text_input)  # (V,wvdim)
#         input_vec = LSTM(hidden_size)(input_emb)
#         pred_probs = Dense(num_classes, activation='softmax', name='pred_probs')(input_vec)
#         # label_encoder:
#         label_input = Input(shape=(num_classes,), name='label_input')
#         if label_embedding_matrix is None:  # 不使用pretrained embedding
#             label_emb = Embedding(num_classes, wvdim, input_length=num_classes, name='label_emb')(
#                 label_input)  # (n,wvdim)
#         else:
#             label_emb = Embedding(num_classes, wvdim, input_length=num_classes, weights=[label_embedding_matrix],
#                                   name='label_emb')(label_input)
#         label_emb = Bidirectional(LSTM(hidden_size, return_sequences=True), merge_mode='ave')(label_emb)  # (n,d)
#         # similarity part:
#         doc_product = Dot(axes=(2, 1))([label_emb, input_vec])  # (n,d) dot (d,1) --> (n,1)
#         label_sim_dict = Dense(num_classes, activation='softmax', name='label_sim_dict')(doc_product)
#         # concat output:
#         concat_output = Concatenate()([pred_probs, label_sim_dict])
#         # compile；
#         self.model = Model(inputs=[text_input, label_input], outputs=concat_output)
#         self.model.compile(loss=lcm_loss, optimizer='adam')
#
#     def my_evaluator(self, model, inputs, label_list):
#         outputs = model.predict(inputs)
#         pred_probs = outputs[:, :self.num_classes]
#         predictions = np.argmax(pred_probs, axis=1)
#         acc = accuracy_score(label_list, predictions)
#         recall = recall_score(label_list, predictions, average='weighted')
#         f1 = f1_score(label_list, predictions, average='weighted')
#         return acc, recall, f1
#
#     def train_val(self, data_package, batch_size, epochs, metric='accuracy', save_best=False):
#         X_train, y_train, X_val, y_val, X_test, y_test = data_package
#         L_train = np.array([np.array(range(self.num_classes)) for i in range(len(X_train))])
#         L_val = np.array([np.array(range(self.num_classes)) for i in range(len(X_val))])
#         L_test = np.array([np.array(range(self.num_classes)) for i in range(len(X_test))])
#         best_val_score = 0
#         learning_curve = []
#         for i in range(epochs):
#             self.model.fit([X_train, L_train], to_categorical(y_train), batch_size=batch_size, verbose=1, epochs=1)
#             acc, recall, f1 = self.my_evaluator(self.model, [X_val, L_val], y_val)
#             if metric == 'accuracy':
#                 score = acc
#                 print('Epoch', i + 1, '| current val %s:' % metric, score)
#             if score > best_val_score:
#                 best_val_score = score
#                 # test:
#                 test_score, test_recall, test_f1 = self.my_evaluator(self.model, [X_test, L_test], y_test)
#                 print('Current Best model! Test score:', test_score, 'current epoch:', i + 1)
#                 if save_best:
#                     self.model.save('best_model.h5')
#                     print('best model saved!')
#             learning_curve.append(score)
#         return best_val_score, learning_curve, test_score
