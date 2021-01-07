import numpy as np
import keras
import tensorflow as tf
import time
from keras.models import Sequential,Model
from keras.layers import Input,Dense,LSTM,Embedding,Conv1D,MaxPooling1D
from keras.layers import Flatten,Dropout,Concatenate,Lambda,Multiply,Reshape,Dot,Bidirectional
import keras.backend as K
from keras.utils import to_categorical
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr


class BERT_Basic:

    def __init__(self,config_path,checkpoint_path,hidden_size,num_classes,model_type='bert'):
        self.num_classes = num_classes
        bert = build_transformer_model(config_path=config_path,checkpoint_path=checkpoint_path,
                                       model=model_type,return_keras_model=False)
        text_emb = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)
        text_emb = Dense(hidden_size,activation='tanh')(text_emb)
        output = Dense(num_classes,activation='softmax')(text_emb)
        self.model = Model(bert.model.input,output)
        AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')
        self.model.compile(loss='categorical_crossentropy',optimizer=AdamLR(learning_rate=1e-4, lr_schedule={1000: 1,2000: 0.1}))
         
    def train_val(self,data_package,batch_size,epochs,save_best=False):
        X_token_train, X_seg_train, y_train, X_token_val, X_seg_val, y_val, X_token_test, X_seg_test, y_test = data_package
        best_val_score = 0
        test_score = 0
        train_score_list = []
        val_socre_list = []
        """实验说明：
        每一轮train完，在val上测试，记录其accuracy，
        每当val-acc达到新高，就立马在test上测试，得到test-acc，
        这样最终保留下来的test-acc就是在val上表现最好的模型在test上的accuracy
        """
        learning_curve = []
        for i in range(epochs):
            t1 = time.time()
            self.model.fit([X_token_train,X_seg_train],to_categorical(y_train),batch_size=batch_size,verbose=0,epochs=1)
            # record train set result:
            pred_probs = self.model.predict([X_token_train, X_seg_train])
            predictions = np.argmax(pred_probs, axis=1)
            train_score = round(accuracy_score(y_train, predictions), 5)
            train_score_list.append(train_score)
            # validation:
            pred_probs = self.model.predict([X_token_val,X_seg_val])
            predictions = np.argmax(pred_probs,axis=1)
            val_score = round(accuracy_score(y_val, predictions), 5)
            val_socre_list.append(val_score)
            t2 = time.time()
            print('(Orig)Epoch', i + 1, '| time: %.3f s' % (t2 - t1), ' | train acc:', train_score, ' | val acc:', val_score)
            # save best model according to validation & test result:
            if val_score>best_val_score:
                best_val_score = val_score
                print('Current Best model!','current epoch:',i+1)
                # test on best model:
                pred_probs = self.model.predict([X_token_test,X_seg_test])
                predictions = np.argmax(pred_probs, axis=1)
                test_score = round(accuracy_score(y_test, predictions), 5)
                print('  Current Best model! Test score:', test_score)
                if save_best:
                    self.model.save('best_model_bert.h5')
                    print('  best model saved!')
        return train_score_list, val_socre_list, best_val_score, test_score


class BERT_LS:
    def __init__(self, config_path, checkpoint_path, hidden_size, num_classes, ls_e=0.1, model_type='bert'):

        def ls_loss(y_true, y_pred, e=ls_e):
            loss1 = K.categorical_crossentropy(y_true, y_pred)
            loss2 = K.categorical_crossentropy(K.ones_like(y_pred) / num_classes, y_pred)
            return (1 - e) * loss1 + e * loss2

        self.num_classes = num_classes
        bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path,
                                       model=model_type, return_keras_model=False)
        text_emb = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)
        text_emb = Dense(hidden_size, activation='tanh')(text_emb)
        output = Dense(num_classes, activation='softmax')(text_emb)
        self.model = Model(bert.model.input, output)
        AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')
        self.model.compile(loss=ls_loss, optimizer=AdamLR(learning_rate=1e-4, lr_schedule={1000: 1, 2000: 0.1}))

    def train_val(self, data_package, batch_size, epochs, save_best=False):
        X_token_train, X_seg_train, y_train, X_token_val, X_seg_val, y_val, X_token_test, X_seg_test, y_test = data_package
        best_val_score = 0
        test_score = 0
        train_score_list = []
        val_socre_list = []
        """实验说明：
        每一轮train完，在val上测试，记录其accuracy，
        每当val-acc达到新高，就立马在test上测试，得到test-acc，
        这样最终保留下来的test-acc就是在val上表现最好的模型在test上的accuracy
        """
        for i in range(epochs):
            t1 = time.time()
            self.model.fit([X_token_train, X_seg_train], to_categorical(y_train), batch_size=batch_size, verbose=0,
                           epochs=1)
            # record train set result:
            pred_probs = self.model.predict([X_token_train, X_seg_train])
            predictions = np.argmax(pred_probs, axis=1)
            train_score = round(accuracy_score(y_train, predictions), 5)
            train_score_list.append(train_score)
            # validation:
            pred_probs = self.model.predict([X_token_val, X_seg_val])
            predictions = np.argmax(pred_probs, axis=1)
            val_score = round(accuracy_score(y_val, predictions), 5)
            val_socre_list.append(val_score)
            t2 = time.time()
            print('(LS)Epoch', i + 1, '| time: %.3f s' % (t2 - t1), ' | train acc:', train_score, ' | val acc:',
                  val_score)
            # save best model according to validation & test result:
            if val_score > best_val_score:
                best_val_score = val_score
                print('Current Best model!', 'current epoch:', i + 1)
                # test on best model:
                pred_probs = self.model.predict([X_token_test, X_seg_test])
                predictions = np.argmax(pred_probs, axis=1)
                test_score = round(accuracy_score(y_test, predictions), 5)
                print('  Current Best model! Test score:', test_score)
                if save_best:
                    self.model.save('best_model_bert_ls.h5')
                    print('  best model saved!')
        return train_score_list, val_socre_list, best_val_score, test_score


class BERT_LCM:
    def __init__(self,config_path,checkpoint_path,hidden_size,num_classes,alpha,wvdim=768,model_type='bert',label_embedding_matrix=None):
        self.num_classes = num_classes

        def lcm_loss(y_true,y_pred,alpha=alpha):
            pred_porbs = y_pred[:,:num_classes]
            label_sim_dist = y_pred[:,num_classes:]
            simulated_y_true = K.softmax(label_sim_dist+alpha*y_true)
            loss1 = -K.categorical_crossentropy(simulated_y_true,simulated_y_true)
            loss2 = K.categorical_crossentropy(simulated_y_true,pred_probs)
            return loss1+loss2

        def ls_loss(y_true, y_pred, e=0.1):
            loss1 = K.categorical_crossentropy(y_true, y_pred)
            loss2 = K.categorical_crossentropy(K.ones_like(y_pred)/num_classes, y_pred)
            return (1-e)*loss1 + e*loss2     

        # text_encoder:
        bert = build_transformer_model(config_path=config_path,checkpoint_path=checkpoint_path,
                                       model=model_type,return_keras_model=False)
        text_emb = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)
        text_emb = Dense(hidden_size,activation='tanh')(text_emb)
        pred_probs = Dense(num_classes,activation='softmax')(text_emb)
        self.basic_predictor = Model(bert.model.input,pred_probs)
        AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')
        self.basic_predictor.compile(loss='categorical_crossentropy',optimizer=AdamLR(learning_rate=1e-4, lr_schedule={1000: 1,2000: 0.1}))


        # label_encoder:
        label_input = Input(shape=(num_classes,),name='label_input')
        if label_embedding_matrix is None: # 不使用pretrained embedding
            label_emb = Embedding(num_classes,wvdim,input_length=num_classes,name='label_emb1')(label_input) # (n,wvdim)
        else:
            label_emb = Embedding(num_classes,wvdim,input_length=num_classes,weights=[label_embedding_matrix],name='label_emb1')(label_input)
#         label_emb = Bidirectional(LSTM(hidden_size,return_sequences=True),merge_mode='ave')(label_emb) # (n,d)
        label_emb = Dense(hidden_size,activation='tanh',name='label_emb2')(label_emb)
                
        # similarity part:
        doc_product = Dot(axes=(2,1))([label_emb,text_emb]) # (n,d) dot (d,1) --> (n,1)
        label_sim_dict = Dense(num_classes,activation='softmax',name='label_sim_dict')(doc_product)
        # concat output:
        concat_output = Concatenate()([pred_probs,label_sim_dict])
        # compile；
        AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')
        self.model = Model(bert.model.input+[label_input],concat_output)
        self.model.compile(loss=lcm_loss, optimizer=AdamLR(learning_rate=1e-4, lr_schedule={1000: 1,2000: 0.1}))


    def lcm_evaluate(self,model,inputs,y_true):
        outputs = model.predict(inputs)
        pred_probs = outputs[:,:self.num_classes]
        predictions = np.argmax(pred_probs,axis=1)
        acc = round(accuracy_score(y_true,predictions),5)
        return acc

    def train_val(self, data_package, batch_size,epochs,lcm_stop=50,save_best=False):
        X_token_train, X_seg_train, y_train, X_token_val, X_seg_val, y_val, X_token_test, X_seg_test, y_test = data_package
        best_val_score = 0
        test_score = 0
        train_score_list = []
        val_socre_list = []
        """实验说明：
        每一轮train完，在val上测试，记录其accuracy，
        每当val-acc达到新高，就立马在test上测试，得到test-acc，
        这样最终保留下来的test-acc就是在val上表现最好的模型在test上的accuracy
        """

        L_train = np.array([np.array(range(self.num_classes)) for i in range(len(X_token_train))])
        L_val = np.array([np.array(range(self.num_classes)) for i in range(len(X_token_val))])
        L_test = np.array([np.array(range(self.num_classes)) for i in range(len(X_token_test))])

        for i in range(epochs):
            t1 = time.time()
            if i < lcm_stop:
                self.model.fit([X_token_train,X_seg_train,L_train],to_categorical(y_train),batch_size=batch_size,verbose=0,epochs=1)
                # record train set result:
                train_score = self.lcm_evaluate(self.model,[X_token_train,X_seg_train,L_train],y_train)
                train_score_list.append(train_score)
                # validation:
                val_score = self.lcm_evaluate(self.model,[X_token_val,X_seg_val,L_val],y_val)
                val_socre_list.append(val_score)
                t2 = time.time()
                print('(LCM)Epoch', i + 1, '| time: %.3f s' % (t2 - t1), ' | train acc:', train_score, ' | val acc:',
                      val_score)
                # save best model according to validation & test result:
                if val_score > best_val_score:
                    best_val_score = val_score
                    print('Current Best model!', 'current epoch:', i + 1)
                    # test on best model:
                    test_score = self.lcm_evaluate(self.model,[X_token_test,X_seg_test,L_test],y_test)
                    print('  Current Best model! Test score:', test_score)
                    if save_best:
                        self.model.save('best_model_bert_lcm.h5')
                        print('  best model saved!')
            else:
                self.basic_predictor.fit([X_token_train,X_seg_train],to_categorical(y_train),batch_size=batch_size,verbose=0,epochs=1)
                # record train set result:
                pred_probs = self.basic_predictor.predict([X_token_train, X_seg_train])
                predictions = np.argmax(pred_probs, axis=1)
                train_score = round(accuracy_score(y_train, predictions),5)
                train_score_list.append(train_score)
                # validation:
                pred_probs = self.basic_predictor.predict([X_token_val, X_seg_val])
                predictions = np.argmax(pred_probs, axis=1)
                val_score = round(accuracy_score(y_val, predictions),5)
                val_socre_list.append(val_score)
                t2 = time.time()
                print('(LCM_stopped)Epoch', i + 1, '| time: %.3f s' % (t2 - t1), ' | train acc:', train_score, ' | val acc:',
                      val_score)
                # save best model according to validation & test result:
                if val_score > best_val_score:
                    best_val_score = val_score
                    print('Current Best model!', 'current epoch:', i + 1)
                    # test on best model:
                    pred_probs = self.basic_predictor.predict([X_token_test, X_seg_test])
                    predictions = np.argmax(pred_probs, axis=1)
                    test_score = round(accuracy_score(y_test, predictions),5)
                    print('  Current Best model! Test score:', test_score)
                    if save_best:
                        self.model.save('best_model_bert_lcm.h5')
                        print('  best model saved!')
        return train_score_list, val_socre_list, best_val_score, test_score