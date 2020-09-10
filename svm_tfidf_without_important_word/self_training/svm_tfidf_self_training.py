#!/usr/bin/env python
# coding: utf-8

# # 将原始数据转换为句向量
# 使用tfidf

# In[1]:


import pandas as pd                                            
from sklearn.feature_extraction.text import TfidfVectorizer    
import numpy as np  
import random
from sklearn.svm import SVC, NuSVC
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score


# In[2]:


hour = 1
EXPERIMENT = 1 # 实验号
C_values = [1.0, 10.0, 100.0] # C值

data_raw = pd.read_excel('data/raw_data.xlsx')
sentences = data_raw['sentence'].values
vectorizer = TfidfVectorizer(min_df=4, max_df=0.8)  #使用TfidfVectorizer
features = vectorizer.fit_transform(sentences)      #将使用句子进行训练
# features = features.todense() 稠密的话太慢了
index = data_raw['number'].values 
true_labels = data_raw['true class'].values
# print(index.shape)
# print(features.shape)


# # 读取训练数据函数，测试数据函数

# In[3]:


# 把句子的标签转换成数字，因为老师给的很多文件中的标签是英文，转成数字更容易计算
LABEL_TO_INT = {'Capability':1, 'Usability':2 ,'Security': 3,'Reliability': 4, 'Performance' : 5, 'Lifecycle': 6, 'Software Interface' : 7}
INT_TO_LABEL = {1:'Capability ', 2:'Usability', 3:'Security', 4:'Reliability', 5:'Performance', 6:'Lifecycle', 7:'Software Interface'}

#返回已经标记的数据数组，未标记的数据数组，已经标记的标签
# 最后返回训练集长度
def get_train_data(filePath):
    data = pd.read_excel('data/' + filePath + '/train.xlsx')
    length = len(data)
    train_data_unlabeled = []
    train_data_labeled_dict = {}
    train_data_labeled = []
    
    for i in range(length):
        if isinstance(data['Tag'][i], float):
            train_data_unlabeled.append(int(data['number'][i]))
        else:
            tag_int = LABEL_TO_INT[data['Tag'][i]]
            train_data_labeled.append(int(data['number'][i]))
            train_data_labeled_dict[int(data['number'][i])] = tag_int
            

    return train_data_labeled, train_data_unlabeled, train_data_labeled_dict,length    
    
    
# 返回一位数组，测试数据的下标
def get_test_data(filePath):
    data = pd.read_excel('data/' + filePath + '/test.xlsx')
    return data['number'].values



# # 设置每一个小时能标记的量

# In[4]:


INITIAL_ONE_HOUR_LABELED = [38,29,35,31,30]
INITIAL_TWO_HOUR_LABELED = [82,71,69,68,70]


# # 开始试验

# In[5]:


# 用来存储要放入文件中的数据
list_number = []
list_accuracy = []
list_precision = [[] for i in range(7)]
list_recall = [[] for i in range(7)]
list_f1 = [[] for i in range(7)]
list_f_beta = [[] for i in range(7)]
list_content = []
list_true_class = []
list_predict = []
list_c = []


# In[6]:


# 训练
def train(train_data_labeled, train_data_labeled_dict, model):
    X = features[train_data_labeled]
    Y = [train_data_labeled_dict[i] for i in train_data_labeled]
    model.fit(X, Y)


# In[7]:


# 预测未标记标签, 返回可能性最大的编号(index_max)，预测的标签,和他的真实标记
def predict_unlabeled(train_data_unlabeled, model, train_data_labeled_dict):
    X =features[train_data_unlabeled]
    Y_prob = model.predict_proba(X)
    Y = model.predict(X)
    
    # 获取可能性最大的编号
    row_max = np.max(Y_prob, axis=1)
    index_max = train_data_unlabeled[np.argmax(row_max)]
    
    # 作为预测的标签
    train_data_labeled_dict[index_max] = Y[np.argmax(row_max)]
    
#     print('选取', index_max, '预测标签为',Y[np.argmax(row_max)])
    return index_max,Y[np.argmax(row_max)] ,true_labels[index_max]
    
    


# In[8]:


# 增加数据
def append_data(len_train_data_labeled, accuracy, precision, recall, f1, f_beta, index_max,index_max_pred_class, index_max_true_class, c):
    list_number.append(len_train_data_labeled)
    list_accuracy.append(accuracy) 
    for j in range(7):
        
        list_precision[j].append(precision[j])
        list_recall[j].append(recall[j])
        list_f1[j].append(f1[j])
        list_f_beta[j].append(f_beta[j])
        
    if(index_max != None):    
        list_content.append(sentences[index_max])
        list_predict.append(INT_TO_LABEL[index_max_pred_class])
        list_true_class.append(INT_TO_LABEL[int(true_labels[index_max])])
    else:
        list_content.append(None)
        list_predict.append(None)
        list_true_class.append(None)
        
    list_c.append(c)
    
        


# In[9]:


# 评判指标
def predict_test(test_data, model):
    X = features[test_data]
    Y_true = [true_labels[i] for i in test_data]
    Y = model.predict(X)
    labels_count = Counter(Y_true)
    # 计算betas
    betas = {}
    for k,v in labels_count.items():
        betas[k] = v / len(Y_true)
    betas = sorted(betas.items(), key=lambda a:a[0])
    betas = np.array([item[1] for item in betas])
    precision, recall, _, _ =  precision_recall_fscore_support(Y_true, Y, average=None, labels=[1,2,3,4,5,6,7])
    f1 = 2*(precision* recall)/(precision + recall)
    f_beta = (1+betas)*(precision * recall) / (betas*precision + recall)
    accuracy = accuracy_score(Y_true, Y)
    
    return accuracy, precision, recall, f1, f_beta


# In[10]:


def process(EXPERIMENT):
    if hour == 1:
        nums_of_labeled = INITIAL_ONE_HOUR_LABELED[EXPERIMENT-1]
    if hour == 2:
        nums_of_labeled = INITIAL_TWO_HOUR_LABELED[EXPERIMENT-1]
    train_data_labeled, train_data_unlabeled, train_data_labeled_dict, length = get_train_data('experiment'+str(EXPERIMENT))

    test_data = get_test_data('experiment'+str(EXPERIMENT))
    # 下面抽出出一定的标记样本作为初始样本
    labeled_samples = random.sample(train_data_labeled, nums_of_labeled)

    # 初始化数组写入文件
    np.savetxt('result/' + 'one_hour'+ str(EXPERIMENT)+'.txt', labeled_samples)


    # 将标记和没标记的重新初始化
    data_to_unlabeled = list(set(train_data_labeled).difference(set(labeled_samples)))
    train_data_unlabeled = list(set(train_data_unlabeled).union(set(data_to_unlabeled)))
    train_data_labeled = labeled_samples



    model = SVC(kernel='rbf', probability= True)

    max_accuracy = 0
    best_c = 1.0
    for i in range(3):
        model.set_params(C = C_values[i])
        # 训练
        train(train_data_labeled, train_data_labeled_dict, model)
        # 测试集测试
        accuracy, precision, recall, f1, f_beta = predict_test(test_data, model)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_c = C_values[i]

    model.set_params(C = best_c)          
    train(train_data_labeled, train_data_labeled_dict, model)
    index_max, index_max_pred_class, index_max_true_class = predict_unlabeled(train_data_unlabeled, model, train_data_labeled_dict)
    accuracy, precision, recall, f1, f_beta = predict_test(test_data, model)
    append_data(len(train_data_labeled), accuracy, precision, recall, f1, f_beta, index_max,index_max_pred_class, index_max_true_class, best_c)
    # 获取到可能性最大的样本并加入标记中
    train_data_labeled.append(index_max)
    train_data_unlabeled.remove(index_max)

    while len(train_data_unlabeled) > 0:
        max_accuracy = 0
        for i in range(3):
            model.set_params(C = C_values[i])
            # 训练
            train(train_data_labeled, train_data_labeled_dict, model)
            # 测试集测试
            accuracy, precision, recall, f1, f_beta = predict_test(test_data, model)
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                best_c = C_values[i]
        model.set_params(C = best_c) 
        train(train_data_labeled, train_data_labeled_dict, model) 
        index_max, index_max_pred_class, index_max_true_class = predict_unlabeled(train_data_unlabeled, model, train_data_labeled_dict)
        accuracy, precision, recall, f1, f_beta = predict_test(test_data, model)
        append_data(len(train_data_labeled), accuracy, precision, recall, f1, f_beta, index_max,index_max_pred_class, index_max_true_class, best_c)
        # 获取到可能性最大的样本并加入标记中
        train_data_labeled.append(index_max)
        train_data_unlabeled.remove(index_max)
        print('线程{}选中训练集长度：{}，进度为{:.2f}%'.format(EXPERIMENT, len(train_data_labeled), len(train_data_labeled)/length*100))


    max_accuracy = 0
    for i in range(3):
        model.set_params(C = C_values[i])
        # 训练
        train(train_data_labeled, train_data_labeled_dict, model)
        # 测试集测试
        accuracy, precision, recall, f1, f_beta = predict_test(test_data, model)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_c = C_values[i]

    model.set_params(C = best_c)          

    train(train_data_labeled, train_data_labeled_dict, model)
    accuracy, precision, recall, f1, f_beta = predict_test(test_data, model)
    append_data(len(train_data_labeled), accuracy, precision, recall, f1, f_beta, None,None, None, best_c)


    df = pd.DataFrame()
    df['number'] = list_number
    df['accuracy'] = list_accuracy

    df['1Precision'] = list_precision[0]
    df['1Recall'] = list_recall[0]
    df['1F1'] = list_f1[0]
    df['1F-beta'] = list_f_beta[0]

    df['2Precision'] = list_precision[1]
    df['2Recall'] = list_recall[1]
    df['2F1'] = list_f1[1]
    df['2F-beta'] = list_f_beta[1]

    df['3Precision'] = list_precision[2]
    df['3Recall'] = list_recall[2]
    df['3F1'] = list_f1[2]
    df['3F-beta'] = list_f_beta[2]

    df['4Precision'] = list_precision[3]
    df['4Recall'] = list_recall[3]
    df['4F1'] = list_f1[3]
    df['4F-beta'] = list_f_beta[3]

    df['5Precision'] = list_precision[4]
    df['5Recall'] = list_recall[4]
    df['5F1'] = list_f1[4]
    df['5F-beta'] = list_f_beta[4]

    df['6Precision'] = list_precision[5]
    df['6Recall'] = list_recall[5]
    df['6F1'] = list_f1[5]
    df['6F-beta'] = list_f_beta[5]

    df['7Precision'] = list_precision[6]
    df['7Recall'] = list_recall[6]
    df['7F1'] = list_f1[6]
    df['7F-beta'] = list_f_beta[6]

    df['Content'] = list_content
    df['True Class'] = list_true_class
    df['Predicted'] = list_predict

    df['C'] = list_c

    df.to_excel('result/' + 'experiment'+ str(EXPERIMENT)+'_hour'+str(hour)+'.xlsx', sheet_name='E1')


# In[ ]:


# 开始多线程训练
for i in range(1, 6):
    threading.Thread(target=process, args=(i,)).start()

