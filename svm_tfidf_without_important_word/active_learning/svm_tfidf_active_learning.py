# 文件说明
# 本文件是用于svm+tfidf+active learning 
# 对于不确定性衡量方法要通过修改参数func分别进行实验 需要进行三次实验分别为1，2，3


# 安装依赖，如果不是第一次跑可以忽略
# get_ipython().system('pip install pandas')
# get_ipython().system('pip install sklearn')
# get_ipython().system('pip install xlrd')
# get_ipython().system('pip install openpyxl')



# 导入依赖
import pandas as pd                                            
from sklearn.feature_extraction.text import TfidfVectorizer    
import numpy as np  
import random
from sklearn.svm import SVC, NuSVC
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import threading



# 参数设置
EXPERIMENT = 1 # 第几折
C_values = [1.0, 10.0, 100.0] # C值
func = 1 # 采取第几种方法



# 读取训练数据
data_raw = pd.read_excel('data/raw_data.xlsx')        # raw data是一个excel。包含三列，句子编号，句子，和真实标签
sentences = data_raw['sentence'].values               # 所有的句子
vectorizer = TfidfVectorizer(min_df=4, max_df=0.8)    #使用TfidfVectorizer
features = vectorizer.fit_transform(sentences)        #将使用句子进行训练
# features = features.todense()                       #稠密的话太慢了
index = data_raw['number'].values                     #句子的编号
true_labels = data_raw['true class'].values           #句子的真实标签


# 读取训练数据函数，测试数据函数


# 把句子的标签转换成数字，因为老师给的很多文件中的标签是英文，转成数字更容易计算
LABEL_TO_INT = {'Capability':1, 'Usability':2 ,'Security': 3,'Reliability': 4, 'Performance' : 5, 'Lifecycle': 6, 'Software Interface' : 7}
INT_TO_LABEL = {1:'Capability ', 2:'Usability', 3:'Security', 4:'Reliability', 5:'Performance', 6:'Lifecycle', 7:'Software Interface'}
# 得到训练集下标
def get_train_data(filePath):
    data = pd.read_excel('data/' + filePath + '/train.xlsx')
    return data['number'].values  
    
# 返回一维数组，测试数据的下标
def get_test_data(filePath):
    data = pd.read_excel('data/' + filePath + '/test.xlsx')
    return data['number'].values



# 开始试验


# 用来存储要放入文件中的数据，可以参照老师给出的结果文件模版，这里用数组存储每一步实验后的结果
list_number = []
list_accuracy = []
list_precision = [[] for i in range(7)]
list_recall = [[] for i in range(7)]
list_f1 = [[] for i in range(7)]
list_f_beta = [[] for i in range(7)]
list_content = []
list_true_class = []
list_c = []


# 训练，每一步实验只要把数据和模型投喂进来即可
def train(train_data_selected, model):
    X = features[train_data_selected]
    Y = [true_labels[i] for i in train_data_selected]
    model.fit(X, Y)


# 预测未标记标签, 返回即将选择的的编号(index_will_select)，和他的真实标记
def predict_unselected(train_data_unselected ,model):
    X =features[train_data_unselected]
    Y_prob = model.predict_proba(X)
    Y = model.predict(X)
    
    # 方法1：所有数据中，预测的各个类型上概率最大的那个类型的概率最小的一个
    if func == 1:
        row_max = np.max(Y_prob, axis=1)
        index_will_select = train_data_unselected[np.argmin(row_max)]
    
    # 方法2：所有数据中，预测的各个类型上概率第一大和第二大之间差最小的一个
    if func == 2:
        Y_prob.sort(axis = 1)
        row_max = Y_prob[:,-1]
        row_max_2 = Y_prob[:,-2]
        row_diff = row_max - row_max_2
        index_will_select = train_data_unselected[np.argmin(row_diff)]
    
    # 方法3：随机选一个
    if func == 3:
        index_will_select = random.sample(train_data_unselected, 1)
    
    return index_will_select, true_labels[index_will_select]



# 增加数据到数组，方便后面写入
def append_data(len_train_data_selected, accuracy, precision, recall, f1, f_beta, index_selected, index_selected_true_class, c):
    list_number.append(len_train_data_selected)
    list_accuracy.append(accuracy) 
    for j in range(7):
        
        list_precision[j].append(precision[j])
        list_recall[j].append(recall[j])
        list_f1[j].append(f1[j])
        list_f_beta[j].append(f_beta[j])
    if(index_selected != None):    
        list_content.append(sentences[index_selected])
        list_true_class.append(INT_TO_LABEL[int(true_labels[index_selected])])
    else:
        list_content.append(None)
        list_true_class.append(None)
        
    list_c.append(c)
         
        



# 评判指标，这里labels只针对当前数据集，即包含七个类别
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



def process(EXPERIMENT):
    train_data = get_train_data('experiment'+str(EXPERIMENT))
    print('线程{}训练集总长度：{}'.format(EXPERIMENT,len(train_data)))
    test_data = get_test_data('experiment'+str(EXPERIMENT))
    # 下面抽出出一定的标记样本作为初始样本
    selected_samples = random.sample(list(train_data), 2)
    set_selected_samples_class = set([true_labels[i] for i in selected_samples])
    while len(set_selected_samples_class) == 1:
        selected_samples = random.sample(list(train_data), 2)
        set_selected_samples_class = set([true_labels[i] for i in selected_samples])


    # 初始化数组写入文件
    np.savetxt('result/' + 'experiment'+ str(EXPERIMENT)+'_func'+str(func)+'.txt', selected_samples)
    # 将标记和没标记的重新初始化
    train_data_selected = []
    train_data_selected.extend(selected_samples)
    train_data_unselected = list(set(train_data).difference(set(train_data_selected)))
    print('线程{}选中训练集长度：{}，进度为{:.2f}%'.format(EXPERIMENT,len(train_data_selected), len(train_data_selected)/len(train_data)*100))


    model = SVC(kernel='rbf', probability= True)
    max_accuracy = 0
    best_c = 1.0
    for i in range(3):
        model.set_params(C = C_values[i])
        # 训练
        train(train_data_selected, model)
        # 测试集测试
        accuracy, precision, recall, f1, f_beta = predict_test(test_data, model)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_c = C_values[i]
    model.set_params(C = best_c)          
    train(train_data_selected, model)   
    index_will_selected, index_will_selected_true_class = predict_unselected(train_data_unselected, model)

    accuracy, precision, recall, f1, f_beta = predict_test(test_data, model)
    append_data(len(train_data_selected), accuracy, precision, recall, f1, f_beta, index_will_selected, index_will_selected_true_class, best_c)

    train_data_selected.append(index_will_selected)
    train_data_unselected.remove(index_will_selected)

    while len(train_data_unselected) > 0:
        max_accuracy = 0
        for i in range(3):
            model.set_params(C = C_values[i])
            # 训练
            train(train_data_selected,  model)
            # 测试集测试
            accuracy, precision, recall, f1, f_beta = predict_test(test_data, model)
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                best_c = C_values[i]
        model.set_params(C = best_c) 
        train(train_data_selected, model)   
        index_will_selected, index_will_selected_true_class = predict_unselected(train_data_unselected, model)
        accuracy, precision, recall, f1, f_beta = predict_test(test_data, model)
        append_data(len(train_data_selected), accuracy, precision, recall, f1, f_beta, index_will_selected, index_will_selected_true_class, best_c)

        train_data_selected.append(index_will_selected)
        train_data_unselected.remove(index_will_selected)
        print('线程{}选中训练集长度：{}，进度为{:.2f}%'.format(EXPERIMENT,len(train_data_selected), len(train_data_selected)/len(train_data)*100))

    max_accuracy = 0
    for i in range(3):
        model.set_params(C = C_values[i])
        # 训练
        train(train_data_selected,  model)
        # 测试集测试
        accuracy, precision, recall, f1, f_beta = predict_test(test_data, model)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_c = C_values[i]
    model.set_params(C = best_c) 
    train(train_data_selected, model)  
    accuracy, precision, recall, f1, f_beta = predict_test(test_data, model)
    append_data(len(train_data_selected), accuracy, precision, recall, f1, f_beta, None, None, best_c)

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


    df['C'] = list_c

    df.to_excel('result/' + 'experiment'+ str(EXPERIMENT)+'_func'+str(func)+'.xlsx', sheet_name='E1')



# 开始多线程训练
for i in range(1, 6):
    threading.Thread(target=process, args=(i,)).start()







