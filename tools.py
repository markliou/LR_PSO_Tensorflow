import numpy as np
import random
import csv
####
# MIT License is applied
#      2017 markliou
###

def csv_train_test_shuffer_split(FileName, Ratio, y = -1):
    '''
        Give: CSV file, and the ratio use for training and test
        Return : Training features, training labels (paired), test features, test labels (also paired), label set (in set type)
    '''
    CSVFile = open(FileName, 'r')
    reader = csv.reader(CSVFile)
    RowConiainer = []
    Tr_features, Tr_labels = [], []
    Ts_features, Ts_labels = [], []
    label_collect = []
    for row in reader:
        RowConiainer.append(row[:])
    pass
    CSVFile.close()
    
    random.shuffle(RowConiainer)
    
    for row in RowConiainer[0:int(len(RowConiainer)*Ratio)]:
        Tr_labels.append(row[y])
        label_collect.append(row[y])
        del row[y]
        Tr_features.append(row)
    pass
    
    for row in RowConiainer[int(len(RowConiainer)*Ratio):]:
        Ts_labels.append(row[y])
        label_collect.append(row[y])
        del row[y]
        Ts_features.append(row)
    pass
    
    #print(len(Tr_labels), len(Ts_labels))
    return(Tr_features, Tr_labels, Ts_features, Ts_labels, set(label_collect))
pass    

def training_data_generator(Tr_features, Tr_labels):
    assert len(Tr_features) == len(Tr_labels)
    s_index = [i for i in range(len(Tr_labels))]
    ss_index = s_index[:]
    random.shuffle(ss_index)
    while(1):
        if len(ss_index) == 0:
            ss_index = s_index[:]
            random.shuffle(ss_index)
            #print("shuffle epoch!")
        else :
            pop_index = ss_index.pop()
            yield(Tr_features[pop_index], Tr_labels[pop_index])
        pass
    pass
pass

def trans2onehot(label, label_set):
    '''
    This function will translate the label in to one hot label. This can also give the one hot label the hash it used.
    '''
    OneHoty = [0 for i in range(len(label_set))]
    label_hash = {}
    if type(label_set) is set:
        c_index = 0
        for i in label_set:
            label_hash[i] = c_index
            c_index += 1
        pass
    else:
        label_hash = label_set
    pass
    
    OneHoty[label_hash[label]] = 1
    
    return OneHoty, label_hash
pass

def main():
    Tr_features, Tr_labels, Ts_features, Ts_labels, label_set = csv_train_test_shuffer_split('iris.csv', 0.8)
    gen = training_data_generator(Tr_features, Tr_labels)
    for i in range(300):
        fea, lab = gen.__next__()
        OH, _ = trans2onehot(lab, label_set)
        print(OH)
pass

if __name__ == '__main__':
    main()
pass