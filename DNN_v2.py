
# coding: utf-8

# In[10]:


import numpy as np
import sys
import json


# In[11]:


def process_rawdata(inputfile):
    with open(inputfile,'r') as f:
        lines = f.readlines()
    x_list = []
    y_list = []
    title = lines.pop(0)
    title = title.strip().split()
    title.pop(-1)
    for line in lines:
        data_piece = line.strip().split()
        label = int(data_piece.pop(-1))
        x_list.append(data_piece)
        y_list.append(label)
    x_arr = np.array(x_list).astype(np.float32)
    y_arr = np.array(y_list).astype(np.int32)
    return title, x_arr, y_arr


# In[12]:


def missingvalue_processing(title,x_arr,y_arr,cpg2gene,gene2cpg):
    r,c = x_arr.shape
    mv_ind = np.where(x_arr == -1,1,0)
    mv_overlap = np.sum(mv_ind,axis=0)
    inds_toremove = np.where(mv_overlap == r)[0]
    inds_tomodify = np.where((mv_overlap > 0) & (mv_overlap < r))[0]
    #print(inds_tomodify)
    cpg2ind = {}
    for ind,cpg in enumerate(title):
        cpg2ind[cpg] = ind
    modify_num = 0
    no_gene_num = 0
    guess_num = 0
    for i in range(len(inds_tomodify)):
        col = inds_tomodify[i]
        rows = np.where(mv_ind[:,col] == 1)[0].tolist()
        cpgname = title[col]
        if cpgname in cpg2gene:
            genes = cpg2gene[cpgname]
            allcpg = []
            for gene in genes:
                allcpg = allcpg + gene2cpg[gene]
            valid_ind = []
            for cpg in allcpg:
                if cpg in cpg2ind:
                    valid_ind.append(cpg2ind[cpg])      
            for row in rows:
                approx_v = 0
                temp_row = x_arr[row,valid_ind]
                row_valid = np.where(temp_row >= 0)[0]
                if len(row_valid) == 0:
                    approx_v = 0.5
                    guess_num += 1
                else:
                    approx_v = np.sum(temp_row[row_valid])/len(row_valid)
                    modify_num += 1
                x_arr[row,col] = approx_v
                
        else:
            for row in rows:
                x_arr[row,col] = 0.5
                no_gene_num += 1
    print('total cpg features to remove : %d' %(len(inds_toremove)))
    print('total modify number : %d'%(modify_num))
    print('total random guess number : %d'%(guess_num))
    print('total no gene number : %d'%(no_gene_num))
    new_x = np.delete(x_arr,inds_toremove,axis=1)
    new_title = np.array(title).astype(np.str)
    new_title = np.delete(new_title,inds_toremove).tolist()
    
    return new_x, y_arr, new_title
            


# In[13]:


def split_data(x_arr,y_arr,ratio):
    x_list = x_arr.tolist()
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for i in range(1,7):
        inds = np.where(y_arr == i)[0]
        train_num = int(len(inds)*ratio)
        train_ind = np.random.choice(inds,train_num,False)
        for ind in list(inds):
            if ind in train_ind:
                x_train.append(x_list[ind])
                y_train.append(i-1)
            else:
                x_test.append(x_list[ind])
                y_test.append(i-1)
    x_train = np.array(x_train).astype(np.float32)
    x_test = np.array(x_test).astype(np.float32)
    y_train = np.array(y_train).astype(np.int32)
    y_test = np.array(y_test).astype(np.int32)  
    return x_train, x_test, y_train, y_test


# In[14]:


import tensorflow as tf
from keras import models
from keras.layers import Dense,Dropout,normalization
from keras import optimizers
import keras

# In[15]:


def training(x_train, x_test, y_train, y_test, lrate = 0.01, bsize =5, epoch_num = 10):
    model = models.Sequential()
    #model.add(Dense(5000,activation='relu'))
    #model.add(Dense(512,activation='relu'))
    model.add(Dense(100,activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(6,activation='softmax'))
    sgd = optimizers.SGD(lr=lrate, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
    y_train = keras.utils.to_categorical(y_train, num_classes=6)
    y_test = keras.utils.to_categorical(y_test, num_classes=6)
    model.fit(x=x_train, y=y_train, batch_size=bsize, epochs=epoch_num, verbose=1)
    loss, acc = model.evaluate(x=x_test, y=y_test)
    print('Test accuracy is {:.4f}'.format(acc))


# In[ ]:


def main():
    argvs = sys.argv
    if len(argvs) < 5:
        print('wrong input parameters!')
        sys.exit(1)
    input_filename = argvs[1]
    cpg2gene = argvs[2]
    gene2cpg = argvs[3]
    with open(gene2cpg,'r') as f:
        g2c_dict=json.load(f)
    with open(cpg2gene,'r') as f:
        c2g_dict=json.load(f)
    split_ratio = float(argvs[4])
    if len(argvs) >= 6:
        lrate = float(argvs[5])
    if len(argvs) >=7:
        bsize = int(argvs[6])
    if len(argvs) == 8:
        epoch_num = int(argvs[7])
    title, x_arr, y_arr = process_rawdata(input_filename)
    new_x, new_y, new_title = missingvalue_processing(title, x_arr, y_arr,c2g_dict,g2c_dict)
    x_train, x_test, y_train, y_test = split_data(new_x,new_y,split_ratio)
    if len(argvs) == 5:
        training(x_train, x_test, y_train, y_test)
    elif len(argvs) == 6:
        training(x_train, x_test, y_train, y_test,lrate)
    elif len(argvs) == 7:
        training(x_train, x_test, y_train, y_test,lrate,bsize)
    elif len(argvs) == 8:
        training(x_train, x_test, y_train, y_test,lrate,bsize,epoch_num)
    


# In[ ]:


if __name__ == '__main__':
    main()

