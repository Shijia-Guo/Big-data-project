
# coding: utf-8

# In[19]:


import numpy as np
import sys
import json
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import tensorflow as tf
from keras import models
from keras.layers import Dense,Dropout,normalization
from keras import optimizers
from sklearn.metrics import  classification_report
import keras


# In[29]:



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


def missingvalue_processing(title,x_arr,y_arr,cpg2gene,gene2cpg):
    r,c = x_arr.shape
    mv_ind = np.where(x_arr == -1,1,0)
    mv_overlap = np.sum(mv_ind,axis=0)
    inds_toremove = np.where(mv_overlap == r)[0]
    inds_tomodify = np.where((mv_overlap > 0) & (mv_overlap < r))[0]
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
         

    
def split_data(x_arr,y_arr,ratio):
    x_list = x_arr.tolist()
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for i in range(0,6):
        inds = np.where(y_arr == i)[0]
        train_num = int(len(inds)*ratio)
        train_ind = np.random.choice(inds,train_num,False)
        for ind in list(inds):
            if ind in train_ind:
                x_train.append(x_list[ind])
                y_train.append(i)
            else:
                x_test.append(x_list[ind])
                y_test.append(i)
    x_train = np.array(x_train).astype(np.float32)
    x_test = np.array(x_test).astype(np.float32)
    y_train = np.array(y_train).astype(np.int32)
    y_test = np.array(y_test).astype(np.int32)  
    return x_train, x_test, y_train, y_test


# In[14]:




# In[15]:

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}): 
        self.losses = {'batch':[], 'epoch':[]} 
#         self.accuracy = {'batch':[], 'epoch':[]} 
        self.val_loss = {'batch':[], 'epoch':[]} 
#         self.val_acc = {'batch':[], 'epoch':[]} 
        
    def on_batch_end(self, batch, logs={}): 
        self.losses['batch'].append(logs.get('loss')) 
#         self.accuracy['batch'].append(logs.get('acc')) 
        self.val_loss['batch'].append(logs.get('val_loss')) 
#         self.val_acc['batch'].append(logs.get('val_acc'))
        
    def on_epoch_end(self, batch, logs={}): 
        self.losses['epoch'].append(logs.get('loss')) 
#         self.accuracy['epoch'].append(logs.get('acc')) 
        self.val_loss['epoch'].append(logs.get('val_loss')) 
#         self.val_acc['epoch'].append(logs.get('val_acc')) 
        
    def loss_plot(self, loss_type, name): 
        iters = range(len(self.losses[loss_type])) 
        plt.figure() # acc 
#         plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc') # loss 
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss') 
        if loss_type == 'epoch': # val_acc 
#             plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc') # val_loss 
            plt.plot(iters, self.val_loss[loss_type], 'k', label='test loss') 
        plt.grid(True) 
        plt.xlabel(loss_type) 
        plt.ylabel('loss') 
        plt.legend(loc="upper right") 
#         loss_file_name = name + '.png'
        savefig(name)
        plt.show()
        
def training(x_train, x_test, y_train, y_test, lrate = 0.01, bsize =5, epoch_num = 10):
    initializer = keras.initializers.he_normal()
    model = models.Sequential()
    #model.add(Dense(5000,activation='relu'))
    #model.add(Dense(512,activation='relu'))
    model.add(Dense(100,activation='relu',kernel_initializer=initializer,bias_initializer='zeros'))
    model.add(Dense(32,activation='relu',kernel_initializer=initializer,bias_initializer='zeros'))
#     model.add(Dropout(0.5))
    model.add(Dense(6,activation='softmax'))
    sgd = optimizers.SGD(lr=lrate, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
    y_train = keras.utils.to_categorical(y_train, num_classes=6)
    true_y = y_test.copy()
    y_test = keras.utils.to_categorical(y_test, num_classes=6)
    History = LossHistory()
    model.fit(x=x_train, y=y_train, batch_size=bsize, validation_data=(x_test, y_test), 
              epochs=epoch_num, verbose=1,callbacks=[History],shuffle = True)
    loss, acc = model.evaluate(x=x_test, y=y_test)
    print('Test accuracy is {:.4f}'.format(acc))
    basic_name = 'lr_' + str(int(lrate*1000)) + 'bsize_' + str(bsize) + 'eponum_' + str(epoch_num)
    pic_name = basic_name + '.png'
    History.loss_plot('epoch',pic_name)
    predicted_fc = model.predict(x_test)
    model_name = basic_name + '.h5'
    model.save(model_name)
    report_name = basic_name + '.txt'
    predict_y = np.argmax(predicted_fc,axis=1)
    with open(report_name,'w') as f:
        f.write(classification_report(true_y,predict_y))
    print(classification_report(true_y,predict_y))
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


# In[31]:


# input_filename = './nn_0_5'
# cpg2gene = './cpg2gene.json'
# gene2cpg = './gene2cpg.json'
# split_ratio=0.75
# lrate = 0.001
# bsize = 5
# epoch_num =100
# with open(gene2cpg,'r') as f:
#     g2c_dict=json.load(f)
# with open(cpg2gene,'r') as f:
#     c2g_dict=json.load(f)
# title, x_arr, y_arr = process_rawdata(input_filename)
# new_x, new_y, new_title = missingvalue_processing(title, x_arr, y_arr,c2g_dict,g2c_dict)
# # print(new_y)
# x_train, x_test, y_train, y_test = split_data(new_x,new_y,split_ratio)
# training(x_train, x_test, y_train, y_test,lrate,bsize,epoch_num)

