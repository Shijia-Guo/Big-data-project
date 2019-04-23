
# coding: utf-8

# In[1]:


import json
import math
import numpy as np


# In[2]:


file_list2 = ['./expand_5','./expand_6','./expand_7']


# In[16]:


sets = [set(),set(),set()]
i = 0
for file_name in file_list2:
    with open(file_name,'r') as f:
        all_data = f.readlines()
    a = all_data.pop(0).strip().split()
    b = a.pop(-1)
    data_list = []
    for data_piece in all_data:
        data_list.append(data_piece.strip().split()[:-1])
    data_arr = np.array(data_list).astype(np.float32)
    r,c = data_arr.shape
    l = r*c
    mv_col = np.sum(data_arr==-1,axis=0)
    m = np.sum(mv_col)
    col_pos = np.where(mv_col == r)[0].tolist()
    for pos in col_pos:
        sets[i].add(a[pos])
    col_num = np.sum(mv_col == r)
    print('bad features: %d' %(col_num))
    print('bad features rate : %.4f' %(col_num/c))
    print('missing value rate: %.4f' %(m/l))
    i+= 1
for j in range(3):
    for k in range(j+1,3):
        print(len(sets[j] & sets[k]))
print(len(sets[0]&sets[1]&sets[2]))

