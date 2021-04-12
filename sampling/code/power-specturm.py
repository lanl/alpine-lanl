#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[16]:


def getkpk(fname):
    #fname = '1drhodm_ps3d.txt'
    df = pd.read_csv(fname, sep=' ', header=None)
    k = df[2].to_numpy()
    pk = df[3].to_numpy()
    return k, pk
fname = 'testrhodm_ps3d.txt'
k,pk = getkpk(fname)

fname = '1drhodm_ps3d.txt'
k_1d,pk_1d = getkpk(fname)

fname = '2drhodm_ps3d.txt'
k_2d,pk_2d = getkpk(fname)


# In[34]:


font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }

plt.figure()
ax = plt.gca()
ax.plot(k,pk,c='k',label='original')
ax.plot(k_1d,pk_1d, c='r', label='1d sampling')
ax.plot(k_2d,pk_2d,c='b', label='2d sampling')

legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
plt.xlabel('k',fontdict=font)
plt.ylabel('p(k)',fontdict=font)
plt.savefig('out.png')
plt.show()


# In[ ]:




