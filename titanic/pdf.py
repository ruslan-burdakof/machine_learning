import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

try:
    with open('data.pickle', 'rb') as f:
        df= pickle.load(f)
except:
    df = pd.read_csv('train.csv')
    with open('data.pickle', 'wb') as f:
        pickle.dump(df, f)

def getNameCat(name, L, R):
    if rL == np.nan or rR == np.nan:
        return "%s:NaN"%(name)
    return "%s:%.2f_%.2f"%('sda',rL,rR)

def histogram(column, nan=True):
    if nan:
        val = column.dropna()
        hist, bin_edges = np.histogram(val)
        hist = np.array([*hist, len(column) - len(val)])
        bin_edges = np.array([*bin_edges, np.nan])
        return (hist, bin_edges)
    else:
        return np.histogram(column.dropna())

def get_mask(values, L, R):
    if np.isnan(R):
        mask = np.isnan(values)
    else:
        mask = (L<=values) & (values<=R)
    return mask

def mean_for_hist(target, values):
    hist, bin_edges = histogram(values)
    a_mean = []
    for inx,val in enumerate(bin_edges[1:],1):
        R = val
        L = bin_edges[inx-1]
        mask = get_mask(values, L, R)
        a_mean.append(np.sum(target[mask]))
    return a_mean

def get_parse_names_hist(name, bin_edges):
    parse_names = []
    for inx,val in enumerate(bin_edges[1:],1):
        if np.isnan(val):
            parse_names.append(name+':nan')
        else:
            parse_names.append(name+':%.2f'%(0.5*(val+bin_edges[inx-1])))
            
    return parse_names

def get_edges(bin_edges):
    for inx,val in enumerate(bin_edges[1:],1):
        R = val
        L = bin_edges[inx-1]
        yield (L,R)
        
hist, bin_edges = histogram(df.Age)
##import seaborn as sns
##sns.violinplot(x=df.Survived, y=df.Age, data=df)
##plt.show()


surv = mean_for_hist(df.Survived, df.Age)
nameV = get_parse_names_hist('Age',bin_edges)
LenAge = [sum(get_mask(df.Age, *edg)) for edg in get_edges(bin_edges)]
plt.subplot(211)
plt.barh(nameV,LenAge)
plt.barh(nameV,surv)
plt.subplot(212)

sex=df.Sex.copy()
sex[sex=='male']=0
sex = sex.astype(dtype='bool', copy=False)
sexL = mean_for_hist(sex, df.Age)
plt.barh(nameV,LenAge)
plt.barh(nameV,sexL)
plt.show()

print('Age')
print(nameV)
print('Live')
print(LenAge)
print('Female')
print(sexL)
print('Male')
print(np.array(LenAge)-sexL)
##import seaborn as sns
##
##sns.violinplot(x=nameV, y=survV)
##plt.show()
