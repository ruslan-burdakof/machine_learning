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

##sex = df.Sex.array.copy()
##sex[sex=='male']=1
##sex[sex=='female']=0
##sex.astype('bool', copy=False)
##
##sur = df.Survived.array.copy()
##
##psurM = np.zeros(sex.shape, dtype = float)
##psurW = np.zeros(sex.shape, dtype = float)
##sm=0.
##sw=0.
##for inx, s in enumerate(sur):
##    if s==1:
##        if sex[inx]:
##            sm+=1
##        else:
##            sw+=1
##    psurM[inx] = sm/(inx+1)
##    psurW[inx] = sw/(inx+1)




##import seaborn as sns
##sns.violinplot(x=df.Survived, y=df.Age, data=df)
##plt.show()
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


def mean_for_hist(target, values, histogram):
    hist, bin_edges = histogram
    a_mean = []
    print(bin_edges)
    for inx,val in enumerate(bin_edges[1:],1):
        R = val
        print(inx)
        L = bin_edges[inx-1]
        mask=[1,2]
        print("L: %.1f R: %.1f Len: %d"%(L,R,len(mask)))
              
        if np.isnan(R):
            mask = np.isnan(target)
        else:
            mask = (L<=values) & (values<=R)
            print(mask)
        print("L: %.1f R: %.1f Len: %d"%(L,R,len(target[mask])))
        a_mean.append(np.sum(len(target[mask])))
    return a_mean


hist = histogram(df.Age)
surv = mean_for_hist(df.Survived, df.Age, hist)

##mask = df.   column[column.isna()]
##mask = (df.Sex=='male')
##sM = len(sur[mask])
##sF = len(sur)-sM
##L = np.arange(1,len(sur)+1)
##pS = sur.cumsum()/L
##pSM = sur[mask].cumsum()/L[0:sM]
##pSF = sur[-mask].cumsum()/L[0:sF]
##plt.plot(pS)
##plt.plot(pSM)
##plt.plot(pSF)
##plt.
