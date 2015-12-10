import numpy as np


def categoricalToNumerical(v):
    dict={}
    biggest=0
    res=[]
    for i in v:
        if not i in dict:
            dict[i]=biggest
            biggest+=1
        res.append(dict[i])
    return np.array(res),biggest
