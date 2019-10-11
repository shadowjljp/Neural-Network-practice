

from sklearn import preprocessing
import numpy as np
import pandas as pd


data = np.array([['', 'Col1', 'Col2'], ['Row1', 1, 2], ['Row2', 3, 4], ['Row3', 5, 6]])

df = pd.DataFrame(data,index=data[:,0])
df1 = pd.DataFrame(data,index=data[:,0])
df3=np.int_(data[1:,1:])

#print(df)
print("---------------------------------------------------")
#print(df1)
print("---------------------------------------------------")
#print(df3)

a = np.array([[1,2,3,7],[4,5,6,8]])
print(np.size(a))
print(np.size(a,1))
print(np.size(a,0))
print("---------------------------------------------------")

mydict = [{'a': 1, 'b': 2, 'c': 3, 'd': 4},
                    {'a': 100, 'b': 200, 'c': 300, 'd': 400},
                    {'a': 1000, 'b': 2000, 'c': 3000, 'd': 4000 }]
df = pd.DataFrame(mydict)
print(df.iloc[1:2])


