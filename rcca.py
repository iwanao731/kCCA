import csv
import numpy as np
import pandas as pd
from scipy.linalg import sqrtm

def kernelFunction(a,b):
	w = np.matmul(a.T, b)
	return np.dot(w, w)

def optimizedProjection(Dr, D):
	InvDrDrT = np.linalg.inv(np.matmul(Dr, Dr.T))
	DrDT = np.matmul(Dr, D.T)
	return np.matmul(InvDrDrT, DrDT)

def readTextFiles(filename, start, end):
	sizes = end - start
	FrameData = np.zeros((sizes, 28))
	for i in range(sizes):
		f = pd.read_csv(filename + str(start + i) + '.txt', header=None)
		Array = f.values.astype(float)
		FrameData[i,:] = (Array[:,0])

#================================
# Offline Process
#================================

readTextFiles('Coeff\BSCoeff', 31, 50)

# loading source and target weights files
## source
df = pd.read_csv('sourceWeights.csv', header=None)
C = df.values.astype(float)
print("source shape : ", C.shape)
# print(C)

## target
df = pd.read_csv('targetWeights.csv', header=None)
D = df.values.astype(float)
print("target shape : ", D.shape)
# print(D)

## modified for computing inverse matrix for C and D
C += np.random.random(C.shape) / 10.0
D += np.random.random(D.shape) / 10.0

# define the sourace and target dimention
NS, NP = C.shape
NT, NP = D.shape
NF = 10

# kernelized vector in eq. (6)
Kc = kernelFunction(C,C)	# dubious
Kd = np.matmul(D.T, D)
print("Kd : \n", Kd)

# Fc, Fd
Kc2 = Kc * Kc
print("Kc2 : \n", Kc2)

Kc2 = np.matmul(Kc, Kc) # Kc2 = Kc * Kc
print("Kc2 : \n", Kc2)

Kc2 = np.matmul(Kc, Kc) # Kc2 = Kc * Kc
Kd2 = np.matmul(Kd, Kd) # Kd * Kd
SqrtKc2Kd2 = np.matmul(sqrtm(Kc2), sqrtm(Kd2)) # sqrtm(Kc2) * sqrtm(Kd2)
InvSqrtKc2Kd2 = np.linalg.inv(SqrtKc2Kd2)
UDVT = np.matmul(np.matmul(Kc, Kd), InvSqrtKc2Kd2)
U, s, V = np.linalg.svd(UDVT, full_matrices=True)

Fc = np.matmul(Kc, U)	# dubious
Fd = np.matmul(Kd, V)	# dubious

# reduce
Fc = Fc[:NF, :]
Fd = Fd[:NF, :]
print("Fc : ", Fc.shape)
print("Fd : ", Fd.shape)
# print("Fc : ", Fc)

# Cr, Dr
Cr = np.zeros((NF, NP))
Dr = np.zeros((NF, NP))

kc = np.zeros((NP, NP))
kd = np.zeros((NP, NP))
for i in range(NP):
	for j in range(NP):
		kc[i][j] = kernelFunction(C[:,i],C[:,j])
		kd[i][j] = kernelFunction(D[:,i],D[:,j])

for i in range(NP):
	Cr[:,i] = (np.matmul(Fc, kc[i])).T
	Dr[:,i] = (np.matmul(Fd, kc[i])).T

# Bc
Bc = optimizedProjection(Cr, Dr)
print("Bc", Bc.shape)
print("Bc : ", Bc)

# Hd
Hd = optimizedProjection(Dr, D)
print("Hd", Hd.shape)
print("Hd : ", Hd)

# Md
Md = np.matmul(np.matmul(Hd.T, Bc), Fc)
print("Md", Md.shape)
print(Md)

#================================
# Online Process
#================================

c_new = C[:,5]
# c_new = np.random.random((NS, 1))
print("c_new ; \n", c_new)

kernlized_Cnew = np.zeros((NP,1))
for i in range(NP):
	kernlized_Cnew[i] = kernelFunction(c_new, C[:,i])
# print(kernlized_Cnew)

target = np.matmul(Md, kernlized_Cnew)
print("target : \n", target)
