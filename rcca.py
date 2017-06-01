import numpy as np

def kernelFunction(a,b):
	w = np.dot(a.T,b)
	return np.dot(w,w)

def optimizedProjection(Dr, D):
	InvDrDrT = np.linalg.inv(np.dot(Dr, Dr.T))
	DrDT = np.dot(Dr, D.T)
	return np.dot(InvDrDrT, DrDT)

NS = 48
NT = 14
NP = 15
NF = 10

C = np.random.random((NS, NP))
D = np.random.random((NT, NP))

Kc = kernelFunction(C,C)
Kd = np.dot(D.T, D)
# print(Kc.shape)
# print(Kd.shape)

# Fc, Fd
InvKc = np.linalg.inv(Kc)
InvKd = np.linalg.inv(Kd)
InvKcKc = np.dot(InvKc, Kc)
KdInvKd = np.dot(Kd, InvKd)
UDVT = np.dot(InvKcKc, KdInvKd)
U, s, V = np.linalg.svd(UDVT, full_matrices=True)
Fc = np.dot(Kc, U)
Fd = np.dot(Kd, V)

# reduce
Fc = Fc[:NF, :]
Fd = Fd[:NF, :]
print("Fc : ", Fc.shape)
print("Fd : ", Fd.shape)

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
	Cr[:,i] = (np.dot(Fc, kc[i])).T
	Dr[:,i] = (np.dot(Fd, kc[i])).T

# Bc
Bc = optimizedProjection(Cr, Dr)
print("Bc", Bc.shape)

# Hd
Hd = optimizedProjection(Dr, D)
print("Hd", Hd.shape)

# Md
Md = np.dot(np.dot(Hd.T, Bc), Fc)
print("Md", Md.shape)