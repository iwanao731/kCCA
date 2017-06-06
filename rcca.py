import numpy as np
from scipy.linalg import svd
from sklearn.metrics.pairwise import pairwise_kernels, euclidean_distances
import csv
import pandas as pd

class KCCA(object):
	def __init__(self, n_components=1, epsilon=1.0, kernel="linear", degree=3, gamma=None, coef0=1, n_jobs=1):
		self.n_components = n_components
		self.epsilon = epsilon
		self.kernel = kernel
		self.degree = degree
		self.gamma = gamma
		self.coef0 = coef0
		self.n_jobs = n_jobs

	def _pairwise_kernels(self, X, Y=None):
		return pairwise_kernels(X, Y, metric=self.kernel, filter_params=True, n_jobs=self.n_jobs, degree=self.degree, gamma=self.gamma, coef0=self.coef0)

	def fit(self, X, Y):
		ndata_x, nfeature_x = X.shape	# dimention such as (2, 3)
		ndata_y, nfeature_y = Y.shape
		if ndata_x != ndata_y:
			raise Exception("Inequality of number of data between X and Y")

		print("ndata_x", ndata_x)
		print("nfeature_x", nfeature_x)
		print("nfeature_y", nfeature_y)

		if self.kernel != "precomputed":
			Kx = self._pairwise_kernels(X)
			Ky = self._pairwise_kernels(Y)

		print("Kx \n", Kx)
		print("Ky \n", Ky)

		I = self.epsilon * np.identity(ndata_x)

		print("I \n", I)

		KxI_inv = np.linalg.inv(Kx + I)
		KyI_inv = np.linalg.inv(Ky + I)

		print("KxI_inv \n", KxI_inv)
		print("KyI_inv \n", KyI_inv)

		L = np.dot(KxI_inv, np.dot(Kx, np.dot(Ky, KyI_inv)))

		print("L \n", L)

		U, s, Vh = svd(L)

		print("U \n", U)
		print("s \n", s)
		print("Vh \n", Vh)

		self.alpha = np.dot(KxI_inv, U[:, :self.n_components])
		self.beta = np.dot(KyI_inv, Vh.T[:, :self.n_components])

		print("Fc \n", self.alpha)
		print("Fd \n", self.beta)

		return self

if __name__ == "__main__":

	NP = 15 # Number of pair dataset
	NS = 48	# Number of source
	NT = 14	# Number of target
	NF = 10	# Number of reduced model

	# loading source and target weights files
	## source
	df = pd.read_csv('sourceWeights.csv', header=None)
	X = df.values.astype(float)
	X = X.T
	print("source shape : ", X.shape)
	# print(C)

	## target
	df = pd.read_csv('targetWeights.csv', header=None)
	Y = df.values.astype(float)
	Y = Y.T
	print("target shape : ", Y.shape)
	# print(D)

	# X = np.random.normal(size=(NP,NS))
	# Y = np.random.normal(size=(NP,NT))
	kcca = KCCA(n_components=NF, kernel="linear", n_jobs=1, epsilon=0.1).fit(X, Y)

	"""
	matching on test data
	"""
	Fc = kcca.alpha
	Fd = kcca.beta
	print("Fc", Fc.shape)
	print("Fd", Fd.shape)

	# X_te = np.random.normal(size=(NP,NS))
	# Y_te = np.random.normal(size=(NP,NT))
	X_te = X
	Y_te = Y
	
	# X_te = np.random.normal(size=(NF,NS))
	# Y_te = np.random.normal(size=(NF,NT))

	Kx = kcca._pairwise_kernels(X_te, X)
	Ky = kcca._pairwise_kernels(Y_te, Y)
	print("Kx", Kx.shape)
	print("Ky", Ky.shape)

	Cr = np.dot(Fc.T, Kx) 	# cr
	Dr = np.dot(Fd.T, Ky)	# dr
	print("Cr", Cr.shape)
	print(Cr)
	print("Dr", Dr.shape)
	print(Dr)

	# Bc
	Bc = np.dot(np.linalg.inv(np.dot(Cr, Cr.T)), np.dot(Cr, Dr.T))
	print("Bc", Bc.shape)
	print(Bc)

	# # Hd
	Hd = np.dot(np.linalg.inv(np.dot(Dr, Dr.T)), np.dot(Dr, Y))
	print("Hd", Hd.shape)
	print(Hd)

	# # Md
	Md = np.dot(Hd.T, (np.dot(Bc, Fc.T)))
	print("Md", Md.shape)
	print(Md)

	Xr = (X[1, :])

	print("Xr \n", Xr)

	C_new = kcca._pairwise_kernels(Xr, X).T

	print("C_new, ", C_new.shape)
	# C_new = np.random.random((NP, 1))
	# C_new = np.random.random((NT, 1))
	print((Md).dot(C_new))

	# D = euclidean_distances(Cr, Dr)
	# print("D", D.shape)	# Br

	# idx_pred = np.argmin(D, axis=0)
	# #print("matching result:")
	# print(idx_pred)

	# """
	# similarity between true object and predicted object on test data
	# """
	# idx_true = range(10)
	# C = pairwise_kernels(Y_te[idx_true], Y_te[idx_pred], metric="cosine")
	# print("1-best mean similarity:")
	# final = np.mean(C.diagonal())
	# print(final)
