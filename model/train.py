import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from numpy import genfromtxt
from sklearn.svm import NuSVR
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn import tree
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor

def scaled_data(file_name):		# rescales the data so that all the features lie in similar ranges
	data = genfromtxt(file_name, delimiter=',')
	scalar = StandardScaler()
	data_normalized = scalar.fit_transform(data[:, 1 : -1])
	return np.concatenate((data[:, 0 : 1], data_normalized, data[:, -1 :]), axis = 1)

def linear_regression(data):	# with 1-hold out validation
	predicted_speedups = np.zeros((data.shape[0]))
	for row_idx in range(0, data.shape[0]):
		training_x = np.concatenate((data[0 : row_idx, 1 : -1], data[row_idx + 1 :, 1 : -1]), axis = 0)
		training_y = np.concatenate((data[0 : row_idx, -1], data[row_idx + 1 :, -1]), axis = 0)
		reg = Ridge(alpha = 3).fit(training_x ,training_y)
		test_x = data[row_idx : row_idx + 1, 1 : -1]
		predicted_speedups[row_idx] = reg.predict(test_x)
		if row_idx == 0:
			print(reg.coef_)
	return predicted_speedups

def gaussian_proces_regressor(data):			# with 1-hold out validation
	predicted_speedups = np.zeros((data.shape[0]))
	for row_idx in range(0, data.shape[0]):
		training_x = np.concatenate((data[0 : row_idx, 1 : -1], data[row_idx + 1 :, 1 : -1]), axis = 0)
		training_y = np.concatenate((data[0 : row_idx, -1], data[row_idx + 1 :, -1]), axis = 0)
		kernel = DotProduct() + WhiteKernel()
		reg = GaussianProcessRegressor(kernel = kernel, random_state = 0).fit(training_x ,training_y)
		test_x = data[row_idx : row_idx + 1, 1 : -1]
		predicted_speedups[row_idx] = reg.predict(test_x)
	return predicted_speedups

def knn(data, k = 5):							# with 1-hold out validation
	predicted_speedups = np.zeros((data.shape[0]))
	for row_idx in range(0, data.shape[0]):
		training_x = np.concatenate((data[0 : row_idx, 1 : -1], data[row_idx + 1 :, 1 : -1]), axis = 0)
		training_y = np.concatenate((data[0 : row_idx, -1], data[row_idx + 1 :, -1]), axis = 0)
		reg = KNeighborsRegressor(n_neighbors = k).fit(training_x ,training_y)
		test_x = data[row_idx : row_idx + 1, 1 : -1]
		predicted_speedups[row_idx] = reg.predict(test_x)
	return predicted_speedups

def support_vector_regression(data):			# with 1-hold out validation
	predicted_speedups = np.zeros((data.shape[0]))
	for row_idx in range(0, data.shape[0]):
		training_x = np.concatenate((data[0 : row_idx, 1 : -1], data[row_idx + 1 :, 1 : -1]), axis = 0)
		training_y = np.concatenate((data[0 : row_idx, -1], data[row_idx + 1 :, -1]), axis = 0)
		reg = SVR(gamma = 0.01, C = 1.0, epsilon = 0.2, kernel = 'rbf').fit(training_x, training_y)
		test_x = data[row_idx : row_idx + 1, 1 : -1]
		predicted_speedups[row_idx] = reg.predict(test_x)
	return predicted_speedups

def random_forrest_regressor(data, n = 100):	# with 1-hold out validation
	predicted_speedups = np.zeros((data.shape[0]))
	for row_idx in range(0, data.shape[0]):
		training_x = np.concatenate((data[0 : row_idx, 1 : -1], data[row_idx + 1 :, 1 : -1]), axis = 0)
		training_y = np.concatenate((data[0 : row_idx, -1], data[row_idx + 1 :, -1]), axis = 0)
		reg = RandomForestRegressor(random_state = 0, n_estimators = n).fit(training_x, training_y)
		test_x = data[row_idx : row_idx + 1, 1 : -1]
		predicted_speedups[row_idx] = reg.predict(test_x)
	return predicted_speedups

def decision_tree_regressor(data, n = 100):		# with 1-hold out validation
	predicted_speedups = np.zeros((data.shape[0]))
	for row_idx in range(0, data.shape[0]):
		training_x = np.concatenate((data[0 : row_idx, 1 : -1], data[row_idx + 1 :, 1 : -1]), axis = 0)
		training_y = np.concatenate((data[0 : row_idx, -1], data[row_idx + 1 :, -1]), axis = 0)
		reg = tree.DecisionTreeRegressor().fit(training_x, training_y)
		test_x = data[row_idx : row_idx + 1, 1 : -1]
		predicted_speedups[row_idx] = reg.predict(test_x)
	return predicted_speedups

	# predicted_speedups = np.zeros((data.shape[0]))
	# for row_idx in range(0, data.shape[0]):
	# 	training_x = np.concatenate((data[0 : row_idx, 1 : -1], data[row_idx + 1 :, 1 : -1]), axis = 0)
	# 	training_y = np.concatenate((data[0 : row_idx, -1 :], data[row_idx + 1 :, -1 :]), axis = 0)
	# 	training_data = np.concatenate((training_x, training_y), axis = 1)
	# 	test_x = data[row_idx : row_idx + 1, 1 : -1]
	# 	distances = {}
	# 	for data_point in training_data:
	# 		x = data_point[: - 1]
	# 		y = data_point[-1]
	# 		distances[np.sqrt(np.mean((x - test_x) ** 2))] = y
	# 	inverse_predicted_speedup = 0.0
	# 	i = 0
	# 	sum_inverse_distances = 0.0
	# 	for key in sorted(distances.keys()):
	# 		i += 1
	# 		if i > k:
	# 			break
	# 		inverse_predicted_speedup += 1 / (key * y)
	# 		sum_inverse_distances += 1 / key
	# 	predicted_speedups[row_idx] = sum_inverse_distances / inverse_predicted_speedup
	# return predicted_speedups
	
data = np.concatenate((scaled_data('data_splash_extended.csv'), scaled_data('data_parsec_extended.csv')), axis = 0)

data_4_threads = []
data_8_threads = []
data_16_threads = []
data_32_threads = []

for i in range(0, data.shape[0]):
	if abs(data[i][0] - 4.00) < 0.1: 
		data_4_threads.append(data[i])
	elif abs(data[i][0] - 8.00) < 0.1: 
		data_8_threads.append(data[i])
	elif abs(data[i][0] - 16.00) < 0.1: 
		data_16_threads.append(data[i])
	elif abs(data[i][0] - 32.00) < 0.1: 
		data_32_threads.append(data[i])

def train_predict(thread_count, data):
	data = np.array(data)
	actual_speedups = data[:, -1]
	predicted_speedup = gaussian_proces_regressor(data)
	print('Evaluation metrics for ' + str(thread_count)  + ' threads:')
	print("Co-relation Coofficient")
	print(np.corrcoef(actual_speedups, predicted_speedup)[0, 1])
	print("Variance Score")
	print(explained_variance_score(actual_speedups, predicted_speedup, multioutput='uniform_average'))
	print("Mean squared error")
	print(mean_squared_error(actual_speedups, predicted_speedup))
	print("Mean absolute error")
	print(mean_absolute_error(actual_speedups, predicted_speedup))
	print("R squared")
	# print(r2_score(actual_speedups, predicted_speedup, multioutput='variance_weighted'))
	SS_Residual = sum((actual_speedups-predicted_speedup)**2)
	SS_Total = sum((actual_speedups-np.mean(actual_speedups))**2)
	r_squared = 1 - (float(SS_Residual))/SS_Total
	print(r_squared)
	adjusted_r_squared = 1 - (1-r_squared)*(len(actual_speedups)-1)/(len(actual_speedups)-data.shape[1]-3)
	print("Adjusted R squared")
	print(str(adjusted_r_squared) + "\n")

train_predict(4, data_4_threads)
train_predict(8, data_8_threads)
train_predict(16, data_16_threads)
train_predict(32 ,data_32_threads)