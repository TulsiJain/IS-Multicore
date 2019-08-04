import numpy as np
from sklearn.linear_model import LinearRegression
from numpy import genfromtxt
import torch
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path

def scaled_data(file_name):
	data = genfromtxt(file_name, delimiter=',')
	scalar = StandardScaler()
	data_normalized = scalar.fit_transform(data[:, 1 : -1])
	return np.concatenate((data[:, 0 : 1], data_normalized, data[:, -1 :]), axis = 1)

class Regression(nn.Module):
	def __init__(self):
		super().__init__()
		self.l1 = nn.Linear(15, 3)
		self.dropout = nn.Dropout(0.1)
		self.l2 = nn.Linear(3, 1)

	def forward(self, x):
		x = F.relu(self.l1(x))
		x = self.dropout(x)
		x = self.l2(x)        
		return x

def nn_regression(data, testing_starting_point, threads):
	predicted_speedups = []
	regression = Regression()
	regression_optim = Adam(regression.parameters(), lr=1e-4)
	training_x = data[0 : testing_starting_point, 1 : -1]
	training_y = data[0 : testing_starting_point, -1:]
	tensor_x = torch.stack([torch.Tensor(i) for i in training_x]) 
	tensor_y = torch.stack([torch.Tensor(i) for i in training_y])
	my_dataset = utils.TensorDataset(tensor_x, tensor_y) # create your datset
	my_test_dataloader = utils.DataLoader(my_dataset, batch_size=64) # create your dataloader

	test_x = data[testing_starting_point :, 1 : -1]
	test_y = data[testing_starting_point :, -1:]
	tensor_test_x = torch.stack([torch.Tensor(i) for i in test_x])
	tensor_test_y = torch.stack([torch.Tensor(i) for i in test_y]) 
	my_dataset_test = utils.TensorDataset(tensor_test_x, tensor_test_y) # create your datset
	my_dataloader_test = utils.DataLoader(my_dataset_test, batch_size=64) # create your dataloader

	epoch_restart = 0
	root_regression_model = Path(r'models' + str(threads))

	if epoch_restart > 0 and root is not None:
		regression_model_file = root_regression_model / Path('regression_loss' + str(epoch_restart) + '.wgt')
		regression.load_state_dict(torch.load(str(regression_model_file)))
	
	batch_size = 100
	criterion = nn.MSELoss(reduction='sum')
	epoch_restart = 0

	for epoch in range(epoch_restart + 1, 201):
		for batch_idx, (feature, target) in enumerate(my_test_dataloader):
			predicted = regression(feature)
			regression_optim.zero_grad()
			loss = criterion(predicted, target)
			loss.backward()
			regression_optim.step()

		if epoch % 100 == 0:
			regression_loss_file = root_regression_model / Path('regression_loss' + str(epoch) + '.wgt')
			regression_loss_file.parent.mkdir(parents=True, exist_ok=True)
			torch.save(regression.state_dict(), str(regression_loss_file))
	
	regression = Regression()
	regression_model_file = root_regression_model / Path('regression_loss200.wgt')
	regression.load_state_dict(torch.load(str(regression_model_file)))
	for batch_idx, (feature, target) in enumerate(my_dataloader_test):
		a = regression(feature)
		a = torch.squeeze(a).data.numpy()
		predicted_speedups.append(a)
	return predicted_speedups
		 
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


data_4_threads = np.array(data_4_threads)
data_8_threads = np.array(data_8_threads)
data_16_threads = np.array(data_16_threads)
data_32_threads = np.array(data_32_threads)

test_data_ratio = 0.8
testing_position_4 = (int) (data_4_threads.shape[0]*test_data_ratio)
actual_speedups_4_threads = data_4_threads[testing_position_4:, -1]
testing_position_8 = (int) (data_8_threads.shape[0]*test_data_ratio)
actual_speedups_8_threads = data_8_threads[testing_position_8:, -1]
testing_position_16 = (int) (data_16_threads.shape[0]*test_data_ratio)
actual_speedups_16_threads = data_16_threads[testing_position_16:, -1]
testing_position_32 = (int) (data_32_threads.shape[0]*test_data_ratio)
actual_speedups_32_threads = data_32_threads[testing_position_32:, -1]

predicted_speedups_4_threads = nn_regression(data_4_threads, testing_position_4, 4)
predicted_speedups_8_threads = nn_regression(data_8_threads, testing_position_8, 8)
predicted_speedups_16_threads = nn_regression(data_16_threads, testing_position_16, 16)
predicted_speedups_32_threads = nn_regression(data_32_threads, testing_position_32, 32)

print('Speedups with 4 threads:')
print(actual_speedups_4_threads)
print('Predicted speedups with 4 threads:')
print(predicted_speedups_4_threads)
print('Correlation:')
print(np.corrcoef(actual_speedups_4_threads, predicted_speedups_4_threads)[0, 1])

print('Speedups with 8 threads:')
print(actual_speedups_8_threads)
print('Predicted speedups with 8 threads:')
print(predicted_speedups_8_threads)
print('Correlation:')
print(np.corrcoef(actual_speedups_8_threads, predicted_speedups_8_threads)[0, 1])

print('Speedups with 16 threads:')
print(actual_speedups_16_threads)
print('Predicted speedups with 16 threads:')
print(predicted_speedups_16_threads)
print('Correlation:')
print(np.corrcoef(actual_speedups_16_threads, predicted_speedups_16_threads)[0, 1])

print('Speedups with 32 threads:')
print(actual_speedups_32_threads)
print('Predicted speedups with 32 threads:')
print(predicted_speedups_32_threads)
print('Correlation:')
print(np.corrcoef(actual_speedups_32_threads, predicted_speedups_32_threads)[0, 1])