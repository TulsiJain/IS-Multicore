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

train_data_final = [] 
for i in range(0, data.shape[0]):
	if data[i][0] != 1:
		train_data_final.append(data[i])
train_data_final = np.array(train_data_final)

test_data_ratio = 0.8
testing_data_position = (int) (train_data_final.shape[0]*test_data_ratio)
actual_speedups = train_data_final[testing_data_position:, -1]


predicted_speedups = nn_regression(train_data_final, testing_data_position, "_nn_regression")
predicted_speedups = np.array(predicted_speedups)[0]

print('Speedups:')
print(actual_speedups)
print('Predicted speedups:')
print(predicted_speedups)
print('Correlation:')
print(np.corrcoef(actual_speedups, predicted_speedups)[0, 1])
