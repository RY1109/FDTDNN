import HybridNet
import torch
import scipy.io as scio
import os
import numpy as np
from numpy import inf
from mytmm import tmm_initial as tmmi
from tqdm import tqdm
from matplotlib import pyplot as plt

dtype = torch.float
device = torch.device("cpu")

TrainingDataSize = 1000000
TestingDataSize = 100000
SpectralSliceNum = 89#WL.size
WL = np.array(range(SpectralSliceNum)) * 4 + 400
numTF = 4
# StartWL = 400
# EndWL = 701
# Resolution = 2
# WL = np.arange(StartWL, EndWL, Resolution)
# SpectralSliceNum = WL.size
def load_data(size):
    import scipy.io as sc
    val_path = "./balloons_ms/val"
    val_name = os.listdir(val_path)
    val = np.zeros([26215*size,89])
    for i in range(size):
        data = sc.loadmat(val_path+'/'+val_name[i])
        val[i*26215:(i+1)*26215,:] = data['val']
    # test = (test - np.min(test))/ (np.max(test)-np.min(test))
    return val
path = 'torchnets/hybnet/TF/'
size = 2
para = scio.loadmat(path+'TrainedParams.mat')
para = para['Params']
d_list = [inf, 100,200,100,200,100,200, 200,200,200,200,inf]
ran = np.array(range(numTF))
T_array = np.ones([numTF,SpectralSliceNum])
val=load_data(size)

for i in tqdm(ran):
    lambda_lis  = np.array(range(100))*4+400
    inpu = para[i,:]*1000
    inpu = inpu.reshape(10,).tolist()
    d_list[1:11] = inpu
    [lambda_list,T_list] = tmmi.sample2(d_list,89)
    # plt.plot(lambda_list,T_list)
    # plt.show()
    T_array[i,:] = np.array(T_list).reshape([1,89])

specs_train  = torch.tensor(val, device=device, dtype=dtype)

del val
assert SpectralSliceNum == WL.size

cmp = ['designed', 'noised']


hybnet = torch.load(path + 'hybnet.pkl')
hybnet.to(device)
hybnet.eval()


HWWeights_designed = torch.tensor(T_array, device=device, dtype=dtype)
# HWWeights_noised = torch.tensor(scio.loadmat(path + 'TrainedCurves_check_sim_noised_6nm.mat')['TFs_noised'], device=device, dtype=dtype)
TFNum = HWWeights_designed.size(0)
TFCurves_designed = HWWeights_designed.cpu().numpy()

output_train = hybnet(specs_train.to(device))
loss_train = HybridNet.MatchLossFcn(specs_train.cpu(), output_train.cpu())
output_train_designed = hybnet.run_swnet(specs_train.to(device), HWWeights_designed)
loss_train_designed = HybridNet.MatchLossFcn(specs_train.cpu(), output_train_designed.cpu())
a = np.array(range(10))
for i in tqdm(a):
    plt.plot(WL,specs_train.detach().numpy()[i,:])
    plt.plot(WL,output_train.detach().numpy()[i,:])
    plt.show()
# output_test = hybnet(specs_test.to(device))
# loss_test = HybridNet.MatchLossFcn(specs_test.cpu(), output_test.cpu())
# output_test_designed = hybnet.run_swnet(specs_test.to(device), HWWeights_designed)
# loss_test_designed = HybridNet.MatchLossFcn(specs_test.cpu(), output_test_designed.cpu())

# output_train_noised = hybnet.run_swnet(specs_train.to(device), HWWeights_noised)
# loss_train_noised = HybridNet.MatchLossFcn(specs_train.cpu(), output_train_noised.cpu())
# output_test_noised = hybnet.run_swnet(specs_test.to(device), HWWeights_noised)
# loss_test_noised = HybridNet.MatchLossFcn(specs_test.cpu(), output_test_noised.cpu())

log_file = open(path + 'RunningLog.txt', 'w+')
print('Running finished!')
print('| train loss using trained curves: %.5f' % loss_train.data.item(),
      '| train loss using designed curves: %.5f' % loss_train_designed.data.item())
      # '| train loss using noised curves: %.5f' % loss_train_noised.data.item())
# print('| test loss using trained curves: %.5f' % loss_test.data.item(),
      # '| test loss using designed curves: %.5f' % loss_test_designed.data.item(),
      # '| test loss using noised curves: %.5f' % loss_test_noised.data.item())
print('Running finished!', file=log_file)
print('| train loss using trained curves: %.5f' % loss_train.data.item(),
      '| train loss using designed curves: %.5f' % loss_train_designed.data.item(),file=log_file)
      # '| train loss using noised curves: %.5f' % loss_train_noised.data.item(), file=log_file)
# print('| test loss using trained curves: %.5f' % loss_test.data.item(),
#       '| test loss using designed curves: %.5f' % loss_test_designed.data.item(),
#       '| test loss using noised curves: %.5f' % loss_test_noised.data.item(), file=log_file)
log_file.close()
