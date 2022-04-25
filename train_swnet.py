import HybridNet
import torch
import matplotlib.pyplot as plt
import scipy.io as scio
import h5py
import numpy as np
import math
import time
import os
from numpy import inf
from mytmm import tmm_initial as tmmi
from tqdm import tqdm

dtype = torch.float
device_data = torch.device("cuda")
device_train = torch.device("cuda")
device_test = torch.device("cuda")

Material = 'Thin'
# Material = 'TF'
TrainingDataSize = 300000
TestingDataSize = 10000
BatchSize = 2000
EpochNum = 100
TestInterval = 10
lr = 1e-4
lr_decay_step = 100
lr_decay_gamma = 0.8
beta_range = 1e-3
TFNum = 16
SpectralSliceNum = 89#WL.size
WL = np.array(range(SpectralSliceNum)) * 4 + 400
if Material == 'Meta':
    params_min = torch.tensor([200, 100, 50, 300])
    params_max = torch.tensor([400, 200, 200, 400])
else:
    params_min = torch.tensor([0.1])
    params_max = torch.tensor([0.3])
    theta_max = torch.tensor([math.cos(0/360*(2*math.pi))])
    theta_min = torch.tensor([math.cos(45/360*(2*math.pi))])

# StartWL = 400
# EndWL = 701
# Resolution = 2
# WL = np.arange(StartWL, EndWL, Resolution)

def load_data(size):
    import scipy.io as sc
    train_path = "./balloons_ms/train"
    test_path = "./balloons_ms/test"
    train_name = os.listdir(train_path)
    test_name = os.listdir(test_path)
    train = np.zeros([157286*size,89])
    test = np.zeros([78644*size,89])
    for i in range(size):
        data = sc.loadmat(train_path+'/'+train_name[i])
        train[i*157286:(i+1)*157286,:] = data['train']
        # train = (train - np.min(train))/ (np.max(train)-np.min(train))
        data = sc.loadmat(test_path+'/'+test_name[i])
        test[i*78644:(i+1)*78644,:] = data['test']
    # test = (test - np.min(test))/ (np.max(test)-np.min(test))
    return [train,test]

[train,test]=load_data(2)
np.random.seed(116)
np.random.shuffle(train)
np.random.seed(116)
np.random.shuffle(test)
Specs_train = torch.tensor(train, device=device_data, dtype=dtype)
Specs_test  = torch.tensor(test, device=device_data, dtype=dtype)
path = path = 'torchnets/hybnet/20220324_094851/'
para = scio.loadmat(path + 'TrainedParams.mat')
# para = para['Params']
thick = para['thick']
theta = para['theta']
para = np.matmul(theta.T, thick)
d_list = [inf, 100, 200, 100, 200, 100, 200, 200, 200, 200, 200, inf]
ran = np.array(range(TFNum))
T_array = np.ones([TFNum, SpectralSliceNum])
# T = sc.loadmat(path+'TargetCurves')['TargetCurves']
for i in tqdm(ran):
    lambda_lis = np.array(range(100)) * 4 + 400
    inpu = para[i, :] * 1000
    inpu = inpu.reshape(10, ).tolist()
    d_list[1:11] = inpu
    [lambda_list, T_list, _] = tmmi.sample2(d_list, 89)
    # plt.plot(lambda_list,T_list)
    # plt.plot(lambda_list,T[i,:])
    # plt.show()
    T_array[i, :] = np.array(T_list).reshape([1, 89])

weights = torch.tensor(T_array, device=device_data, dtype=dtype)

thick = torch.tensor(thick,device=device_data,dtype=dtype)
theta = torch.tensor(theta,device=device_data,dtype=dtype)



# path_data = './data/'
# Specs_train = torch.zeros([TrainingDataSize, SpectralSliceNum], device=device_data, dtype=dtype)
# Specs_test = torch.zeros([TestingDataSize, SpectralSliceNum], device=device_test, dtype=dtype)
# data = h5py.File(path_data + 'ICVL/SpectralCurves/ICVLSpecs_PchipInterp.mat', 'r')
# Specs_all = np.array(data['Specs_interp']).T
# np.random.shuffle(Specs_all)
# Specs_train[0:TrainingDataSize//2, :] = torch.tensor(Specs_all[0:TrainingDataSize//2, :])
# Specs_test[0:TestingDataSize//2, :] = torch.tensor(
#     Specs_all[TrainingDataSize//2:TrainingDataSize//2 + TestingDataSize//2, :])
# data = h5py.File(path_data + 'CAVE/SpectralCurves/ColumbiaSpecs_PchipInterp.mat', 'r')
# Specs_all = np.array(data['Specs'])
# np.random.shuffle(Specs_all)
# Specs_train[TrainingDataSize//2:TrainingDataSize, :] = torch.tensor(Specs_all[0:TrainingDataSize//2, :])
# Specs_test[TestingDataSize//2:TestingDataSize, :] = torch.tensor(
#     Specs_all[TrainingDataSize//2:TrainingDataSize//2 + TestingDataSize//2, :])

# data.close()
del train,test
assert SpectralSliceNum == Specs_train.size(1)

folder_name = time.strftime("swnet"+"%Y%m%d_%H%M%S", time.localtime())
path = './torchnets/hybnet/' + folder_name + '/'
if Material == 'Meta':
    fnet_path = './torchnets/fnet/Meta/fnet.pkl'
else:
    fnet_path = './torchnets/fnet/TF_100-300nm/fnet.pkl'
    # fnet_path = 'nets/fnet/TF_0-150nm/fnet.pkl'

hybnet_size = [SpectralSliceNum, TFNum, 500, 500, SpectralSliceNum]
hybnet = HybridNet.HybridNet(fnet_path, params_min, params_max,theta_min,theta_max ,hybnet_size, device_train)
hybnet.load_para(thick,theta)
LossFcn = HybridNet.HybnetLoss()

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, hybnet.parameters()), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)

loss = torch.tensor([0], device=device_train)
loss_train = torch.zeros(math.ceil(EpochNum / TestInterval))
loss_test = torch.zeros(math.ceil(EpochNum / TestInterval))

os.makedirs(path, exist_ok=True)
log_file = open(path + 'TrainingLog.txt', 'w+')
time_start = time.time()
time_epoch0 = time_start
for epoch in range(EpochNum):
    Specs_train = Specs_train[torch.randperm(TrainingDataSize), :]
    for i in range(0, TrainingDataSize // BatchSize):
        Specs_batch = Specs_train[i * BatchSize: i * BatchSize + BatchSize, :].to(device_train)
        Output_pred = hybnet(Specs_batch,weights)
        DesignParams = hybnet.show_design_params()
        loss = LossFcn(Specs_batch, Output_pred, DesignParams,
                       params_min.to(device_train), params_max.to(device_train),
                       theta_min.to(device_train),theta_max.to(device_train) ,
                       beta_range)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
    scheduler.step()
    if epoch % TestInterval == 0:
        hybnet.to(device_test)
        hybnet.eval()
        Out_test_pred = hybnet(Specs_test,weights)
        hybnet.to(device_train)
        hybnet.train()
        hybnet.eval_fnet()
        loss_train[epoch // TestInterval] = loss.data
        loss_t = HybridNet.MatchLossFcn(Specs_test, Out_test_pred)
        loss_test[epoch // TestInterval] = loss_t.data
        if epoch == 0:
            time_epoch0 = time.time()
            time_remain = (time_epoch0 - time_start) * EpochNum
        else:
            time_remain = (time.time() - time_epoch0) / epoch * (EpochNum - epoch)
        print('Epoch: ', epoch, '| train loss: %.5f' % loss.item(), '| test loss: %.5f' % loss_t.item(),
              '| learn rate: %.8f' % scheduler.get_lr()[0], '| remaining time: %.0fs (to %s)'
              % (time_remain, time.strftime('%H:%M:%S', time.localtime(time.time() + time_remain))))
        print('Epoch: ', epoch, '| train loss: %.5f' % loss.item(), '| test loss: %.5f' % loss_t.item(),
              '| learn rate: %.8f' % scheduler.get_lr()[0], file=log_file)
time_end = time.time()
time_total = time_end - time_start
m, s = divmod(time_total, 60)
h, m = divmod(m, 60)
print('Training time: %.0fs (%dh%02dm%02ds)' % (time_total, h, m, s))
print('Training time: %.0fs (%dh%02dm%02ds)' % (time_total, h, m, s), file=log_file)

hybnet.eval()
hybnet.eval_fnet()
torch.save(hybnet, path + 'hybnet.pkl')
hybnet.to(device_test)

HWweights = hybnet.show_hw_weights()
TargetCurves = HWweights.double().detach().cpu().numpy()[:,:89]
scio.savemat(path + 'TargetCurves.mat', mdict={'TargetCurves': TargetCurves})

DesignParams = hybnet.show_design_params()
thick = DesignParams[0].double().detach().cpu().numpy()
theta = DesignParams[1].double().detach().cpu().numpy()
print(DesignParams)
TargetCurves_FMN = hybnet.run_fnet(DesignParams).double().detach().cpu().numpy()[:,:89]
scio.savemat(path + 'TargetCurves_FMN.mat', mdict={'TargetCurves_FMN': TargetCurves_FMN})
# Params = DesignParams.double().detach().cpu().numpy()
scio.savemat(path + 'TrainedParams.mat', mdict={'thick': thick,'theta':theta})

plt.figure()
for i in range(TFNum):
    plt.subplot(math.ceil(math.sqrt(TFNum)), math.ceil(math.sqrt(TFNum)), i + 1)
    plt.plot(WL, TargetCurves[i, :], WL, TargetCurves_FMN[i, :])
    plt.ylim(0, 1)
plt.savefig(path + 'ROFcurves')
plt.show()

Output_train = hybnet(Specs_train[100, :].to(device_test).unsqueeze(0),weights).squeeze(0)
FigureTrainLoss = HybridNet.MatchLossFcn(Specs_train[100, :].to(device_test), Output_train)
plt.figure()
plt.plot(WL, Specs_train[100, :].cpu().numpy())
plt.plot(WL, Output_train.detach().cpu().numpy())
plt.ylim(0, 1)
plt.legend(['GT', 'pred'], loc='upper right')
plt.savefig(path + 'train')
plt.show()

Output_test = hybnet(Specs_test[100, :].to(device_test).unsqueeze(0),weights).squeeze(0)
FigureTestLoss = HybridNet.MatchLossFcn(Specs_test[100, :].to(device_test), Output_test)
plt.figure()
plt.plot(WL, Specs_test[100, :].cpu().numpy())
plt.plot(WL, Output_test.detach().cpu().numpy())
plt.ylim(0, 1)
plt.legend(['GT', 'pred'], loc='upper right')
plt.savefig(path + 'test')
plt.show()

print('Training finished!',
      '| loss in figure \'train.png\': %.5f' % FigureTrainLoss.data.item(),
      '| loss in figure \'test.png\': %.5f' % FigureTestLoss.data.item())
print('Training finished!',
      '| loss in figure \'train.png\': %.5f' % FigureTrainLoss.data.item(),
      '| loss in figure \'test.png\': %.5f' % FigureTestLoss.data.item(), file=log_file)
log_file.close()

plt.figure()
plt.plot(range(0, EpochNum, TestInterval), loss_train.detach().cpu().numpy())
plt.plot(range(0, EpochNum, TestInterval), loss_test.detach().cpu().numpy())
plt.semilogy()
plt.legend(['Loss_train', 'Loss_test'], loc='upper right')
plt.savefig(path + 'loss')
plt.show()