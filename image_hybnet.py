import HybridNet
import torch
import scipy.io as scio
import os
import numpy as np
from numpy import inf
from mytmm import tmm_initial as tmmi
from tqdm import tqdm
from matplotlib import pyplot as plt
import scipy.io as sc

dtype = torch.float
device = torch.device("cpu")

TrainingDataSize = 1000000
TestingDataSize = 100000
SpectralSliceNum = 89#WL.size
WL = np.array(range(SpectralSliceNum)) * 4 + 400
numTF = 16
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 12,
}
# StartWL = 400
# EndWL = 701
# Resolution = 2
# WL = np.arange(StartWL, EndWL, Resolution)
# SpectralSliceNum = WL.size
def load_data(size):
    import scipy.io as sc
    val_path = "./balloons_ms/image"
    val_name = os.listdir(val_path)
    for i in range(size):
        data = sc.loadmat(val_path+'/'+val_name[i+16])
        print(val_name[i+16])
        val= data['hyper']
    # test = (test - np.min(test))/ (np.max(test)-np.min(test))
    return val
path = 'torchnets/hybnet/TF3/'
size = 1
para = scio.loadmat(path+'TrainedParams.mat')
# para = para['Params']
thick = para['thick']
theta = para['theta']
para = np.matmul(theta.T,thick)
d_list = [inf, 100,200,100,200,100,200,200,200,200,200,inf]
ran = np.array(range(numTF))
T_array = np.ones([numTF,SpectralSliceNum])
val=load_data(size)

for i in tqdm(ran):
    lambda_lis  = np.array(range(100))*4+400
    inpu = para[i,:]*1000
    inpu = inpu.reshape(10,).tolist()
    d_list[1:11] = inpu
    [lambda_list,T_list,_] = tmmi.sample2(d_list,89)
    # plt.plot(lambda_list,T_list)
    # plt.show()
    T_array[i,:] = np.array(T_list).reshape([1,89])

specs_train  = torch.tensor(val, device=device, dtype=dtype)

assert SpectralSliceNum == WL.size

cmp = ['designed', 'noised']


hybnet = torch.load(path + 'hybnet.pkl')
hybnet.to(device)
hybnet.eval()


HWWeights_designed = torch.tensor(T_array, device=device, dtype=dtype)
# HWWeights_noised = torch.tensor(scio.loadmat(path + 'TrainedCurves_check_sim_noised_6nm.mat')['TFs_noised'], device=device, dtype=dtype)
TFNum = HWWeights_designed.size(0)
TFCurves_designed = HWWeights_designed.cpu().numpy()

output_train = hybnet(specs_train.to(device),[])
loss_train = HybridNet.MatchLossFcn(specs_train.cpu(), output_train.cpu())
output_train_designed = hybnet.run_swnet(specs_train.to(device), HWWeights_designed)
loss_train_designed = HybridNet.MatchLossFcn(specs_train.cpu(), output_train_designed.cpu())
image = output_train_designed.detach().numpy().reshape([512,512,89])
val = val.reshape([512,512,89])
for pos in range(6):
    curve = plt.figure()##重建图像
    curve_ = curve.add_subplot(1,1,1)
    curve_.plot(image[int(50+pos*500/6),250,:],label='pre')
    curve_.plot(val[int(50+pos*500/6),250,:],label='true')
    mse = np.sum(np.square(image[int(50+pos*500/6),250,:]-val[int(50+pos*500/6),250,:]))/89
    curve_.set(title=str(mse))
    curve_.legend(prop=font1)
    curve.savefig('./result/figure/feathers_image2/curve_'+str(pos))
    sc.savemat('./result/data/feathers_image2/curve_'+str(pos)+'.mat', {'pre': image[int(50+pos*500/6),250,:],
                                                                       'true': val[int(50+pos*500/6),250,:],
                                                                       'mse':mse})
for lam in range(9):
    fig = plt.figure()##重建图像
    fig_ = fig.add_subplot(1,1,1)
    img = np.sum(image[:,:,int(10*lam):int(10*lam+10)],2)
    img = np.flip(img,0)
    fig_.imshow(img,cmap=plt.cm.gray)
    # plt.show()
    fig.savefig('./result/figure/feathers_image2/'+str(lam))
    sc.savemat('./result/data/feathers_image2/'+str(lam)+'.mat', {'image': image})



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
