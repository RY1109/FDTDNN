import torch
import torch.nn as nn
import torch.nn.functional as func
dtype = torch.float

class HybridNet(nn.Module):
    def __init__(self, fnet_path, thick_min, thick_max, theta_min, theta_max,size, device):
        super(HybridNet, self).__init__()
        self.fnet = torch.load(fnet_path,map_location='cpu')
        self.fnet.to(device)
        self.fnet.eval()
        for p in self.fnet.parameters():
            p.requires_grad = False
        self.tf_layer_num = self.fnet.state_dict()['0.weight'].data.size(1)
        self.thick = nn.Parameter(
            (thick_max - thick_min) * torch.rand(1,self.tf_layer_num) + thick_min.float(), requires_grad=True)
        self.theta = nn.Parameter(
            (theta_max - theta_min) * torch.rand(1,size[1]) + theta_min.float(), requires_grad=True)
        self.DesignParams = nn.Parameter(torch.mul(self.theta.T,self.thick),requires_grad=False)
        self.SWNet = nn.Sequential()
        # self.SWNet.add_module('BatchNorm0', nn.BatchNorm1d(size[1]))
        self.SWNet.add_module('LReLU0', nn.LeakyReLU(inplace=True))
        for i in range(1, len(size) - 1):
            self.SWNet.add_module('Linear' + str(i), nn.Linear(size[i], size[i + 1]))
            # self.SWNet.add_module('BatchNorm' + str(i), nn.BatchNorm1d(size[i+1]))
            # self.SWNet.add_module('DropOut' + str(i), nn.Dropout(p=0.2))
            self.SWNet.add_module('LReLU' + str(i), nn.LeakyReLU(inplace=True))
        self.to(device)

    def forward(self, data_input,weights):
        # a=self.fnet(self.DesignParams)
        # a=a[:,:89]
        # self.thick = self.thick + torch.rand(self.thick.size()) * 0.05
        # self.theta = self.theta + torch.rand(self.theta.size(),)*0.05
        self.DesignParams = nn.Parameter(torch.mul(self.theta.T, self.thick), requires_grad=False)
        if weights==[]:
            result = self.SWNet(func.linear(data_input, self.fnet(self.DesignParams)[:,:89], None))
        else:
            result = self.SWNet(func.linear(data_input, weights, None))
        return result

    def show_design_params(self):
        return [self.thick,self.theta]

    def show_hw_weights(self):
        return self.fnet(self.DesignParams)

    def eval_fnet(self):
        self.fnet.eval()
        return 0

    def run_fnet(self, design_params_input):
        design_params=torch.mul(design_params_input[1].T,design_params_input[0])
        return self.fnet(design_params)

    def run_swnet(self, data_input, hw_weights_input):
        # assert hw_weights_input.size(0) == self.DesignParams.size(0)
        return self.SWNet(func.linear(data_input, hw_weights_input, None))

    def load_para(self,thick,theta):
        self.thick = nn.Parameter(thick, requires_grad=False)
        self.theta = nn.Parameter(theta, requires_grad=False)
        return 0

    def randpara(self):
        self.theta.data = self.theta + torch.rand(self.theta.size(), device=torch.device('cuda')) * 0.05
        self.thick.data = self.thick + torch.rand(self.thick.size(), device=torch.device('cuda')) * 0.05
        return 0

MatchLossFcn = nn.MSELoss(reduction='mean')


class HybnetLoss(nn.Module):
    def __init__(self):
        super(HybnetLoss, self).__init__()

    def forward(self, t1, t2, params, thick_min, thick_max,  theta_min, theta_max,beta_range):
        # MSE loss
        match_loss = MatchLossFcn(t1, t2)

        # Structure parameter range regularization.
        # U-shaped function，U([param_min + delta, param_max - delta]) = 0, U(param_min) = U(param_max) = 1。
        delta = 0.01
        thick_loss = torch.max((params[0] - thick_min - delta) / (-delta), (params[0] - thick_max + delta) / delta)
        range_loss_thick = torch.mean(torch.max(thick_loss, torch.zeros_like(thick_loss)))
        theta_loss = torch.max((params[1] - theta_min - delta) / (-delta), (params[1] - theta_max + delta) / delta)
        range_loss_theta = torch.mean(torch.max(theta_loss, torch.zeros_like(theta_loss)))

        return match_loss + beta_range * (range_loss_theta+range_loss_thick)

if __name__ == '__main__':
    folder_name = 'test'
    path = './torchnets/hybnet/' + folder_name + '/'
    fnet_path = './torchnets/fnet/TF_100-300nm/fnet.pkl'
    params_min = torch.tensor([0.1])
    params_max = torch.tensor([0.3])
    theta_max = torch.tensor([0.8])
    theta_min = torch.tensor([0.7])
    device_data = torch.device("cuda")
    TFNum = 16
    SpectralSliceNum = 89  # WL.size
    hybnet_size = [SpectralSliceNum, TFNum, 500, 500, SpectralSliceNum]
    net = HybridNet(fnet_path,params_min,params_max,theta_min,theta_max, hybnet_size,device_data)
    loss = HybnetLoss()