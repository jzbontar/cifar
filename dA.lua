require 'image'
require 'torch'
require 'cunn'
require 'cutorch'
require 'nn'
require 'jzt'
require 'prof-torch'

cutorch.setDevice(arg[1])

data_path = 'data_gcn_whitened/'
X = torch.FloatTensor(torch.FloatStorage(data_path .. 'X')):resize(60000, 3, 32, 32):cuda()
y = torch.FloatTensor(torch.FloatStorage(data_path .. 'y')):cuda()

enc = nn.Sequential()
enc:add(nn.SpatialConvolutionRing2(3, 64, 5, 5))
enc:add(nn.Threshold())
enc:cuda()

dec = nn.Sequential()
dec:add(nn.SpatialZeroPadding(4, 4, 4, 4))
dec:add(nn.SpatialConvolutionRing2(64, 3, 5, 5))
dec:cuda()
