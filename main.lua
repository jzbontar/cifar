require 'image'
require 'torch'
require 'cunn'
require 'cutorch'
require 'nn'
require 'prof-torch'

cutorch.setDevice(arg[1])

torch.manualSeed(42)

batch_size = 128
momentum = 0.9
weight_decay = 0
learning_rate = 0.001
learning_rate_decay = 1

data_path = '/misc/vlgscratch2/LecunGroup/goroshin/Data/data_gcn_whitened/'
X = torch.FloatTensor(torch.FloatStorage(data_path .. 'X')):resize(60000, 3, 32, 32)
y = torch.FloatTensor(torch.FloatStorage(data_path .. 'y'))

X = X:cuda()
y = y:cuda()

pool = nn.SpatialLPPooling(64, 2, 2, 2, 2, 2)
pool:get(3).eps = 1e-6

net = nn.Sequential{bprop_min=1,debug=0}
net:add(nn.SpatialZeroPadding(2, 2, 2, 2))
net:add(nn.SpatialConvolutionRing2(3, 64, 5, 5))
net:add(nn.Threshold())
--net:add(nn.SpatialMaxPooling(2, 2, 2, 2))
net:add(pool:clone())

net:add(nn.SpatialZeroPadding(2, 2, 2, 2))
net:add(nn.SpatialConvolutionRing2(64, 64, 5, 5))
net:add(nn.Threshold())
--net:add(nn.SpatialMaxPooling(2, 2, 2, 2))
net:add(pool:clone())

net:add(nn.SpatialZeroPadding(2, 2, 2, 2))
net:add(nn.SpatialConvolutionRing2(64, 64, 5, 5))
net:add(nn.Threshold())
--net:add(nn.SpatialMaxPooling(2, 2, 2, 2))
net:add(pool:clone())

-- print(net:cuda():forward(X:narrow(1, 1, 128)):size())

size = 64 * 4 * 4
net:add(nn.Reshape(size))
net:add(nn.Linear(size, 10))
net:add(nn.LogSoftMax())

measure = nn.ClassNLLCriterion()

net = net:cuda()
measure = measure:cuda()

parameters, grad_parameters = net:getParameters()
grad_momentum = torch.CudaTensor(grad_parameters:nElement()):zero()

start = prof.time()
for epoch = 1,100 do
   for t = 1,50000 - batch_size,batch_size do
      X_batch = X:narrow(1, t, batch_size)
      y_batch = y:narrow(1, t, batch_size)
      net:forward(X_batch)
      measure:forward(net.output, y_batch)
      measure:backward(net.output, y_batch)
      net:zeroGradParameters()
      net:backward(X_batch, measure.gradInput)

      grad_momentum:mul(momentum)
      grad_momentum:add(-weight_decay * learning_rate, parameters)
      grad_momentum:add(-learning_rate, grad_parameters)

      parameters:add(grad_momentum)
      learning_rate = learning_rate * learning_rate_decay
   end

   if epoch % 1 == 0 then
      -- torch.save(('net/%05d.bin'):format(epoch), net)
      err = 0
      for t = 50001,60000 - batch_size,batch_size do
         X_batch = X:narrow(1, t, batch_size)
         y_batch = y:narrow(1, t, batch_size)
         net:forward(X_batch)
         _, i = net.output:float():max(2)
         err = err + i:float():ne(y_batch:float()):sum()
      end
      print(epoch, err / 10000, prof.time() - start)
   end
end
