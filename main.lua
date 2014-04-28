require 'image'
require 'torch'
require 'cunn'
require 'cutorch'
require 'nn'
require 'prof-torch'

torch.manualSeed(42)

batch_size = 128
momentum = 0.9
weight_decay = 0 --0.0000001
learning_rate = 0.0005
learning_rate_decay = 1

X = torch.FloatTensor(torch.FloatStorage('data_gcn_whitened/X')):resize(60000, 3, 32, 32)
y = torch.FloatTensor(torch.FloatStorage('data_gcn_whitened/y'))

-- X = torch.FloatTensor(60000, 3, 32, 32)
-- y = torch.FloatTensor(60000)
-- tr = torch.load('data_ross/CIFAR_CN_train.t7')
-- te = torch.load('data_ross/CIFAR_CN_test.t7')
-- X:narrow(1, 1, 50000):copy(tr['datacn'])
-- y:narrow(1, 1, 50000):copy(tr['labels'])
-- X:narrow(1, 50001, 10000):copy(te['datacn'])
-- y:narrow(1, 50001, 10000):copy(te['labels'])

X = X:cuda()
y = y:cuda()

net = nn.Sequential{bprop_min=1,debug=0}
net:add(nn.SpatialZeroPadding(2, 2, 2, 2))
net:add(nn.SpatialConvolutionRing(3, 64, 5, 5))
net:add(nn.Threshold())
net:add(nn.SpatialLPPooling(64, 2, 2, 2, 2, 2))

net:add(nn.SpatialZeroPadding(2, 2, 2, 2))
net:add(nn.SpatialConvolutionBatch(64, 64, 5, 5))
net:add(nn.Threshold())
net:add(nn.SpatialLPPooling(64, 2, 2, 2, 2, 2))

net:add(nn.SpatialZeroPadding(2, 2, 2, 2))
net:add(nn.SpatialConvolutionBatch(64, 64, 5, 5))
net:add(nn.Threshold())
net:add(nn.SpatialLPPooling(64, 2, 2, 2, 2, 2))

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

      print(t, net:get(2).weight:mean())
      if t > 1 then
         os.exit()
      end

      grad_parameters:zero()
      X_batch = X:narrow(1, t, batch_size)
      y_batch = y:narrow(1, t, batch_size)
      net:forward(X_batch)
      measure:forward(net.output, y_batch)
      measure:backward(net.output, y_batch)
      net:backward(X_batch, measure.gradInput)
      print(net:get(10).gradInput:mean())

      grad_momentum:mul(momentum):add(-weight_decay * learning_rate, parameters):add(-learning_rate, grad_parameters)
      parameters:add(grad_momentum)
      learning_rate = learning_rate * learning_rate_decay

      collectgarbage()
      cutorch.synchronize()
   end

   if epoch % 1 == 0 then
      -- torch.save(('net/%05d'):format(epoch), net)

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

net:print_timing()
