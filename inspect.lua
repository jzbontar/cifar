require 'cutorch'
require 'cunn'
require 'image'
require 'jzt'

function zoom(img, scale)
   assert(img:nDimension() == 3)
   h = img:size(2)
   w = img:size(3)
   img_big = torch.Tensor(3, h * scale, w * scale)

   for k = 1,3 do
      for i = 1,img_big:size(2) do
         for j = 1,img_big:size(3) do
            img_big[{k,i,j}] = img[{k,math.ceil(i/scale),math.ceil(j/scale)}]
         end
      end
   end
   return img_big
end

net = torch.load('net/00100')
weight = net:get(2).weight
dd = image.toDisplayTensor{input=weight, padding=1, nrow=8, symmetric=true}
ddd = zoom(dd, 8)
image.savePNG('foo.png', ddd)
