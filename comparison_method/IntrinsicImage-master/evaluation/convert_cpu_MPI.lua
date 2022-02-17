require 'torch'
require 'nn'
require 'image'
require 'sys'
require 'cunn'
require 'cutorch'
require 'cudnn'
require 'nngraph'

local model = torch.load('../netfiles/model_MPI_main_SceneSplit_front_combine_best.net')
cudnn.convert(model, nn)
nn.utils.recursiveType(model, 'torch.CudaTensor')
model = model:float()
print(model)
torch.save("../netfiles/model_MPI_main_SceneSplit_front_combine_best_cpu.net", model, 'ascii')
--model = model:cuda()
--model = model:float()
--model:training()
--print('model structure is:')
--print(model)

