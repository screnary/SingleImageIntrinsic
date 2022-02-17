require 'torch'
require 'image'
require 'sys'
require 'cunn'
require 'cutorch'
require 'cudnn'
require 'nngraph'

--imgPath = '../../datasets/MPI/MPI-main-input-300/'
imgPath = '../../datasets/MPI/MPI-main-clean/'
-- saveDir = '../results/MPI_main_SceneSplit_front_combine/'
saveDir = '../results/test/'

local n_W = 768 --512
local n_H = 328 --218
--local file_name = '../../datasets/MPI/MPI_main_sceneSplit-300-test.txt'
local file_name = '../../datasets/MPI/MPI_main_sceneSplit-fullsize-NoDefect-test.txt'
local files = {}
local f = io.open(file_name, "rb")
while true do
  local line = f:read()
  if line == nil then break end
  table.insert(files, imgPath .. line)
end
f:close()
local testsetSize = #files

model = torch.load('../netfiles/model_MPI_main_SceneSplit_front_combine_best.net')
model = model:cuda()
--model:training()
print('model structure is:')
print(model)
local n = 0
for _,file in ipairs(files) do
	local tempInput = image.load(file)
	print(n)
	--print(tempInput:size())        
	tempInput = image.scale(tempInput, n_W, n_H, bilinear)
        --print(tempInput:size())
	local height = tempInput:size(2)
	local width = tempInput:size(3)
	local savColor = string.gsub(file,'../../datasets/MPI/',saveDir)
	--print('savColor0: ' .. savColor)
	savColor = saveDir.."/MPI-main-clean/"..tostring(n)..".png"
	--print('savColor1: ' .. savColor)
	--print('load images')
	local labelAFile = string.gsub(file,'clean','albedo')
	local labelSFile = string.gsub(file,'clean','shading')
	local templabelA = image.load(labelAFile)
	local templabelS = image.load(labelSFile)
        templabelA = image.scale(templabelA, n_W, n_H, bilinear)
        templabelS = image.scale(templabelS, n_W, n_H, bilinear)
	--print(file)
	--print(labelAFile)
	--print(labelSFile)

	--local savLabelA = string.gsub(savColor,'.png','-label-albedo.png')
	--local savLabelS = string.gsub(savColor,'.png','-label-shading.png')
	--local savAlbedo = string.gsub(savColor,'.png','-predict-albedo.png')
	--local savShading = string.gsub(savColor,'.png','-predict-shading.png')
	--local savColor = string.gsub(savColor,'.png','-input.png')

	local savLabelA = string.gsub(savColor,'.png','_reflect-real.png')
	local savLabelS = string.gsub(savColor,'.png','_shading-real.png')
	local savAlbedo = string.gsub(savColor,'.png','_reflect-pred.png')
	local savShading = string.gsub(savColor,'.png','_shading-pred.png')
	local savColor = string.gsub(savColor,'.png','-input.png')

	local input = torch.CudaTensor(1, 3, height, width)
	local labelA = torch.CudaTensor(1, 3, height, width)
	local labelS = torch.CudaTensor(1, 3, height, width)

	input[1] = tempInput
	labelA[1] = templabelA
	labelS[1] = templabelS

	image.save(savColor,input[1])
	image.save(savLabelA,labelA[1])
	image.save(savLabelS,labelS[1])
	
	--print(input[1])

	input = input * 255

	--print('forward')
	local predictions = model:forward(input)
	predictionsA = predictions[8]
	predictionsS = predictions[1]

	--print('pred_A:')
	--print(predictionsA[1][1]) --[0,255]
	--print('label_A:')
	--print(labelA[1][1])  --[0,1.0]
	for m = 1,3 do
	 local numerator = torch.dot(predictionsA[1][m], labelA[1][m])
	 local denominator = torch.dot(predictionsA[1][m], predictionsA[1][m])
	 local alpha = numerator/denominator
	 print('numerator:')
	 print(numerator)
	 predictionsA[1][m] = predictionsA[1][m] * alpha
	 --predictionsA[1][m] = predictionsA[1][m] / 255.0
	end

	for m = 1,3 do
	 local numerator = torch.dot(predictionsS[1][m], labelS[1][m])
	 local denominator = torch.dot(predictionsS[1][m], predictionsS[1][m])
	 local alpha = numerator/denominator
	 predictionsS[1][m] = predictionsS[1][m] * alpha
	 --predictionsS[1][m] = predictionsS[1][m] / 255.0
	end

	image.save(savAlbedo,predictionsA[1])
	image.save(savShading,predictionsS[1])
	--print('saved')
	n = n + 1
end
