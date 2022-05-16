clear all;
totalMSEA = 0;
totalLMSEA = 0;
totalDSSIMA = 0;
totalMSES = 0;
totalLMSES = 0;
totalDSSIMS = 0;
count = 0;

DataSet = 'MPI';
result_folder = ['test-imgs_ep300', '/'];
inputDir = ['../',...
    '/ckpoints-Basic-self-sup+ms+fd+pers+vgg19_v11',...
    '-MPI-main-sceneSplit-decoder_Residual/log/'];
DataDir = ['../datasets/',DataSet,'/'];
MaskDir = [DataDir, 'MPI-main-mask/'];
test_file = [DataDir, 'MPI_main_sceneSplit-fullsize-NoDefect-test.txt'];
images = importdata(test_file);  % a cell

for n = 1:length(images)
    albedoName = [inputDir result_folder num2str(n-1) '_reflect-pred.png'];
    shadingName = [inputDir result_folder num2str(n-1) '_shading-pred.png'];
    labelAlbedoName = [inputDir result_folder num2str(n-1) '_reflect-real.png'];
    labelShadingName = [inputDir result_folder num2str(n-1) '_shading-real.png'];
    maskName = [MaskDir, images{n}];

    albedo = im2double(imread(albedoName));
    labelAlbedo = im2double(imread(labelAlbedoName));
    shading = im2double(imread(shadingName));
    labelShading = im2double(imread(labelShadingName));
    [height, width, channel] = size(albedo);
    maskimg = repmat(imresize(imread(maskName), [height, width], 'nearest'),[1,1,3]);
    valid_idx = maskimg == 255;
    albedo(~valid_idx) = labelAlbedo(~valid_idx);
    shading(~valid_idx) = labelShading(~valid_idx);
    totalMSEA = totalMSEA + evaluate_one_k(albedo,labelAlbedo);
    totalLMSEA = totalLMSEA + levaluate_one_k(albedo,labelAlbedo);
    totalDSSIMA = totalDSSIMA + (1-evaluate_ssim_one_k_fast(albedo,labelAlbedo))/2;

    totalMSES = totalMSES + evaluate_one_k(shading,labelShading);
    totalLMSES = totalLMSES + levaluate_one_k(shading,labelShading);
    totalDSSIMS = totalDSSIMS + (1-evaluate_ssim_one_k_fast(shading,labelShading))/2;

    count = count + 1;
    if length(images) >= 100 && mod(count,100)==0
        disp(count);
    elseif length(images) < 100
        disp(count);
    end
end
totalMSEA = totalMSEA/count;
totalLMSEA = totalLMSEA/count;
totalDSSIMA = totalDSSIMA/count;
totalMSES = totalMSES/count;
totalLMSES = totalLMSES/count;
totalDSSIMS = totalDSSIMS/count;
disp('albedo mse: shading mse:   albedo lmse: shading lmse:   albedo dssim: shading dssim:');
disp(sprintf('%f  %f\t%f  %f\t%f  %f', totalMSEA, totalMSES, totalLMSEA, totalLMSES, totalDSSIMA, totalDSSIMS));
% disp(sprintf('albedo: mse: %f, lmse: %f, dssim: %f',totalMSEA,totalLMSEA,totalDSSIMA));
% disp(sprintf('shading: mse: %f, lmse: %f, dssim: %f',totalMSES,totalLMSES,totalDSSIMS));