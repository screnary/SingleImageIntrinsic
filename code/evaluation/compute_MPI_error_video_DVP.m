clear all;
totalMSEA = 0;
totalLMSEA = 0;
totalDSSIMA = 0;
totalMSES = 0;
totalLMSES = 0;
totalDSSIMS = 0;
count = 0;

% Deep Video Prior

DataSet = 'MPI';
% inputDir_in = ['../../intrinsic_image_project/',...
%     'framewise-ckpoints-v8_self-sup+ms+fd+pers-MPI-main-RD-sceneSplit-w_ssim-0.5-w_grad-1.5',...
%     '/log/test-imgs_ep195_renamed/'];
inputDir_in = ['../../intrinsic_image_project/',...
    'framewise-ckpoints-v8_self-sup+ms+fd+pers-MPI-main-RD-sceneSplit-w_ssim-0.5-w_grad-1.5',...
    '/log/test-imgs_ep200_renamed/'];
% inputDir_res = ['../../3rdparty/deep-video-prior/result/Intrinsic_IRT0_initial0/'];
inputDir_res = ['../../3rdparty/deep-video-prior/result/Intrinsic_IRT0_initial0_ep200/'];
result_folder_r = ['reflect', '/0025/'];
result_folder_s = ['shading', '/0025/'];

% for DI setting
%inputDir = inputDir_1;
%result_folder = result_folder_1

DataDir = ['../../intrinsic_image_project/datasets/',DataSet,'/'];

test_file = [DataDir, 'MPI_main_sceneSplit-fullsize-NoDefect-test.txt'];
% test_file = [DataDir, 'MPI_main_imageSplit-fullsize-ChenSplit-test.txt'];
images = importdata(test_file);  % a cell, 'bandage_2_frame_0028.png'

disp([inputDir_res, result_folder_r]);
disp(inputDir_in);

for n = 1:length(images)
    img_name = images{n};

    albedoName =  [inputDir_res result_folder_r img_name '.jpg'];
    shadingName = [inputDir_res result_folder_s img_name '.jpg'];
    labelAlbedoName = [inputDir_in img_name(1:end-4) '_reflect-real.png'];
    labelShadingName = [inputDir_in img_name(1:end-4) '_shading-real.png'];

    albedo = im2double(imread(albedoName));
    labelAlbedo = im2double(imread(labelAlbedoName));
    shading = im2double(imread(shadingName));
    labelShading = im2double(imread(labelShadingName));
    [height, width, channel] = size(albedo);

    labelAlbedo = labelAlbedo(1:height, 1:width, :);
    labelShading = labelShading(1:height, 1:width, :);

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