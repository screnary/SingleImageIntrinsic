clear all;
totalMSEA = 0;
totalLMSEA = 0;
totalDSSIMA = 0;
totalMSES = 0;
totalLMSES = 0;
totalDSSIMS = 0;
count = 0;

DataSet = 'MPI';

% frame_by_frame settings
%result_folder_1 = ['test-imgs_ep225', '/'];
%inputDir_1 = ['../../intrinsic_image_project/',...
%    'framewise-ckpoints-direct_intrinsics-MPI-main-RD-sceneSplit',...
%    '/log/'];

%result_folder_2 = ['test-imgs_ep200', '/'];
%inputDir_2 = ['../../intrinsic_image_project/',...
%    'framewise-ckpoints-v8_self-sup+ms+fd+pers-MPI-main-RD-sceneSplit-w_ssim-0.5-w_grad-1.5',...
%    '/log/'];

% for DI setting
%inputDir = inputDir_1;
%result_folder = result_folder_1

% for mine setting
%inputDir = inputDir_2;
%result_folder = result_folder_2

% optical flow settings
result_folder_1 = ['test-imgs_ep250', '/'];
inputDir_1 = ['/home/wzj/intrinsic/code/video-ckpoints-v8_self-sup+ms+fd+pers-MPI-main-RD-sceneSplit-w_flow-50.0-lambda_r-1.0-lambda_s-1.0', ...
              '/log/'];

result_folder_2 = ['test-imgs_ep250', '/'];
inputDir_2 = ['/home/wzj/intrinsic/code/video-ckpoints-v8_self-sup+ms+fd+pers-MPI-main-RD-sceneSplit-w_flow-50.0-lambda_r-4.0-lambda_s-0.0', ...
              '/log/'];

result_folder_3 = ['test-imgs_ep5', '/'];
inputDir_3 = ['/home/wzj/intrinsic/code/video-ckpoints-fromV8-v8_self-sup+ms+fd+pers-MPI-main-RD-sceneSplit-w_flow-50.0-lambda_r-10.0-lambda_s-5.0', ...
              '/log/'];

inputDir = inputDir_3;
result_folder = result_folder_3;

DataDir = ['../../intrinsic_image_project/datasets/',DataSet,'/'];

test_file = [DataDir, 'MPI_main_sceneSplit-fullsize-NoDefect-test.txt'];
% test_file = [DataDir, 'MPI_main_imageSplit-fullsize-ChenSplit-test.txt'];
images = importdata(test_file);  % a cell

disp([inputDir, result_folder]);

for n = 1:length(images)
    albedoName = [inputDir result_folder num2str(n-1) '_reflect-pred.png'];
    shadingName = [inputDir result_folder num2str(n-1) '_shading-pred.png'];
    labelAlbedoName = [inputDir result_folder num2str(n-1) '_reflect-real.png'];
    labelShadingName = [inputDir result_folder num2str(n-1) '_shading-real.png'];

    albedo = im2double(imread(albedoName));
    labelAlbedo = im2double(imread(labelAlbedoName));
    shading = im2double(imread(shadingName));
    labelShading = im2double(imread(labelShadingName));
    [height, width, channel] = size(albedo);

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