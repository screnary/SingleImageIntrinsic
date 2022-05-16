clear all;
total_mean_i = 0;
total_mean_a = 0;
total_mean_s = 0;
count = 0;

DataSet = 'MPI';
result_folder = ['test-imgs_ep150', '/'];
inputDir = ['../',...
    '/ckpoints-Basic-self-sup+ms+fd+pers+vgg19_v2',...
    '-MPI-main-sceneSplit-decoder_Residual/log/'];
DataDir = ['../datasets/',DataSet,'/'];
test_file = [DataDir, 'MPI_main_sceneSplit-fullsize-NoDefect-train.txt'];
images = importdata(test_file);  % a cell

for n = 1:length(images)
    inputName = [DataDir 'MPI-main-clean/' images{n}];
    albedoName = [DataDir 'MPI-main-albedo/' images{n}];
    shadingName = [DataDir 'MPI-main-shading/' images{n}];

%     albedoName = [inputDir result_folder num2str(n-1) '_reflect-pred.png'];
%     shadingName = [inputDir result_folder num2str(n-1) '_shading-pred.png'];
%     labelAlbedoName = [inputDir result_folder num2str(n-1) '_reflect-real.png'];
%     labelShadingName = [inputDir result_folder num2str(n-1) '_shading-real.png'];

    input = im2double(imread(inputName));
    albedo = im2double(imread(albedoName));
    shading = im2double(imread(shadingName));

    [height, width, channel] = size(albedo);
    mean_i = squeeze(mean(mean(input, 1), 2));
    mean_a = squeeze(mean(mean(albedo, 1), 2));
    mean_s = squeeze(mean(mean(shading, 1), 2));
    
    total_mean_i = total_mean_i + mean_i;
    total_mean_a = total_mean_a + mean_a;
    total_mean_s = total_mean_s + mean_s;

    count = count + 1;
    if length(images) >= 100 && mod(count,100)==0
        disp(count);
    elseif length(images) < 100
        disp(count);
    end
end
total_meani = total_mean_i/count;
total_meana = total_mean_a/count;
total_means = total_mean_s/count;
disp('mean_i:')
disp(total_meani')
disp('mean_a:')
disp(total_meana')
disp('mean_s:')
disp(total_means')