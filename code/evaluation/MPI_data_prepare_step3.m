% % % check the refined_gray scale shading, for I = A.*S formula

clear all;
close all;

DataSet = 'MPI';

DataDir = ['../datasets/',DataSet,'/'];
OutDir = ['../datasets/', DataSet, '/refined_gs/'];
OutDir_new = ['../datasets/', DataSet, '/refined_gs_check/'];

MaskDir = [DataDir, 'MPI-main-mask/'];
InputDir = [OutDir, 'MPI-main-clean/'];
AlbedoDir = [OutDir, 'MPI-main-albedo/'];
ShadingDir = [OutDir, 'MPI-main-shading/'];
file_lists = dir([ShadingDir, '*.png']);

for n = 1:length(file_lists)
 
    frame_name = file_lists(n).name;
    albedoName = [AlbedoDir frame_name];
    shadingName = [ShadingDir frame_name];
    maskName = [MaskDir frame_name];
    inputName = [InputDir frame_name];
    
    disp(['...Processing: ', num2str(n), '; FileName: ', frame_name]);

    input = im2double(imread(inputName));
    albedo = im2double(imread(albedoName));
    shading = im2double(imread(shadingName));
    [height, width, channel] = size(albedo);
    maskimg = repmat(imresize(imread(maskName), [height, width], 'nearest'),[1,1,3]);
    valid_idx = maskimg == 255;
    
    %% check iamge normalization
    mu = 0.5;
    std = 0.5;
    norm_i = (input - mu)*std;
    norm_a = (albedo - mu)*std;
    norm_s = (shading - mu)*std;
    
    I_rec_norm = norm_a .* norm_s;
    I_rec = I_rec_norm / std + mu;
    
    %% simple assumption: I = A.*S
    input_rec = albedo .* shading;
    albedo_rec = input ./ (shading);
    shading_rec = input ./ (albedo);

    shading_zero_mask = shading_rec(:,:,1)==0 | shading_rec(:,:,2)==0 | shading_rec(:,:,3)==0;
    shading_inf_mask = shading_rec(:,:,1)>1 | shading_rec(:,:,2)>1 | shading_rec(:,:,3)>1;
    shading_invalid_mask = shading_zero_mask | shading_inf_mask;
    if sum(shading_invalid_mask(:)) > 0
        disp('Warning! shading_rec has invalid values!')
        disp('albedo:'), disp([min(albedo(:)), max(albedo(:))])
        disp('shading:'), disp([min(shading(:)), max(shading(:))])
        disp('input:'), disp([min(input(:)), max(input(:))])
        disp('shading_rec:'), disp([min(shading_rec(:)), max(shading_rec(:))])
        figure(1), imshow([shading_zero_mask; shading_inf_mask]);
    end
    
    albedo_zero_mask = albedo_rec(:,:,1)==0 | albedo_rec(:,:,2)==0 | albedo_rec(:,:,3)==0;
    albedo_inf_mask = albedo_rec(:,:,1)>1 | albedo_rec(:,:,2)>1 | albedo_rec(:,:,3)>1;
    albedo_invalid_mask = albedo_zero_mask | albedo_inf_mask;
    if sum(albedo_invalid_mask(:))
        disp('Warning! albedo_rec has invalid values!')
        disp('albedo:'), disp([min(albedo(:)), max(albedo(:))])
        disp('shading:'), disp([min(shading(:)), max(shading(:))])
        disp('input:'), disp([min(input(:)), max(input(:))])
        disp('albedo_rec:'), disp([min(albedo_rec(:)), max(albedo_rec(:))])
        figure(2), imshow([albedo_zero_mask; albedo_inf_mask]);
    end
    
    input_zero_mask = input_rec(:,:,1)==0 | input_rec(:,:,2)==0 | input_rec(:,:,3)==0;
    input_inf_mask = input_rec(:,:,1)>1 | input_rec(:,:,2)>1 | input_rec(:,:,3)>1;
    input_invalid_mask = input_zero_mask | input_inf_mask;
    if sum(albedo_invalid_mask(:))
        disp('Warning! input_rec has invalid values!')
        disp('albedo:'), disp([min(albedo(:)), max(albedo(:))])
        disp('shading:'), disp([min(shading(:)), max(shading(:))])
        disp('input:'), disp([min(input(:)), max(input(:))])
        disp('input_rec:'), disp([min(input_rec(:)), max(input_rec(:))])
        figure(3), imshow([input_zero_mask; input_inf_mask]);
    end
    %% save files
%     imwrite([input; input_rec], [OutDir_new, 'MPI-main-clean/', frame_name]);
%     imwrite([albedo; albedo_rec], [OutDir_new, 'MPI-main-albedo/', frame_name]);
%     imwrite([shading; shading_rec], [OutDir_new, 'MPI-main-shading/', frame_name]);

%     imwrite([input; input_new], [DumpDir, num2str(n,'%04d'),'_input-simple', '.png']);
%     imwrite([albedo; albedo_new], [DumpDir, num2str(n,'%04d'), '_reflectance-simple', '.png']);
%     imwrite([shading; shading_new], [DumpDir, num2str(n,'%04d'), '_shading-simple', '.png']);
end
% disp(sprintf('albedo: mse: %f, lmse: %f, dssim: %f',totalMSEA,totalLMSEA,totalDSSIMA));
% disp(sprintf('shading: mse: %f, lmse: %f, dssim: %f',totalMSES,totalLMSES,totalDSSIMS));