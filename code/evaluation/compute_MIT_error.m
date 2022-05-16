clear all;
% DataSet = 'MIT';
% % result_folder = ['FD+pers-fromMPI-test-imgs_ep190', '/'];  % not good
% result_folder = ['FD+pers-test-imgs_ep120', '/'];
% inputDir = ['../result-data/',DataSet,'/'];
% % inputDir = ['../result-data/',DataSet,'/'];
% DataDir = ['../datasets/',DataSet,'/'];

DataSet = 'MIT';
%result_folder = ['test-imgs_ep335', '/'];
result_folder = ['test-imgs_ep380', '/'];
inputDir = ['../../',...
    '/ckpoints-Basic-MIT_v2_continue-self-sup+ms+fd+pers',...
    '-MIT-decoder_Residual/log/'];

DataDir = ['../../datasets/',DataSet,'/'];
test_file = [DataDir, 'test.txt'];
images = importdata(test_file);  % a cell

mse_albedo = {};
mse_shading = {};
lmse = {};
lmse_albedo = {};
lmse_shading = {};

for m =1:length(images)
    disp(images{m})
    maskname_label = [DataDir, 'MIT-mask-fullsize/', images{m}];
    albedoname_predict = [inputDir result_folder num2str(m-1) '_reflect-pred.png'];
    shadingname_predict = [inputDir result_folder num2str(m-1) '_shading-pred.png'];
    albedoname_label = [inputDir result_folder num2str(m-1) '_reflect-real.png'];
    shadingname_label = [inputDir result_folder num2str(m-1) '_shading-real.png'];
    
    albedo_predict = im2double(imread(albedoname_predict));
    shading_predict = im2double(imread(shadingname_predict));
    albedo_label = im2double(imread(albedoname_label));
    shading_label = im2double(imread(shadingname_label));
    mask = (imread(maskname_label));
    V = mask > 0;

    V3 = repmat(V,[1,1,size(shading_label,3)]);  
    
    errs_grosse = nan(1, size(albedo_label,3));
    errs_grosse_albedo = nan(1, size(albedo_label,3));
    errs_grosse_shading = nan(1, size(albedo_label,3));
    for c = 1:size(albedo_label,3)
      errs_grosse(c) = 0.5 * MIT_mse(shading_predict(:,:,c), shading_label(:,:,c), V)...
          + 0.5 * MIT_mse(albedo_predict(:,:,c), albedo_label(:,:,c), V);
      errs_grosse_shading(c) = MIT_mse(shading_predict(:,:,c), shading_label(:,:,c), V);
      errs_grosse_albedo(c) = MIT_mse(albedo_predict(:,:,c), albedo_label(:,:,c), V);
    end
    lmse{m} = mean(errs_grosse);
    lmse_albedo{m} = mean(errs_grosse_albedo);
    lmse_shading{m} = mean(errs_grosse_shading);
    
    alpha_shading = sum(shading_label(V3) .* shading_predict(V3)) ./ max(eps, sum(shading_predict(V3) .* shading_predict(V3)));
    S = shading_predict * alpha_shading;

    alpha_reflectance = sum(albedo_label(V3) .* albedo_predict(V3)) ./ max(eps, sum(albedo_predict(V3) .* albedo_predict(V3)));
    A = albedo_predict * alpha_reflectance;

    mse_shading{m} =  mean((S(V3) - shading_label(V3)).^2);
    mse_albedo{m} =  mean((A(V3) - albedo_label(V3)).^2);
end

ave_lmse = 0;
ave_lmse_albedo = 0;
ave_lmse_shading = 0;
ave_mse_albedo = 0;
ave_mse_shading = 0;
for m =1:length(images)
    ave_lmse = ave_lmse + log(lmse{m});
    ave_lmse_albedo = ave_lmse_albedo + log(lmse_albedo{m});
    ave_lmse_shading = ave_lmse_shading + log(lmse_shading{m});

    ave_mse_albedo = ave_mse_albedo + log(mse_albedo{m});
    ave_mse_shading = ave_mse_shading + log(mse_shading{m});
end
ave_lmse = exp(ave_lmse/length(images));
ave_lmse_albedo = exp(ave_lmse_albedo/length(images));
ave_lmse_shading = exp(ave_lmse_shading/length(images));
ave_mse_albedo = exp(ave_mse_albedo/length(images));
ave_mse_shading = exp(ave_mse_shading/length(images));

disp(result_folder);
disp(sprintf('albedo: mse: %f, lmse: %f, total_lmse: %f',ave_mse_albedo,ave_lmse_albedo,ave_lmse));
disp(sprintf('shading: mse: %f, lmse: %f, total_lmse: %f',ave_mse_shading,ave_lmse_shading,ave_lmse));