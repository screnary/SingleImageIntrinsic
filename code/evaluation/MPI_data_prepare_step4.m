% % % check the refined_gray scale shading, for I = A.*S formula
% % % step 4: statistics of refined shadings

clear all;
close all;

DataSet = 'MPI';

DataDir = ['../datasets/',DataSet,'/origin/'];
OutDir = ['../datasets/', DataSet, '/refined_gs_v3/'];
OutDir_new = ['../datasets/', DataSet, '/refined_check_202012/'];

% OutDir = DataDir;
InShadingDir = [DataDir, 'MPI-main-shading/'];

MaskDir = [DataDir, 'MPI-main-mask/'];
InputDir = [OutDir, 'MPI-main-clean/'];
AlbedoDir = [OutDir, 'MPI-main-albedo/'];
ShadingDir = [OutDir, 'MPI-main-shading/'];
file_lists = dir([MaskDir, '*.png']);  %(890 imgs)

run synset_settings;
% synnames = {'market_2', 'market_5', 'sleeping_2'};
synnames = {'market_5'};
for syn_i = 1:length(synnames)
    synname = synnames{syn_i};
    start_id = 1;
    count = 50;
    for syn_id = 1:size(synsets,1)
        if strcmp(synname, synsets{syn_id, 1})
            start_id = synsets{syn_id, 2};
            count = synsets{syn_id, 3};
            break;
        end
    end

    mu_stack = [];
    sigma_stack = [];
    statistics = [];

    mu_stack_ref = [];
    sigma_stack_ref = [];
    statistics_ref = [];

    disp(['processing synname: ', synname]);
    for iter = 0:(count-1)
        n = iter + start_id;
        frame_name = file_lists(n).name;

        inShadingName = [InShadingDir frame_name];
        albedoName = [AlbedoDir frame_name];
        shadingName = [ShadingDir frame_name];
        maskName = [MaskDir frame_name];
        inputName = [InputDir frame_name];

%         disp(['...Processing: ', num2str(n), '; FileName: ', frame_name]);

        inShading = im2double(imread(inShadingName));
        input = im2double(imread(inputName));
        albedo = im2double(imread(albedoName));
        shading = im2double(imread(shadingName));
        [height, width, channel] = size(albedo);

        inShading_lab = rgb2lab(inShading);
        input_lab = rgb2lab(input);
        albedo_lab = rgb2lab(albedo);
        shading_lab = rgb2lab(shading);

        L_in_s = inShading_lab(:,:,1);
        L_i = input_lab(:,:,1);  % the Luminance channel, [0,100]
        L_a = albedo_lab(:,:,1);
        L_s = shading_lab(:,:,1);

        mu = mean(L_s(:));
        sigma = std(L_s(:));
        ps = get_ps_curve(uint8(L_s), 100);
        mu_stack = [mu_stack, mu];
        sigma_stack = [sigma_stack, sigma];
        statistics = [statistics; ps];

        mu_ref = mean(L_in_s(:));
        sigma_ref = std(L_in_s(:));
        ps_ref = get_ps_curve(uint8(L_in_s), 100);
        mu_stack_ref = [mu_stack_ref, mu_ref];
        sigma_stack_ref = [sigma_stack_ref, sigma_ref];
        statistics_ref = [statistics_ref; ps_ref];
    %     maskimg = repmat(imresize(imread(maskName), [height, width], 'nearest'),[1,1,3]);
    %     valid_idx = maskimg == 255;
    end
    %% get the jittered frames
    tmp_a = zeros(size(statistics,1),1);
    tmp_b = zeros(size(statistics,1),1);
    tmp_a(2:end) = sum(abs(statistics(2:end,:) - statistics(1:end-1,:)), 2);
    tmp_b(2:end) = sum(abs(statistics_ref(2:end,:) - statistics_ref(1:end-1,:)), 2);
    var = std(tmp_a);
    frame_ids = double((tmp_a - tmp_b) > var);
    % flood the vally
    st_res = [];
    ed_res = [];
    st = -1;
    ed = -1;
    for i = 1:length(tmp_a)
        cur_flag = frame_ids(i);
        if st == -1
            % skip 0 elements
            if cur_flag == 0
                continue;
            elseif cur_flag == 1
                st = i;
            end
        elseif ed == -1
            % st pointer already assigned
            if cur_flag == 0
                continue;
            else
                % cur_flag == 1
                if frame_ids(i-1) == 1
                    continue;
                else
                    ed = i;
                end
            end
        end

        if st ~= -1 && ed ~= -1
            st_res = [st_res, st];
            ed_res = [ed_res, ed];
            % reset
            st = -1;
            ed = -1;
        end

        if i == length(tmp_a)
            if st ~= -1 && ed ~= -1
                st_res = [st_res, st];
                ed_res = [ed_res, ed];
            elseif st ~= -1 && ed == -1
                ed = i;
                st_res = [st_res, st];
                ed_res = [ed_res, ed];
            end
        end
    end
    %% process the jittered frames
    % % use python code align_image_pixel.py instead
%     num_interval = length(st_res);
%     for m = 1:num_interval
%         st = st_res(m);
%         ed = ed_res(m);
%         if ed==st
%             ed = ed+1;
%         end
%         for i = st:ed-1
%             frame_name = file_lists(start_id+i-1).name;
% 
%             inShadingName = [InShadingDir frame_name];
%             albedoName = [AlbedoDir frame_name];
%             shadingName = [ShadingDir frame_name];
%             maskName = [MaskDir frame_name];
%             inputName = [InputDir frame_name];
% 
%             disp(['...Processing: ', num2str(start_id+i-1), '; FileName: ', frame_name]);
% 
%             inShading = im2double(imread(inShadingName));
%             shading = im2double(imread(shadingName));
%             [height, width, channel] = size(albedo);
% 
%             inShading_lab = rgb2lab(inShading);
%             shading_lab = rgb2lab(shading);
% 
%             L_in_s = inShading_lab(:,:,1);
%             L_s = shading_lab(:,:,1);
%             mu_ref = (mu_stack(st-1) + mu_stack(st-1))/2;
%             std_ref = (sigma_stack(st-1) + sigma_stack(st-1))/2;
%             L_s_ = (L_s - mean(L_s(:))) * 0.99*std_ref / std(L_s(:)) + 0.9*mu_ref;
%             [L_s_shift, mask_shift_1] = shift_ps_curve(uint8(L_s_), max(uint8(L_s_(:))), 96);
% %             [L_s_shift, mask_shift_2] = shift_ps_curve(L_s_shift, min(uint8(L_s_shift(:))), 5);
%             disp(['......max luminance of shifted shading is ', num2str(max(L_s_shift(:)))]);
%             disp(['......min luminance of shifted shading is ', num2str(min(L_s_shift(:)))]);
% 
%             ps_before = get_ps_curve(uint8(L_s), 100);
%             ps_shift = get_ps_curve(uint8(L_s_shift), 100);
%             shading_shift_new = repmat(double(L_s_shift) / 100, [1,1,3]);
%             shading_new = repmat(double(L_s_) / 100, [1,1,3]);
%     %         figure, imshow([shading_new; shading_shift_new]);
%     %         figure, bar(ps_before), hold on, bar(ps_shift);
%             imwrite(shading_shift_new, [OutDir_new, 'MPI-main-shading/', frame_name]);
        end
    end

end