function I_out = LLE_smoothing(I_defect, I_ref, mask, K)
% image RGB xy space guided disparity smoothing(filling hole), Using LLE
% input data
    % first, we need to resize the image to feasible size for KNN
%     rect = [350,370,170,190];
%     Iref = imcrop(I_ref, rect);
%     Idef = imcrop(I_defect, rect);
    Iref = I_ref;
    Idef = I_defect;
    [h,w,c] = size(Iref);
    RGB = reshape(Iref, [h*w,3]);
    Ilab = rgb2lab(Iref);
    Lab = reshape(Ilab, [h*w,3]);

    % feature distance and k nearest neighbors
    [I,J] = ind2sub([h,w],1:h*w);
    X = [I;J;Lab']; % data D*N; D=5, N=num of points

    %% find the points need to be computed (for inpainting)
%     SE = strel('square', 11);  %11, image dilation, 5
%     mask_dilated = imdilate(mask, SE);
    mask_dilated = mask;
    holeIndex = find(mask_dilated);
    N = length(holeIndex);
    fprintf(1,'-->Finding %d nearest neighbours for %d queries.\n',K, N);
    
    D = size(X,1);
    Nei = zeros(N,K);
    X_hole = X(:, holeIndex);
    X_filtered = X; % change this X_filtered to local patch
%     X_filtered(:, holeIndex) = -1; % filter out holes in Iref, check this
    w = 301;  % patch size
    h = 301;
    parfor i = 1:size(X_hole,2)
%     for i = 1:size(X_hole,2)
        patchIndex = get_patch(size(Idef), w, h, holeIndex(i)); % 100*100
        X_patch = X_filtered(:, patchIndex);
        X_tmp = repmat(X_hole(:,i),1,size(X_patch,2));
        distance = sum((X_patch-X_tmp).^2,1); % distance vector
        [B, I] = sort(distance); % B=distance(I)
        sorted_index = patchIndex(I);
        neighbor_index = sorted_index(~ismember(sorted_index, holeIndex));  % do not use hole pixel
%         neighbor_index = sorted_index;  % use hole pixel as well

%         neighbor_index = I(~ismember(I,holeIndex)); % used for non patch format, check this
%         neighbor_index = I(ismember(I,holeIndex));
%         neighbor_index = I(:);

        Nei(i,:) = neighbor_index(1:K);
    end

    %% construct reconstruction weights
    fprintf(1,'-->Solving for reconstruction weights.\n');
    if(K>D)
      fprintf(1,'   [note: K>D; regularization will be used]\n'); 
      tol=1e-3; % regularlizer in case constrained fits are ill conditioned
    else
      tol=0;
    end

    W = zeros(K, N);
    parfor ii = 1:N
        z = X(:, Nei(ii,:)) - repmat(X(:,ii),1,K); % shift ith pt to origin
        C = z'*z;                                  % local covariance
        C = C + diag(diag(C));
        C = C + eye(K,K)*tol*trace(C);             % regularlization (K>D)
        W(:,ii) = C\ones(K,1);                     % solve Cw=1
        W(:,ii) = W(:,ii)/sum(W(:,ii));            % enforce sum(w)=1
    end

    % interpolate implanting values
    % because the neighbors are outside from the set, we can compute directly
    defect_new = zeros(N,1);
    parfor jj = 1:N
        defect_new(jj) = double(Idef(Nei(jj,:))) * W(:,jj);
    end
    I_out = Idef;
    I_out(holeIndex) = defect_new;

