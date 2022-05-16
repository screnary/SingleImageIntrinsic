function I_out = LLE_smoothing_v0(I_defect, I_ref, mask, K)
% search neighbors from the whole image, rather than a local patch (LLE_smoothing)
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
    X = [I;J;Lab']; % data D*N

    %% find the points need to be computed (for inpainting)
    holeIndex = find(mask);
    N = length(holeIndex);
    fprintf(1,'-->Finding %d nearest neighbours for %d queries.\n',K, N);
    
    D = size(X,1);
    Nei = zeros(N,K);
    X_hole = X(:, holeIndex);
    X_filtered = X;
    X_filtered(:, holeIndex) = -1; % filter out holes in Iref
    parfor i = 1:size(X_hole,2)
        X_tmp = repmat(X_hole(:,i),1,size(X,2));
        distance = sum((X_filtered-X_tmp).^2,1); % distance vector
        [B, I] = sort(distance); % B=distance(I)
        neighbor_index = I(~ismember(I,holeIndex));
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