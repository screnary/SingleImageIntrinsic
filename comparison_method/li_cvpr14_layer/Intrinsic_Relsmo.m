function [R, S] = Intrinsic_Relsmo( I,  lambda )
% Layer Separation using Relative Smoothness (specific for intrinsic images)
% [R, S] = Intrinsic_Relsmo( I,  lambda )
% I: input
% lambda: control the smoothness of illuninace layer S

I(I>1) = 1;
eps = 1e-16;
I_log =  log(max(1/256,I));
[N,M,D] = size(I_log);

f1 = [1, -1];
f2 = [1; -1];
f3 = [0, -1, 0; 
      -1, 4, -1;
      0, -1, 0];

sizeI2D = [N,M];
otfFx = psf2otf(f1,sizeI2D);
otfFy = psf2otf(f2,sizeI2D);
otfL = psf2otf(f3,sizeI2D);

Denormin1 = abs(otfL).^2 ;
Denormin2 = abs(otfFx).^2 + abs(otfFy ).^2;
if D>1
    Denormin1 = repmat(Denormin1,[1,1,D]);
    Denormin2 = repmat(Denormin2,[1,1,D]);
end

R = I_log;

 tic

 thr = 1;  % may need to tune
 chrom_wt = 0.001;
 
for i = 1:5 % iteration #

    beta = 2^(i-1)/thr; 

    Denormin   = lambda*Denormin1 + beta*Denormin2;

    %% update g
    gFx = -imfilter(R,f1,'circular');
    gFy = -imfilter(R,f2,'circular'); 
    gL = imfilter(I_log,f3,'circular');
    
    h_g = repmat(mean(gFx,3),[1 1 3]);
    v_g = repmat(mean(gFy,3),[1 1 3]);
    h_c = gFx - h_g;
    v_c = gFy - v_g;

    th_g = sum((h_g.^2),3);
    th_c = sum((h_c.^2),3);
    tv_g = sum((v_g.^2),3);
    tv_c = sum((v_c.^2),3);

    th = and(th_g<1/beta, th_c<chrom_wt/beta);
    tv = and(tv_g<1/beta, tv_c<chrom_wt/beta);
    th = repmat(th,[1,1,D]);
    tv = repmat(tv,[1,1,D]);
    gFx(th)=0; gFy(tv)=0;
    

    %% compute reflectance (L1)
    Normin2 = [gFx(:,end,:) - gFx(:, 1,:), -diff(gFx,1,2)];
    Normin2 = Normin2 + [gFy(end,:,:) - gFy(1, :,:); -diff(gFy,1,1)];
    Normin1 =   fft2(imfilter(gL, f3, 'circular'));
    FR = (lambda*Normin1 + beta*fft2(Normin2))./(Denormin+eps);
    R = real(ifft2(FR));
    
   %% normalize reflectance (L1)
    for c = 1:D
        Rt = R(:,:,c);
        for k = 1:500
        dt = (sum(Rt(Rt>0)) + sum(Rt(Rt<log(1/256)) - log(1/256)))/numel(Rt);
        Rt = Rt-dt;
        if abs(dt)<1/numel(Rt)
            break; 
        end
        end
        R(:,:,c) = Rt;
    end
    
    R(R>0) = 0;
    R(R<log(1/256)) = log(1/256);
 
end

toc
R = exp(R);
S = mean(I./R,3);


