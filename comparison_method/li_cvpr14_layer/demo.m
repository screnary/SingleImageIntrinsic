close all;

%% Intrinsic Image Decomposition
disp('Intrinsic Images Example');
I1 = im2double(imread('intrinsic_in1.png')); 
[R S] = Intrinsic_Relsmo(I1, 2);
figure(1),
subplot 131, imshow(I1), title('input');
subplot 132, imshow(R); title('reflectace');
subplot 133, imshow(S); title('shading');


%% Reflection Separation Using Focus 
disp('Reflection Removal Example');
I2 = im2double(imread('reflection_in.jpg')); 
[H W D] = size(I2);
[LB LR] = septRelSmo(I2, 50, zeros(H,W,D), I2);
figure(2),
subplot 131, imshow(I2) , title('input');
subplot 132, imshow(LB*2), title('background'); 
subplot 133, imshow(LR*2), title('reflection');
