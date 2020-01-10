% if svhn
data = load('./Digit-Five/svhn_train_32x32.mat');
% data = load('./Digit-Five/svhn_test_32x32.mat');
X = data.X;
y = data.y;

% If synth data
% data = load('./Digit-Five/syn_number.mat');
% X = permute(data.train_data, [2, 3, 4, 1]);
% y = permute(data.train_label, [2, 3, 4, 1]);
% X = permute(data.test_data, [2, 3, 4, 1]);
% y = permute(data.test_label, [2, 3, 4, 1]);

[w,h,d,n] = size(X);
len = length(y);
X_final = zeros(28,28,d,len);

for i=1:len
    im = X(:,:,:,i);

    %print(size(im));
    %fprintf('%d %d %d\n', size(im, 1), size(im,2),size(im,3))
    im = imresize(im, [28,28], 'cubic');
    X_final(:,:,:,i) = im;

end

X_final = uint8(X_final);
X = X_final;
y = y(1:len);
save( '-v6', './Digit-Five/svhn_train_28x28.mat', 'X', 'y');
% save('-v6', './Digit-Five/svhn_test_28x28.mat', 'X', 'y');

% save( '-v6', './Digit-Five/synth_train_28x28.mat', 'X', 'y');
% save( '-v6', './Digit-Five/synth_test_28x28.mat', 'X', 'y');
