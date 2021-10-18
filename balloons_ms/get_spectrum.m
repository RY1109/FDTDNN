dirOutput_train = dir(fullfile('*.png'));
fileNames_train = sortnat({dirOutput_train.name})'; 
hyper = zeros(512,512,31);
for i=1:31
	hyper(:,:,i) = im2double(imread(char(fileNames_train(i))));
end
hyper=reshape(hyper,[512*512,31]);
hyper = interp1([1:31],hyper',[1:0.34:31])';
hyper = hyper(:,1:89);
% m=max(max(hyper));
% n=min(min(hyper));
% hyper = (hyper-n)/(m-n);
r=randperm(size(hyper,1));   %生成关于行数的随机排列行数序列
rand_hyper=hyper(r,:);       
train = rand_hyper(1:262144*0.6,:);
test = rand_hyper(262144*0.6:262144*0.9,:);
val =rand_hyper(262144*0.9:end,:);
save('balloons_train','train','-v7');
save('balloons_test','test','-v7');
save('balloons_val','val','-v7');