path = dir('./data')
lendir = length(path)
for j=3:lendir
    dirpath = path(j).name;
    dirOutput_train = dir(fullfile('./data/',dirpath,dirpath,'*.png'));
    fileNames_train = sortnat({dirOutput_train.name})'; 
    hyper = zeros(512,512,31);
    for i=1:31
        file_train = fullfile('./data/',dirpath,dirpath,fileNames_train(i));
        hyper(:,:,i) = im2double(imread(char(file_train)));
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
    name = char(fileNames_train(1))
    name = name(1:end-10)
    save(['./train/',name,'_train'],'train','-v7');
    save(['./test/',name,'_test'],'test','-v7');
    save(['./val/',name,'_val'],'val','-v7');
end