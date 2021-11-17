path = dir('./data');
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
    hyper2 = reshape(hyper,[512,512,89]);
    % m=max(max(hyper));
    % n=min(min(hyper));
    % hyper = (hyper-n)/(m-n);
    name = char(fileNames_train(1));
    name = name(1:end-10);
    save(['./U_net/',name,'.tif'],'hyper2','-v7');
end