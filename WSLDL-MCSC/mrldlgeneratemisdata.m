function []=mrldlgeneratemisdata(features,labels,misrate)

feature=features;
label=labels;

r=randperm(size(feature,1) ); %打乱feature的每一行
[row,col]=size(feature);%cow:行数（示例的数目）  row:列数（特征的数目）
feature = feature(r,:);
label = label(r,:);

 for j=misrate:misrate
        s = RandStream.create('mt19937ar','seed',1);
        RandStream.setGlobalStream(s);
        [mis_label] = mrldlmisdata(label', 0.1*j);
        mis_label=mis_label';
 end
 
cd('./data');
save tempData.mat  feature  label mis_label;
cd('../');

fprintf('\nFinished!\n');
end


