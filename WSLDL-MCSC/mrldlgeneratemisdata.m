function []=mrldlgeneratemisdata(features,labels,misrate)

feature=features;
label=labels;

r=randperm(size(feature,1) ); %����feature��ÿһ��
[row,col]=size(feature);%cow:������ʾ������Ŀ��  row:��������������Ŀ��
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


