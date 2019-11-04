
clear;
clc;

cd('./data');
load Yeast_alpha;
cd('../');

misrate=6;
mrldlgeneratemisdata(features,labels,misrate);
clear;


current_path=cd;
dir=strcat(cd,'/data/');
load(strcat(dir,'tempData','.mat'));
X=feature;
tYY=label;
YY=mis_label;
W = ones(size(YY));
for i=1:size(YY,1)
    for j=1:size(YY,2)
        if YY(i,j)==0
            W(i,j)=0;
        end
    end
end


for rep=1:10
    indices = crossvalind('Kfold',size(feature,1),10);
    
    mu_set=[0.01];
    gama_set = [0.1];
    lambda_set = [1];
    delta_set = [0.0001];
    
    for iter1 = 1:length(mu_set)
        for iter2 = 1:length(gama_set)
            for iter3=1:length(lambda_set)
                for iter4 = 1:length(delta_set)
                    
                    option.mu=mu_set(iter1);
                    option.gama=gama_set(iter2);
                    option.lambda=lambda_set(iter3);
                    option.delta=delta_set(iter4);
                    
                    for v=1:10
                        tic;
                        fprintf('========================================rep=======================================>>>>>>: %d \n', rep);
                        fprintf('==========================================v=====================================>>>>>>: %d \n', v);
                        test = (indices == v);
                        train = ~test;
                        X_train=X(train,:);
                        X_test=X(test,:);
                        YY_train=YY(train,:);
                        YY_test=YY(test,:);
                        tYY_train=tYY(train,:);
                        tYY_test=tYY(test,:);
                        W_train = W(train,:);
                        W_test = W(test,:);
                        
                        
                        b=ones(size(X,1),1);
                        finalX=[X_train;X_test];
                        finalY=[YY_train;zeros(size(YY_test,1),size(YY_test,2))];
                        finalW=[W_train;zeros(size(W_test,1),size(W_test,2))];
                        
                        ii=11;
                        admmrho=2^(ii-11);
                        [newZ,mytime,convergence] = ADMMiter(b,finalX,finalY,finalW,option,admmrho,rep,v);

                        n0=size(X_train,1);
                        n1=size(X_train,2);
                        n2=size(b,2);
                        preDistribution = newZ(n0+1:end,n1+n2+1:end); 
                        com_YY = newZ(1:n0,n1+n2+1:end); 
                        testDistribution = tYY_test;
                        
                        
                        cd('./measures');
                        cow=size(testDistribution,1);
                        for i=1: cow
                            dist(i,1)=clark(testDistribution(i,:), preDistribution(i,:));
                            dist(i,2)=canberra(testDistribution(i,:), preDistribution(i,:));
                            dist(i,3)=kldist(testDistribution(i,:), preDistribution(i,:));
                            dist(i,4)=chebyshev(testDistribution(i,:), preDistribution(i,:));
                            dist(i,5)=intersection(testDistribution(i,:), preDistribution(i,:));
                            dist(i,6)=cosine(testDistribution(i,:), preDistribution(i,:));
                        end
                        cd('../');

                        for i=1:6
                            mea(i,v)=mean(dist(:,i));
                        end
                        toc
                        
                    end
                    
                    row=size(mea,1);
                    for i=1:row
                        meanres(rep,i)=mean(mea(i,:));
                    end
                end
            end
        end
    end
end

[row,col]=size(meanres);
for i=1:col
    finalmean(i)=mean(meanres(:,i));
    finalstd(i)=std(meanres(:,i));
end
finalmean
finalstd
