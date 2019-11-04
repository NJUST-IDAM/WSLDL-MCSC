function [newZ,mytime,convergence] = ADMMiter(b,finalX,finalY,finalW,option,admmrho,rep,v)

tic;
[n,d]=size(finalX);
max_iter=200;
convergence1=zeros(max_iter,1);
convergence2=zeros(max_iter,1);
convergence3=zeros(max_iter,1);

epsilon_primal=zeros(max_iter,1);
epsilon_dual=zeros(max_iter,1);
epsilon_abs=1e-4;
epsilon_rel=1e-3;

%��ʼ��
init_rank=6;
% [U,Sigma,V]=svds([finalX b finalY],init_rank);
% Z=U*Sigma*V';
% S=ones(size(Z,1),size(Z,2));
Z=[finalX b finalY];
S=Z;
Gamma=zeros(size(Z,1),size(Z,2));


% Gamma=ones(size(Z,1),size(Z,2));
mu=option.mu;
gama=option.gama;
lambda=option.lambda;
delta=option.delta;

para.NN=5; 
para.GraphDistanceFunction='cosine'; 
para.GraphWeights='distance';
para.GraphWeightParam=0;
para.LaplacianNormalize=0;
para.LaplacianDegree=0;
L=laplacian(para,[finalX,b]);  

t=0;
while(t<max_iter)
    t=t+1;
    fprintf('rep:%d v:%d Iteration: %d \n',rep,v,t);     
    Z=optimizeZ(@(Z)solveZ(b,finalX,finalY,finalW,option,admmrho,Gamma,Z,S,L),Z);
    [xrow,xcol]=size(finalX);
    Z(:,xcol+1)=1;    
    S0=S;
    S=optimizeS(option,admmrho,Gamma,Z);
    Gamma = Gamma + admmrho*(Z-S);
    
    
    %primal residual
    convergence1(t,1)=norm(Z-S,'fro');
    
    %dual residual
    convergence2(t,1)=norm(admmrho*(S-S0),'fro');
    
    %primal epsilon
    epsilon_primal(t,1)=sqrt(n)*epsilon_abs+epsilon_rel*max(norm(Z,'fro'),norm(S,'fro'));
    
    %dual epsilon
    epsilon_dual(t,1)=sqrt(d)*epsilon_abs+epsilon_rel*norm(Gamma,'fro');
    
    %objectiveĿ�꺯��ֵ
    [xrow,xcol]=size(finalX);
    ZX=Z(1:xrow,1:xcol+1);
    ZY=Z(1:xrow,xcol+2:end);
    maskZY=Z(1:xrow,xcol+2:end).*finalW;
    
    temp1=mu*sum(svd(Z,'econ'))+0.5*norm(ZX-[finalX,b],'fro')^2+gama*norm(maskZY-finalY,'fro')^2;
    temp2=trace(ZY'*L*ZY);
    convergence3(t,1)=temp1+lambda*temp2;
    convergence3(t,1)
    
    if (convergence1(t,1)<=epsilon_primal(t,1) && convergence2(t,1)<=epsilon_dual(t,1))
        break;
    end
end

convergence=cell(5,1);
convergence{1,1}=convergence1;
convergence{2,1}=convergence2;
convergence{3,1}=convergence3;
convergence{4,1}=epsilon_primal;
convergence{5,1}=epsilon_dual;

newZ = Z;
mytime = toc;
end