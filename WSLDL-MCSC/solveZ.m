function [target,grad]=solveZ(b,finalX,finalY,finalW,option,admmrho,Gamma,Z,S,L)

%para
mu=option.mu;
gama=option.gama;
lambda=option.lambda;
delta=option.delta;

[xrow,xcol]=size(finalX);
ZX=Z(1:xrow,1:xcol+1);
ZY=Z(1:xrow,xcol+2:end);
maskZY=Z(1:xrow,xcol+2:end).*finalW;

%object
temp1=0.5*norm(ZX-[finalX,b],'fro')^2+gama*norm(maskZY-finalY,'fro')^2;
temp2=lambda*trace(ZY'*L*ZY)+0.5*norm(ZY*ones(size(ZY,2),1)-ones(size(ZY,1),1))^2;
temp3=0.5*admmrho*norm(Z-S)^2+sum(sum(Gamma.*(Z-S)));
target = temp1+temp2+temp3;


%gradient
Il=ones(size(ZY,2),1);
In=ones(size(ZY,1),1);

GX=ZX-[finalX,b];
T=ZY*Il-In;
GY=2*gama*(ZY-finalY).*finalW+repmat(T,1,size(ZY,2))+lambda*(L+L')*ZY;
%GY=2*gama*(ZY-finalY).*finalW+ZY*Il-In+lambda*(L+L')*ZY;
GZ=admmrho*(Z-S)+Gamma;
grad=GZ+[GX,GY];

% SX = S(:,1:xcol+1);
% SY = S(:,xcol+2:end);
% GammaX = Gamma(:,1:xcol+1);
% GammaY = Gamma(:,xcol+2:end);
% finalZX = ([finalX,b]+admmrho*SX-GammaX)./(1+admmrho);
% finalZY1 = 1./(2*gama+lambda*(L+L')+admmrho)*(2*gama*finalY+admmrho*SY-GammaY);
% finalZY2 = 1./(lambda*(L+L')+admmrho)*(admmrho*SY-GammaY);
% finalZY = finalZY1.*finalW+finalZY2;
% col=size(finalZY,2);
% temp=sum(finalZY,2);
% finalZY = finalZY./repmat(temp,1,col);
% finalZ = [finalZX,finalZY];
end

