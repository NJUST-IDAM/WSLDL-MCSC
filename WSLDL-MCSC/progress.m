function [target,grad]=progress(b,finalX,finalY,finalW,option,admmrho,Gamma,Z,S)

%para
mu=option.mu;
gama=option.gama;
lambda=option.lambda;
delta=option.delta;

%objective
[xrow,xcol]=size(finalX);
ZX=Z(1:xrow,1:xcol+1);
ZY=Z(1:xrow,xcol+2:end);
maskZY=Z(1:xrow,xcol+2:end).*finalW;

temp1=0.5*norm(ZX-[finalX,b],'fro')^2+gama*norm(maskZY-finalY,'fro')^2;
sum0=0;
for i=1:xrow
    for j=1:xrow
        x1 = finalX(i,:);
        x2 = finalX(j,:);
        y1 = ZY(i,:);
        y2 = ZY(j,:);
        sum0 = sum0+norm(x1-x2)^2+0.5*delta*norm(y1-y2)^2;
    end
end
temp2=lambda*sum0+0.5*norm(ZY*ones(size(ZY,2),1)-ones(size(ZY,1),1))^2;
temp3=0.5*admmrho*norm(Z-S)^2+sum(sum(Gamma.*(Z-S)));
target = temp1+temp2+temp3;

%gradient
Il=ones(size(ZY,2),1);
In=ones(size(ZY,1),1);

GX=ZX-[finalX,b];
rel1=computeRel(ZY);
GY=2*gama*(ZY-finalY).*finalW+ZY*Il-In+lambda*delta*rel1;
GZ=admmrho*(Z-S)+Gamma;
grad=GZ+[GX,GY];
end

function [distance]=computeRel(ZY)
    distance=0;
    row=size(ZY,1);
        for i=1:row-1
            for j=i+1 :row
                distance =distance+euclideandist(ZY(i,:),ZY(j,:));
            end
        end
end