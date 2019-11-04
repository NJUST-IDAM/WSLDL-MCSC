function newS=optimizeS(option,admmrho,Gamma,Z)
[U,Sig,V]=svd(Z+Gamma./admmrho);
[row,col]=size(Sig);
temp = zeros(size(Sig,1),size(Sig,2));
for i=1:row
    for j=1:col
        temp(i,j)=max(Sig(i,j)-option.mu/admmrho,0);
    end
end
newS = U*temp*V';

end