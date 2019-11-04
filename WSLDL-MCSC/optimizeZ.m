function [Z,fval,exitFlag,output,grad]=optimizeZ(funfcn,xInit,optim)
OPTIONS.MaxIter=10;
% fprintf('Begin training of BFGS-LCLLD-W. \n');

% Read Optimalisation Parameters%检测optim中的变量是否存在 如果不存在返回0,存在返回1
if (~exist('optim','var')) 
    % Function is written by D.Kroon University of Twente (Updated Nov.
    % 2010).
    [Z,fval,exitFlag,output,grad] = fminlbfgs(funfcn,xInit);

else
    % Function is written by D.Kroon University of Twente (Updated Nov.
    % 2010).
    [Z,fval,exitFlag,output,grad] = fminlbfgs(funfcn, xInit,optim);
end
end