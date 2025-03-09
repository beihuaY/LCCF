clc;
clear; close all;


addpath ./ClusteringMeasure/
addpath ./functions
path = './data/';


Database = 'MSRC';  
percentDel = 0.1;

Datafold = [path, Database];
load(Datafold)
Indexfold = [path,'Index/Index_',Database,'_percentDel_',num2str(percentDel),'.mat'];
load(Indexfold)
  

gt=Y;
for i=1:5
    X{i}=X{i}';
end
cls_num = numel(unique(gt));
param.cls_num = cls_num;
perf = []; 
Xc = X;
ind = Index{1};


for i=1:length(Xc)
        Xci = Xc{i};
        Xci = NormalizeFea(Xci,0);
        indi = ind(:,i);
        pos = find(indi==0);
        Xci(:,pos)=[]; 
        Xc{i} = Xci;
end   

clear Xci i indi pos 

param.sp = [3   2  1  1   1]';
param.rho=0.7;
param.alpha =0.1;
param.beta =0.01;
param.lambda  =0.001;
param.gamma=10;


[YY,alpha,converge_G,G] = YK(Xc,gt,ind,param);
[~, Clus] = max(YY,[],2);
NMI = nmi(Clus,gt);
Purity = purity(gt, Clus);
ACC = Accuracy(Clus,double(gt));
[Fscore,Precision,~] = compute_f(gt,Clus);
[AR,RI,~,~]=RandIndex(gt,Clus);
results_log = [NMI,ACC,Purity,Fscore,Precision];
fprintf('result:\tNMI:%f, ACC:%f, Purity:%f, Fsocre:%f, Precision:%f\n',results_log);
