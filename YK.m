function  [F,alpha,converge_G,G] = YK(Xc,gt,index,param)

alpha = param.alpha; 
beta = param.beta; 
lambda = param.lambda; 
rho=param.rho;
sp=param.sp; 
gamma=param.gamma;
r  = 2;
mu = 0.0001; max_mu = 10e12;  pho_mu = 2;
c = param.cls_num;%聚类簇数
[n, num_view] = size(index);
R = cell(1, num_view); G = R; E=R; 
%------------------------initialize-------------------------------
for iv = 1:num_view
    pos0{iv} = find(index(:,iv) == 0);
    pos1{iv} = find(index(:,iv) == 1);  
    I = eye(n);
    I(pos0{iv},:) = [];
    R{iv} = I;%nv*n
  
    I = eye(n);
    I(:,pos0{iv}) = [];
    O{iv} = I;   %n*nv 
    di = size(Xc{iv},1); 

      V{iv}=ones(c,size(Xc{iv},2)); 
      W{iv}=ones(size(Xc{iv},2),c); 
      E{iv} = ones(size(Xc{iv},2),c);    
      G{iv} = O{iv}*V{iv}';
      P{iv} = eye(c);
      I_d{iv}=eye(di);
      L{iv} = ones(n, c);
      J{iv} = ones(n, c);
end
l = ones(n*c*num_view,1);
j = ones(n*c*num_view,1);
sX = [n, c, num_view];
converge_G=[];
sY = 0;
for iv = 1:num_view
   sY = sY + G{iv};
end
mm = max(sY,[],2);
F = 1 - double(sY < mm(:));
w = ones(1, num_view)/num_view;

MAXITER = 50;
%%
for iter = 1:MAXITER
    %%--------------------------optimize E nv*k----------------
    for iv=1:num_view
        E{iv} = prox_l12(V{iv}' - R{iv}*G{iv}, alpha/2);
    end
    %%--------------------------optimize W d*k----------------
     for iv = 1:num_view
        W{iv} = max(V{iv}',0);
     end
     %%-------------------------optimize V k*nv----------------
    for iv = 1:num_view
        Va=lambda*eye(c)+W{iv}'*Xc{iv}'*Xc{iv}*W{iv};
        Vb=lambda*G{iv}'*R{iv}'+W{iv}'*Xc{iv}'*Xc{iv}+lambda*E{iv}';
        V{iv} =Va\Vb; 
    end
    %%-------------------------optimize G n*k-----------------
    for iv=1:num_view
         Ua=V{iv}'-E{iv};
         Za=J{iv}-L{iv}/mu;
         Fa=lambda*R{iv}'*Ua+beta*w(iv)^r*F*P{iv}'+(mu/2)*Za;
         [Fa,~,Fb] = svd(Fa,'econ');
         G{iv} = Fa*Fb'; 
    end
  %%-------------------------optimize J  sp -------------------
     G_tensor = cat(3, G{:,:});
     L_tensor = cat(3, L{:,:});
     g = G_tensor(:);
     l= L_tensor(:);
     [j, ~] =weight_lp(g + 1/mu*l,gamma*sp./mu,sX, 0,3,rho);
     J_tensor = reshape(j, sX);
    for iv=1:num_view
        J{iv} = J_tensor(:,:,iv);
    end     
  %%-------------------------optimize P k*k-------------------
  for iv = 1:num_view
       [Uc,~, Zc]=svd(G{iv}'*F,'econ');
       P{iv} = Uc*Zc';  
  end
  %%-------------------------optimize Y k*n-------------------
     sY = 0;
    for iv = 1:num_view
        sY = sY + beta*w(iv)^r*G{iv}*P{iv};
    end
    mm = max(sY,[],2);
    F = 1 - double(sY < mm(:));  
   %%------------------------ optimize w ----------------------
    err = zeros(1, num_view);
    sw= 0;
    for iv = 1:num_view
        dd = F-G{iv}*P{iv};
        err(iv) = 1/sum(sum(dd.*dd));
        sw = sw + err(iv)^(1/(r-1));
    end
    w = err.^(1/(r-1))/sw;
    %%----------------------- optimize I -----------------------
     l = l + mu*(g- j);
   %%------------------------ optimize mu---------------------  
    mu = min(mu*pho_mu, max_mu);
end


    
