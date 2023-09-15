function [RMSD,objval,RMSDref,objvalref] = SOSGDbox(PP,m,dd,FULLSOSDcode,poly4loss,grad_poly4loss)

tol = 5e-3; tolref = 1e-5;

[~,nm] = size(PP); 

n = nm-m;

[~,RMSD,~,PPsol]=FULLSOSDcode(PP,m,dd,tol);

%RMSD, sqrt(norm(PPsol-PP,'fro')^2/n)

PPs = PPsol(:,1:n);

objval = poly4loss(PPs,dd);


PPc = FOM(PPs,dd,grad_poly4loss,poly4loss,tolref);


objvalref = poly4loss(PPc,dd);
RMSDref = sqrt(norm(PPc-PP(:,1:n),'fro')^2/n);


end