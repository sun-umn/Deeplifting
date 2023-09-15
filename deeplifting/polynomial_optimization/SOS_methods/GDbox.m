function [RMSDlist,objlist] = GDbox(PP,m,dd,kk,poly4loss,grad_poly4loss)

tol = 1e-5;

[~,nm] = size(PP);
n = nm - m;

objlist = zeros(1,kk);
RMSDlist = zeros(1,kk);

for kkk = 1:kk
    kkk
    rng(kkk+100)
    PP0 = rand(2,n)-.5*ones(2,n);
    PPc = FOM(PP0,dd,grad_poly4loss,poly4loss,tol);
    objlist(kkk) = poly4loss(PPc,dd);
    PPtrue = PP(:,n);
    RMSDlist(kkk) = sqrt(norm(PPc-PPtrue,'fro')^2/n);
end



end