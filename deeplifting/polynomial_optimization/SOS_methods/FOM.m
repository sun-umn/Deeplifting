function PPc = FOM(PP0,dd,grad,fun,e)

eta = 0.5; sgm = 0.01; M = 5; a0 = 1;
amax = 1e2; amin = 1e-2;
fval_lastM = ones(M+1,1) * fun(PP0,dd);
PPc = PP0; ac = a0;

N = 20000;



for k = 1:N
    Gc = grad(PPc,dd);
    PPn = PPc - ac* Gc;
    while fun(PPn,dd) >= max(fval_lastM) - sgm * ac * norm(Gc,'fro')^2/2
        ac = ac * eta;
        PPn = PPc - ac * Gc;
        if abs(ac) < 1e-5
            break
        end
    end
Gn = grad(PPn,dd);

% BB stepsize
DS = PPn - PPc; DY = Gn - Gc;
Ds = reshape(DS,[],1); Dy = reshape(DY,[],1);

ac = Ds'*Ds/(Ds'*Dy); ac = min(amax,max(ac,amin));


% termination criterion
if norm(Gc,'fro')<=e 
    break
end

% update current iterate and last M objective values
PPc = PPn; fval_lastM(2:M+1) = fval_lastM(1:M); fval_lastM(1) = fun(PPc,dd);


end
PPc = PPn;
k
end