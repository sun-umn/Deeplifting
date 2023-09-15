function radiorandomgen(n,radiorange)

x = zeros(2,n+4);
x(1:2,1:n) = rand(2,n) - .5*ones(2,n);
x(1:2,n+1:n+4) = [-0.45 -0.45 0.45 0.45; 0.45 -0.45 0.45 -0.45];

n = n+4;
m = 4;

dd = sparse(n,n);

% distance between 


end