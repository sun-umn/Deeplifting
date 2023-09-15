function randomgen(n,r)

x=zeros(2,n+4);
x(1:2,1:n)=rand(2,n)-.5*ones(2,n);
x(1:2,n+1:n+4)=[-0.45 -0.45 0.45 0.45;0.45 -0.45 0.45 -0.45];
dim=2;
m=4;
n=n+4;

% save PP
PP=x;

% create distance matrix
dd=sparse(n,n);
% sensor anchor connections
for i=n-m+1:n
  for j=1:n-m
    s=rand(1);
    if s<=r
      dd(j,i)=norm(x(:,i)-x(:,j));
    end
  end
end
% sensor sensor connections
for i=1:n-m
  for j=i+1:n-m
    s=rand(1);
    if s<=r
      dd(i,j)=norm(x(:,i)-x(:,j));
    end
  end
end

save Prob PP m dd