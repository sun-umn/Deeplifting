function radiogen(n,radiorange)

x=zeros(2,n+4);
x(1:2,1:n)=rand(2,n)-.5*ones(2,n);
x(1:2,n+1:n+4)=[-0.45 -0.45 0.45 0.45;0.45 -0.45 0.45 -0.45];
n=n+4;
m=4;

dd=sparse(n,n);

for i=n-m+1:n
  for j=1:n-m
    dist=norm(x(:,i)-x(:,j));
    if dist<radiorange
      dd(j,i)=dist;
    else
      dd(j,i)=0;
    end
  end
end

for i=1:n-m
  for j=i+1:n-m
    dist=norm(x(:,i)-x(:,j));
    if dist<radiorange
      dd(i,j)=dist;
    else
      dd(i,j)=0;
    end
  end
end
PP=x;
save Prob m PP dd  