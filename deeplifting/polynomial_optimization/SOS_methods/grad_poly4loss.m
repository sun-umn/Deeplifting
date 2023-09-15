function grad_poly4 = grad_poly4loss(x,dd)

anchors = [-0.45 -0.45 0.45 0.45; 0.45 -0.45 0.45 -0.45];

[~,n] = size(x); [~,npm] = size(dd); m = npm - n;


dds = dd(1:n,1:n); dda = dd(1:n,n+1:n+m);

[Is,Js,~] = find(dds); sizeAs = length(Is);

[Ia,Ja,~] = find(dda); sizeAa = length(Ia);


grad_poly4 = zeros(2,n);

for s = 1:sizeAs
    grad_poly4(:,Is(s)) = grad_poly4(:,Is(s)) + 4 * (norm(x(:,Is(s))-x(:,Js(s)))^2-dds(Is(s),Js(s))^2) * (x(:,Is(s))-x(:,Js(s)));
    grad_poly4(:,Js(s)) = grad_poly4(:,Js(s)) + 4 * (norm(x(:,Is(s))-x(:,Js(s)))^2-dds(Is(s),Js(s))^2) * (x(:,Js(s))-x(:,Is(s)));
end


for a = 1:sizeAa
    grad_poly4(:,Ia(a)) = grad_poly4(:,Ia(a)) + 4 * (norm(x(:,Ia(a))-anchors(:,Ja(a)))^2-dda(Ia(a),Ja(a))^2) * (x(:,Ia(a))-anchors(:,Ja(a)));
end


end