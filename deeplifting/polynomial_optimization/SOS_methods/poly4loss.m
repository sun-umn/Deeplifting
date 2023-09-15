function poly4 = poly4loss(x,dd)

anchors = [-0.45 -0.45 0.45 0.45; 0.45 -0.45 0.45 -0.45];

[~,n] = size(x); [~,npm] = size(dd); m = npm - n;


dds = dd(1:n,1:n); dda = dd(1:n,n+1:n+m);

[Is,Js,~] = find(dds); sizeAs = length(Is);

[Ia,Ja,~] = find(dda); sizeAa = length(Ia);

poly4 = 0;

for s = 1:sizeAs
    poly4 = poly4 + (norm(x(:,Is(s))-x(:,Js(s)))^2-dds(Is(s),Js(s))^2)^2;
end


for a = 1:sizeAa
    poly4 = poly4 + (norm(x(:,Ia(a))-anchors(:,Ja(a)))^2-dda(Ia(a),Ja(a))^2)^2;
end


end