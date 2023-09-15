% This finds the largest connected component containing the set of
% anchors
function [PP,m,dd] = new_component(PP,m,dd)

D = sign(dd + dd'); % incidence matrix
n = size(PP,2);
index_set = n-m+1:n;

% find all sensors connected to anchors
Dtmp = D(:,index_set);
Y = max((Dtmp'));
I = find(Y > 0);
index_tmp = [I index_set];

% clean index_set
index_tmp = sort(index_tmp);
tag = [];
tag(1) = 1;
count = 2;
for i = 2:length(index_tmp)
  if index_tmp(i-1) < index_tmp(i)
    tag(count) = i;
    count = count + 1;
  end
end
index_set_new = index_tmp(tag);

while length(index_set_new) > length(index_set)
  index_set = index_set_new; % update index_set
  
  % find all sensors connected to anchors
  Dtmp = D(:,index_set);
  Y = max((Dtmp'));
  I = find(Y > 0);
  index_tmp = [I index_set];

  % clean index_set
  index_tmp = sort(index_tmp);
  tag = [];
  tag(1) = 1;
  count = 2;
  for i = 2:length(index_tmp)
    if index_tmp(i-1) < index_tmp(i)
      tag(count) = i;
      count = count + 1;
    end
  end
  index_set_new = index_tmp(tag);  
end

PP = PP(:,index_set);
dd = dd(index_set,index_set);

