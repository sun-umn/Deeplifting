% This checks whether each sensor is connected to at least one other sensor
function [D_indicator,kk,yyy] = sensor_check
load Prob

n = size(PP,2);
DD = sign(dd + dd');

if max(max(DD(1:n-m,n-m+1:end))) == 0 % nothing connected to any anchors
  D_indicator = 0;
  kk = 0;
  yyy = 0;
  return
end

D_index = max(DD(1:n-m,1:n-m));
D_indicator = min(D_index); % this is 1 if all sensors are connected to some sensor, otherwise 0
kk = 0;
yyy = 0;
clear DD

if D_indicator == 0; % zero means it is not connected to other sensors
  DD = dd + dd';
  I = find(D_index == 0);
  YY = DD(I,n-m+1:end);
  if max(max(YY)) == 0 % this means it is completely disconnected, accept
    D_indicator = 1;
  else
    yy = max((YY'));
    [yyy,J] = max(yy);
    kk = length(J);
  end
end