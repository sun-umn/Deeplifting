clear

addpath SeDuMi_1_3

repeats = 10;




R = [1.2, 1.3, 1.4, 1.5];
N = [50, 100, 150, 200];

RMSDsum_table = zeros(3,length(R)*length(N));

objsum_table = zeros(3,length(R)*length(N));

for i = 1:4

for j = 1:2

[i,j]
RMSDtable = zeros(3,repeats);
objvaltable = zeros(3,repeats);    
for k = 1:repeats

rng(k)


n = N(j);
r = R(i)*sqrt(log(n+4)/(n+4));
p = 0.5;

D_ind = 0;
while D_ind ==0
    radiogen2(n,r,p);
    D_ind = sensor_check;
end

load Prob

[RMSD1,objval1,RMSD2,objval2] = SOSGDbox(PP,m,dd,@FULLSOSDcode,@poly4loss,@grad_poly4loss);


kk = 10;
[RMSDlist,objlist] = GDbox(PP,m,dd,kk,@poly4loss,@grad_poly4loss);
[objval3,minpos] = min(objlist);
RMSD3 = RMSDlist(minpos);

RMSDtable(:,k) = [RMSD1,RMSD2,RMSD3]';
objvaltable(:,k) = [objval1,objval2,objval3]';

end

RMSDsum_table(:,i+(j-1)*length(R)) = mean(RMSDtable,2);

objsum_table(:,i+(j-1)*length(R)) = mean(objvaltable,2);

end
end
