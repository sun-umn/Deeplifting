%% data generation

N = [100, 200, 300, 400, 500];

r = 0.8;

for i = 5:5

    rng(i)
    n = N(i);
        
    radiorange = r *sqrt(log(n+4)/(n+4));

    D_ind = 0;
    while D_ind ==0
        radiogen(n,radiorange);
        D_ind = sensor_check;
    end
    load Prob
        
    tol = 5e-3;

    [~,RMSD,~,PPsol] = FULLSOSDcode(PP,m,dd,tol);
    sensors = PP(:,1:n);
    distmat = dd;
    sensors_sos = PPsol(:,1:n);

    save data5 n sensors distmat sensors_sos  
end

poly4loss(sensors_sos(:,1:n),dd), RMSD

