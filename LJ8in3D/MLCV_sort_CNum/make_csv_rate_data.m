function make_csv_rate_data()
fname = ["LDA12_BETA10","LDA12_BETA15","LDA12_BETA20",...
    "LDA23_BETA10","LDA23_BETA15","LDA23_BETA20",...
    "MLCV_BETA10","MLCV_BETA15","MLCV_BETA20"];

prefix = "brute_force_rate_data_";
ext = ".mat";
suffix = ".csv";

N = length(fname);
for k = 1 : N
    input_name = strcat(fname(k),ext);
    data = load(input_name);
    rhoA = data.rhoA;
    rhoB = data.rhoB;
    kA = data.kA;
    kB = data.kB;
    AB_event_counter = data.AB_event_counter;
    M = [AB_event_counter,rhoA,rhoB,kA,kB];
    output_name = strcat(prefix,fname(k),suffix);
    writematrix(M,output_name);
end
end