function read_NN_data()
BETA = 20;
dir = sprintf("FEMdataBETA%d/Committor_mu2mu3_BETA%d/",BETA,BETA);

fname_A1 = strcat(dir,'linear1_weight_[25,2].csv');  
fname_b1 = strcat(dir,'linear1_bias_[25].csv');  

fname_A2 = strcat(dir,'linear2_weight_[25,25].csv');  
fname_b2 = strcat(dir,'linear2_bias_[25].csv');  

fname_A3 = strcat(dir,'linear3_weight_[25,25].csv');  
fname_b3 = strcat(dir,'linear3_bias_[25].csv');  

fname_A4 = strcat(dir,'linear4_weight_[1,25].csv'); 
fname_b4 = strcat(dir,'linear4_bias_[1].csv'); 

A1 = readmatrix(fname_A1);
A2 = readmatrix(fname_A2);
A3 = readmatrix(fname_A3);
A4 = readmatrix(fname_A4);
b1 = readmatrix(fname_b1);
b2 = readmatrix(fname_b2);
b3 = readmatrix(fname_b3);
b4 = readmatrix(fname_b4);

[dim1,dim0] = size(A1);
dim2 = size(A2,1);
dim3 = size(A3,1);
dim4 = size(A4,1);
fprintf("dim0 = %d, dim1 = %d, dim2 = %d, dim3 = %d, dim4 = %d\n",dim0,dim1,dim2,dim3,dim4);

fid = fopen(strcat(dir,"RC_dimensions.txt"),"w");
fprintf(fid,"%d\t%d\t%d\t%d\t%d\t%d\n",dim0,dim1,dim2,dim3,dim4);
fclose(fid);


fid = fopen(strcat(dir,"RC_NNdata.txt"),"w");
print2file(fid,A1)
print2file(fid,A2)
print2file(fid,A3)
print2file(fid,A4)
print2file(fid,b1)
print2file(fid,b2)
print2file(fid,b3)
print2file(fid,b4)
fclose(fid);

writematrix(A1,strcat(dir,"A1.csv"))
writematrix(A2,strcat(dir,"A2.csv"))
writematrix(A3,strcat(dir,"A3.csv"))
writematrix(A4,strcat(dir,"A4.csv"))
writematrix(b1,strcat(dir,"b1.csv"))
writematrix(b2,strcat(dir,"b2.csv"))
writematrix(b3,strcat(dir,"b3.csv"))
writematrix(b4,strcat(dir,"b4.csv"))

end
%%
function print2file(fid,A)
n = size(A,1);
for j = 1:n
    fprintf(fid,"%.8e\t",A(j,:));
    fprintf(fid,"\n");
end
end


