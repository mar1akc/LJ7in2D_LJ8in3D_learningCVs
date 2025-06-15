function read_NN_data()
dir = "LJ7_CV_data/";
sz = 45;
fname_A1 = strcat(dir,'encoder_0_0_weight_[',num2str(sz),',7].csv'); 
fname_b1 = strcat(dir,'encoder_0_0_bias_[',num2str(sz),'].csv'); 

fname_A2 = strcat(dir,'encoder_1_0_weight_[',num2str(sz),',',num2str(sz),'].csv'); 
fname_b2 = strcat(dir,'encoder_1_0_bias_[',num2str(sz),'].csv'); 

fname_A3 = strcat(dir,'latent_layer_weight_[2,',num2str(sz),'].csv'); 
fname_b3 = strcat(dir,'latent_layer_bias_[2].csv'); 

A1 = readmatrix(fname_A1);
A2 = readmatrix(fname_A2);
A3 = readmatrix(fname_A3);
b1 = readmatrix(fname_b1);
b2 = readmatrix(fname_b2);
b3 = readmatrix(fname_b3);

[dim1,dim0] = size(A1);
dim2 = size(A2,1);
dim3 = size(A3,1);
fprintf("dim0 = %d, dim1 = %d, dim2 = %d, dim3 = %d\n",dim0,dim1,dim2,dim3);

fid = fopen("MargotCV_CNum_dimensions.txt","w");
fprintf(fid,"%d\t%d\t%d\t%d\n",dim0,dim1,dim2,dim3);
fclose(fid);


fid = fopen("MargotCV_CNum_NNdata.txt","w");
print2file(fid,A1)
print2file(fid,A2)
print2file(fid,A3)
print2file(fid,b1)
print2file(fid,b2)
print2file(fid,b3)
fclose(fid);

writematrix(A1,strcat(dir,"A1.csv"))
writematrix(A2,strcat(dir,"A2.csv"))
writematrix(A3,strcat(dir,"A3.csv"))
writematrix(b1,strcat(dir,"b1.csv"))
writematrix(b2,strcat(dir,"b2.csv"))
writematrix(b3,strcat(dir,"b3.csv"))

end
%%
function print2file(fid,A)
n = size(A,1);
for j = 1:n
    fprintf(fid,"%.8e\t",A(j,:));
    fprintf(fid,"\n");
end
end


