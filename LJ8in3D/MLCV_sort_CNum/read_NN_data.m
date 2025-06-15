function read_NN_data()
dir = "CV_parameters_eigens_normalized/";
fname_A1 = strcat(dir,'parameter_encoder.0.0.weight_shape_torch.Size([45, 8]).csv'); % 45-by-8
fname_b1 = strcat(dir,'parameter_encoder.0.0.bias_shape_torch.Size([45]).csv'); % 45 -by -1

fname_A2 = strcat(dir,'parameter_encoder.1.0.weight_shape_torch.Size([30, 45]).csv'); % 30-by-45
fname_b2 = strcat(dir,'parameter_encoder.1.0.bias_shape_torch.Size([30]).csv'); % 30-by-1

fname_A3 = strcat(dir,'parameter_encoder.2.0.weight_shape_torch.Size([25, 30]).csv'); % 25-by-30
fname_b3 = strcat(dir,'parameter_encoder.2.0.bias_shape_torch.Size([25]).csv'); % 25-by-1

fname_A4 = strcat(dir,'parameter_latent_layer.weight_shape_torch.Size([2, 25]).csv'); % 2-by-25
fname_b4 = strcat(dir,'parameter_latent_layer.bias_shape_torch.Size([2]).csv'); % 2-by-1

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

fid = fopen("MargotCV_dimensions.txt","w");
fprintf(fid,"%d\t%d\t%d\t%d\t%d\n",dim0,dim1,dim2,dim3,dim4);
fclose(fid);


fid = fopen("MargotCV_NNdata.txt","w");
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


