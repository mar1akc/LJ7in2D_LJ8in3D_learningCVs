function read_committorNN_data()
fname_A1 = 'linear1_weight_[10,2].csv'; 
fname_b1 = 'linear1_bias_[10].csv'; 

fname_A2 = 'linear2_weight_[10,10].csv'; 
fname_b2 = 'linear2_bias_[10].csv'; 

fname_A3 = 'linear3_weight_[1,10].csv'; 
fname_b3 = 'linear3_bias_[1].csv'; 


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

fid = fopen("RC_dimensions.txt","w");
fprintf(fid,"%d\t%d\t%d\t%d\n",dim0,dim1,dim2,dim3);
fclose(fid);


fid = fopen("RC_NNdata.txt","w");
print2file(fid,A1)
print2file(fid,A2)
print2file(fid,A3)
print2file(fid,b1)
print2file(fid,b2)
print2file(fid,b3)
fclose(fid);

writematrix(A1,"A1.csv")
writematrix(A2,"A2.csv")
writematrix(A3,"A3.csv")
writematrix(b1,"b1.csv")
writematrix(b2,"b2.csv")
writematrix(b3,"b3.csv")

A = [1.4907376502647303, 1.1156981470306966];
vA = [0.15, 1.0];
rA = [0.6 0.04];

B = [-0.8985453607320371 0.5362685308247025];
vB = [-0.15 1.0];
rB = [0.25 0.05];

fid = fopen("RC_committor_paramsA.txt","w");
fprintf(fid,"%.10e\t%.10e\n",A(1),A(2));
fprintf(fid,"%.10e\t%.10e\n",vA(1),vA(2));
fprintf(fid,"%.10e\t%.10e\n",rA(1),rA(2));
fclose(fid);

fid = fopen("RC_committor_paramsB.txt","w");
fprintf(fid,"%.10e\t%.10e\n",B(1),B(2));
fprintf(fid,"%.10e\t%.10e\n",vB(1),vB(2));
fprintf(fid,"%.10e\t%.10e\n",rB(1),rB(2));
fclose(fid);

end
%%
function print2file(fid,A)
n = size(A,1);
for j = 1:n
    fprintf(fid,"%.8e\t",A(j,:));
    fprintf(fid,"\n");
end
end


