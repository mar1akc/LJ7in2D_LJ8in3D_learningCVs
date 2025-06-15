#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timestep_helpers.h"
#include "MargotColVar_CoordNum.h"
#include "macros_and_constants.h"

// compile command: gcc -Wall CommittorRC.c MargotColVar.c timestep_helpers.c -lm -O3

struct NNRC {
	int dim0; 
	int dim1;
	int dim2;
	int dim3;
	int dim4;
	double *A1;
	double *b1;
	double *A2;
	double *b2;
	double *A3;
	double *b3;
	double *A4;
	double *b4;
};


// int main(void);

double ReLU(double z);
double ReLU_grad(double z);
double sigmoid(double z);
double sigmoid_grad(double z);
double reaction_coordinate(double *CV,struct NNRC *nnrc);
void reaction_coordinate_grad(double *RCval,double *RCgrad,double *CV,struct NNRC *nnrc);
void readRCdata(char *fname_par,char *fname_dim,struct NNRC *nn);
// void Axpy(double *A,double *x,double *y,double *w,int d_in,int d_out);
// void matmul_diag_times_full(double *D,double *A,int nrowA,int ncolA);
// void matmul(double *A,double *B,double *AB,int nrowA,int nrowB,int ncolB);
//-------------------------------------------------------------
// 
// void Axpy(double *A,double *x,double *y,double *w,int d_in,int d_out) {
// // computes w = A*x + y
// 	int i,j,ishift;
// 	
// 	for( i = 0; i < d_out; i++ ) {
// 		ishift = i*d_in;
// 		w[i] = y[i];
// 		for( j = 0; j < d_in; j++ ) {
// 			w[i] += A[ishift+j]*x[j];
// 		}
// 	}	
// }
// 
// void matmul_diag_times_full(double *D,double *A,int nrowA,int ncolA) {
// // diagonal matrix of size nrowA-by-nrowA is multiplied by A and overwritten on A
// 
// 	int i,j;
// 	
// 	for( i = 0; i < nrowA; i++ ) {		
// 		for( j = 0; j < ncolA; j++ ) {
// 			A[i*ncolA + j] = D[i]*A[i*ncolA + j];		
// 		}
// 	}
// }
// 
// void matmul(double *A,double *B,double *AB,int nrowA,int nrowB,int ncolB) {
// // matrices A and B must be indexed row-wise
// // A is nrowA-by-nrowB
// // B is nrowB-by-ncolB
// // AB is nrowA-by-ncolB
// 	int i,j,k;
// 	
// 	for( i = 0; i < nrowA; i++ ) {		
// 		for( j = 0; j < ncolB; j++ ) {
// 			AB[i*ncolB + j] = 0.0;
// 			for( k = 0; k < nrowB; k++ ) {
// 				AB[i*ncolB + j] += A[i*nrowB + k]*B[k*ncolB + j];			
// 			}
// 		
// 		}
// 	}
// }
// 

double ReLU(double z) {
	return max(z,0.0);
}

double ReLU_grad(double z) {
	if (z > 0) return 1.0;
	else return 0.0;
}

double sigmoid(double z) {
	return 1.0/(1.0 + exp(-z));
}

double sigmoid_grad(double x) {
	double aux = sigmoid(x);
	return aux*(1.0 - aux);	
}

//-------------------------------------------------------------
//-------------------------------------------------------------

double reaction_coordinate(double *CV,struct NNRC *nnrc) {

	int j, dim0 = nnrc->dim0,dim1 = nnrc->dim1,dim2 = nnrc->dim2,dim3 = nnrc->dim3,dim4 = nnrc->dim4;
	double *w1, *w2, *w3, *w4;
	double RCval;
	
	w1 = (double *)malloc(dim1*sizeof(double));
	w2 = (double *)malloc(dim2*sizeof(double));
	w3 = (double *)malloc(dim3*sizeof(double));
	w4 = (double *)malloc(dim4*sizeof(double));
	
	Axpy(nnrc->A1,CV,nnrc->b1,w1,dim0,dim1);
	for( j = 0; j < dim1; j++ ) {
// 		if( PRINT == 'y') printf("w1[%i] = %.4e\t",j,w1[j]);
		w1[j] = ReLU(w1[j]);
// 		if( PRINT == 'y') printf("tanh(w1[%i]) = %.4e\n",j,w1[j]);
	}
	Axpy(nnrc->A2,w1,nnrc->b2,w2,dim1,dim2);
	for( j = 0; j < dim2; j++ ) {
// 		if( PRINT == 'y') printf("w2[%i] = %.4e\t",j,w2[j]);
		w2[j] = ReLU(w2[j]);
// 		if( PRINT == 'y') printf("tanh(w2[%i]) = %.4e\n",j,w2[j]);
	}
	Axpy(nnrc->A3,w2,nnrc->b3,w3,dim2,dim3);
	for( j = 0; j < dim3; j++ ) {
// 		if( PRINT == 'y') printf("w2[%i] = %.4e\t",j,w2[j]);
		w3[j] = ReLU(w3[j]);
// 		if( PRINT == 'y') printf("tanh(w2[%i]) = %.4e\n",j,w2[j]);
	}

	Axpy(nnrc->A4,w3,nnrc->b4,w4,dim3,dim4);
	RCval = sigmoid(*w4);

	free(w1);
	free(w2);
	free(w3);
	free(w4);
	
	// By default, RC is the committor from min 1 to min 0, i.e. STRTING_MIN = 1
// 	printf("Starting min = %i, finish min = %i, RCval = %.4e\n",STARTING_MIN,FINISH_MIN,RCval);
	if( STARTING_MIN == 3 ) RCval = 1.0 - RCval;
// 	printf("Starting min = %i, finish min = %i, RCval = %.4e\n",STARTING_MIN,FINISH_MIN,RCval);
	
	return RCval;
}

//-------------------------------------------------------------

void reaction_coordinate_grad(double *RCval,double *RCgrad,double *CV,struct NNRC *nnrc) {

	int j, dim0 = nnrc->dim0,dim1 = nnrc->dim1,dim2 = nnrc->dim2,dim3 = nnrc->dim3,dim4 = nnrc->dim4;
	double *w1,*w2,*w3,*w4,*D1,*D2,*D3,*Jaux1,*Jaux2,*Jaux3,*Jaux4;
	double D4;
	int dim01 = dim0*dim1;
	
	w1 = (double *)malloc(dim1*sizeof(double));
	w2 = (double *)malloc(dim2*sizeof(double));
	w3 = (double *)malloc(dim3*sizeof(double));
	w4 = (double *)malloc(dim4*sizeof(double));
	D1 = (double *)malloc(dim1*sizeof(double));
	D2 = (double *)malloc(dim2*sizeof(double));
	D3 = (double *)malloc(dim3*sizeof(double));
	Jaux1 = (double *)malloc(dim1*dim0*sizeof(double));	
	Jaux2 = (double *)malloc(dim2*dim0*sizeof(double));	
	Jaux3 = (double *)malloc(dim3*dim0*sizeof(double));	
	Jaux4 = (double *)malloc(dim4*dim0*sizeof(double));	


	Axpy(nnrc->A1,CV,nnrc->b1,w1,dim0,dim1); // w1 = A1*CV + b1
	for( j = 0; j < dim1; j++ ) {
		D1[j] = ReLU_grad(w1[j]); // D1 = diag(dReLU(w1))
		w1[j] = ReLU(w1[j]);   // w1 = ReLU(w1)
	}
	// compute D(w1)/Dcv = D1*A1
	for( j = 0; j<dim01; j++ ) Jaux1[j] = nnrc->A1[j];
	matmul_diag_times_full(D1,Jaux1,dim1,dim0); // J1 = D1*A1
	
	Axpy(nnrc->A2,w1,nnrc->b2,w2,dim1,dim2); // w2 = A2*w1 + b2
	for( j = 0; j < dim2; j++ ) {
		D2[j] = ReLU_grad(w2[j]); // D2 = diag(dReLU(w2))
		w2[j] = ReLU(w2[j]);  // w2 = ReLU(w2)
	}
	// compute D(w2)/Dcv = D2*A2*Jaux1
	matmul(nnrc->A2,Jaux1,Jaux2,dim2,dim1,dim0); // J2 = A2*J1
	matmul_diag_times_full(D2,Jaux2,dim2,dim0);  // J2 = D2*J2

	Axpy(nnrc->A3,w2,nnrc->b3,w3,dim2,dim3); // w3 = A3*w2 + b3
	for( j = 0; j < dim3; j++ ) {
		D3[j] = ReLU_grad(w3[j]); // D3 = diag(dReLU(w3))
		w3[j] = ReLU(w3[j]);  // w3 = ReLU(w3)
	}
	// compute D(w3)/Dcv = D3*A3*Jaux2
	matmul(nnrc->A3,Jaux2,Jaux3,dim3,dim2,dim0); // J3 = A3*J2
	matmul_diag_times_full(D3,Jaux3,dim3,dim0);  // J2 = D2*J2

	matmul(nnrc->A4,Jaux3,Jaux4,dim4,dim3,dim0); // J4 = A4*J3, size dim4-by-dim0
	Axpy(nnrc->A4,w3,nnrc->b4,w4,dim3,dim4); // w4 = A4*w3 + b4
	*RCval = sigmoid(*w4); // RC = sigmoid(w3)
	D4 = sigmoid_grad(*w4); 
	for( j = 0; j < dim0; j++ ) {
		RCgrad[j] = D4*Jaux4[j];   // RC_grad = D4*J4
	}

	free(w1);
	free(w2);
	free(w3);
	free(w4);
	free(D1);
	free(D2);
	free(D3);
	free(Jaux1);
	free(Jaux2);
	free(Jaux3);
	free(Jaux4);
	
}


//********************************
void readRCdata(char *fname_par,char *fname_dim,struct NNRC *nn) {
	
	int i,j;
	FILE *f0, *f_dim;
	int dim0, dim1, dim2, dim3, dim4;
	
	printf("In readRCdata\n");
	f_dim = fopen(fname_dim,"r");
	
	fscanf(f_dim,"%i\t",&dim0);
	fscanf(f_dim,"%i\t",&dim1);
	fscanf(f_dim,"%i\t",&dim2);
	fscanf(f_dim,"%i\t",&dim3);
	fscanf(f_dim,"%i\t",&dim4);
	fclose(f_dim);
	printf("NNRC dimensions: %i -> %i -> %i -> %i -> %i\n",dim0,dim1,dim2,dim3,dim4);
	
	f0 = fopen(fname_par,"r");
	printf("opened file %s\n",fname_par);
	
	double *A1, *A2, *A3, *A4, *b1, *b2, *b3, *b4;

	A1 = (double *)malloc(dim0*dim1*sizeof(double));
	A2 = (double *)malloc(dim1*dim2*sizeof(double));
	A3 = (double *)malloc(dim2*dim3*sizeof(double));
	A4 = (double *)malloc(dim3*dim4*sizeof(double));
	b1 = (double *)malloc(dim1*sizeof(double));
	b2 = (double *)malloc(dim2*sizeof(double));
	b3 = (double *)malloc(dim3*sizeof(double));
	b4 = (double *)malloc(dim4*sizeof(double));

	f0 = fopen(fname_par,"r");
	printf("opened file %s\n",fname_par);	
	for( j = 0; j < dim1; j++ ) {
		for( i = 0; i < dim0; i++ ) {
			fscanf(f0,"%le\t",A1+i+j*dim0);
		}
		fscanf(f0,"\n");
	}
	for( j = 0; j < dim2; j++ ) {
		for( i = 0; i < dim1; i++ ) {
			fscanf(f0,"%le\t",A2+i+j*dim1);
		}
		fscanf(f0,"\n");
	}
	for( j = 0; j < dim3; j++ ) {
		for( i = 0; i < dim2; i++ ) {
			fscanf(f0,"%le\t",A3+i+j*dim2);
		}
		fscanf(f0,"\n");

	}
	for( j = 0; j < dim4; j++ ) {
		for( i = 0; i < dim3; i++ ) {
			fscanf(f0,"%le\t",A4+i+j*dim3);
		}
		fscanf(f0,"\n");
	}
	
// 	while(~feof(f0)){
// 	fscanf(f0,"%le",b4);
// 	printf("%.4e\n",*b4);
// 	}
	
	
//   	printf("\n");
	for( j = 0; j < dim1; j++ ) {
		fscanf(f0,"%le",b1+j);
//  		printf("%.4e\t",b1[j]);
	}
	
//  	printf("\n");
	for( j = 0; j < dim2; j++ ) {
		fscanf(f0,"%le",b2+j);
//  		printf("%.4e\t",b2[j]);
	}
	
//  	printf("\n");
	for( j = 0; j < dim3; j++ ) {
		fscanf(f0,"%le",b3+j);
//  		printf("%.4e\t",b3[j]);
	}
	
//  	printf("\n");
	fscanf(f0,"%le",b4);
//  	printf("%.4e\n",*b4);
	
	fclose(f0);
	
	
	nn->dim0 = dim0;
	nn->dim1 = dim1;
	nn->dim2 = dim2;
	nn->dim3 = dim3;
	nn->dim4 = dim4;
	nn->A1 = A1;
	nn->A2 = A2;
	nn->A3 = A3;
	nn->A4 = A4;
	nn->b1 = b1;
	nn->b2 = b2;
	nn->b3 = b3;
	nn->b4 = b4;
	
	printf("Finished reading the NN data for RC\n");
	
}

//-------------------------------------------------------------
// 
// double reaction_coordinate(double *CV,struct NNRC *nnrc) {
// 
// 	int j, dim0 = nnrc->dim0, dim1 = nnrc->dim1, dim2 = nnrc->dim2, dim3 = nnrc->dim3;
// 	double *w1, *w2, *w3;
// 	double RCval;
// 	
// 	w1 = (double *)malloc(dim1*sizeof(double));
// 	w2 = (double *)malloc(dim2*sizeof(double));
// 	w3 = (double *)malloc(dim3*sizeof(double));
// 	
// 	Axpy(nnrc->A1,CV,nnrc->b1,w1,dim0,dim1);
// 	for( j = 0; j < dim1; j++ ) {
// // 		if( PRINT == 'y') printf("w1[%i] = %.4e\t",j,w1[j]);
// 		w1[j] = ReLU(w1[j]);
// // 		if( PRINT == 'y') printf("tanh(w1[%i]) = %.4e\n",j,w1[j]);
// 	}
// 	Axpy(nnrc->A2,w1,nnrc->b2,w2,dim1,dim2);
// 	for( j = 0; j < dim2; j++ ) {
// // 		if( PRINT == 'y') printf("w2[%i] = %.4e\t",j,w2[j]);
// 		w2[j] = ReLU(w2[j]);
// // 		if( PRINT == 'y') printf("tanh(w2[%i]) = %.4e\n",j,w2[j]);
// 	}
// 
// 	Axpy(nnrc->A3,w2,nnrc->b3,w3,dim2,dim3);
// 	RCval = sigmoid(*w3);
// 
// 	free(w1);
// 	free(w2);
// 	free(w3);
// 	
// 	return RCval;
// }
// 
// 
// //-------------------------------------------------------------
// 
// void reaction_coordinate_grad(double *RCval,double *RCgrad,double *CV,struct NNRC *nnrc) {
// 
// 	int j, dim0 = nnrc->dim0,dim1 = nnrc->dim1,dim2 = nnrc->dim2,dim3 = nnrc->dim3;
// 	double *w1,*w2,*w3,*D1,*D2,*Jaux1,*Jaux2,*Jaux3;
// 	double D3;
// 	int dim01 = dim0*dim1;
// 	
// 	w1 = (double *)malloc(dim1*sizeof(double));
// 	w2 = (double *)malloc(dim2*sizeof(double));
// 	w3 = (double *)malloc(dim3*sizeof(double));
// 	D1 = (double *)malloc(dim1*sizeof(double));
// 	D2 = (double *)malloc(dim2*sizeof(double));
// 	Jaux1 = (double *)malloc(dim1*dim0*sizeof(double));	
// 	Jaux2 = (double *)malloc(dim2*dim0*sizeof(double));	
// 	Jaux3 = (double *)malloc(dim3*dim0*sizeof(double));	
// 
// 
// 	Axpy(nnrc->A1,CV,nnrc->b1,w1,dim0,dim1); // w1 = A1*CV + b1
// 	for( j = 0; j < dim1; j++ ) {
// 		D1[j] = ReLU_grad(w1[j]); // D1 = diag(dReLU(w1))
// 		w1[j] = ReLU(w1[j]);   // w1 = ReLU(w1)
// 	}
// 	// compute D(w1)/Dcv = D1*A1
// 	for( j = 0; j<dim01; j++ ) Jaux1[j] = nnrc->A1[j];
// 	matmul_diag_times_full(D1,Jaux1,dim1,dim0); // J1 = D1*A1
// 	
// 	Axpy(nnrc->A2,w1,nnrc->b2,w2,dim1,dim2); // w2 = A2*w2 + b2
// 	for( j = 0; j < dim2; j++ ) {
// 		D2[j] = ReLU_grad(w2[j]); // D2 = diag(dReLU(w2))
// 		w2[j] = ReLU(w2[j]);  // w2 = ReLU(w2)
// 	}
// 	// compute D(w2)/Dcv = D2*A2*Jaux1
// 	matmul(nnrc->A2,Jaux1,Jaux2,dim2,dim1,dim0); // J2 = A2*J1
// 	matmul_diag_times_full(D2,Jaux2,dim2,dim0);  // J2 = D2*J2
// 
// 	matmul(nnrc->A3,Jaux2,Jaux3,dim3,dim2,dim0); // J3 = A3*J2, size dim3-by-dim
// 	Axpy(nnrc->A3,w2,nnrc->b3,w3,dim2,dim3); // w3 = A3*w2 + b3
// 	*RCval = sigmoid(*w3); // RC = sigmoid(w3)
// 	D3 = sigmoid_grad(*w3); // D2 = diag(dReLU(w3))
// 	for( j = 0; j < dim0; j++ ) {
// 		RCgrad[j] = D3*Jaux3[j];   // RC_grad = D3*J3
// 	}
// 
// 	free(w1);
// 	free(w2);
// 	free(w3);
// 	free(D1);
// 	free(D2);
// 	free(Jaux1);
// 	free(Jaux2);
// 	free(Jaux3);
// 	
// }
// 
// //********************************
// void readRCdata(char *fname_par,char *fname_dim,struct NNRC *nn) {
// 	
// 	int i,j;
// 	FILE *f0, *f_dim;
// 	int dim0, dim1, dim2, dim3;
// 	
// 	printf("In readRCdata\n");
// 	f_dim = fopen(fname_dim,"r");
// 	
// 	fscanf(f_dim,"%i\t",&dim0);
// 	fscanf(f_dim,"%i\t",&dim1);
// 	fscanf(f_dim,"%i\t",&dim2);
// 	fscanf(f_dim,"%i\t",&dim3);
// 	fclose(f_dim);
// 	printf("NNRC dimensions: %i -> %i -> %i -> %i\n",dim0,dim1,dim2,dim3);
// 	
// 	f0 = fopen(fname_par,"r");
// 	printf("opened file %s\n",fname_par);
// 	
// 	double *A1, *A2, *A3, *b1, *b2, *b3;
// 
// 	A1 = (double *)malloc(dim0*dim1*sizeof(double));
// 	A2 = (double *)malloc(dim1*dim2*sizeof(double));
// 	A3 = (double *)malloc(dim2*dim3*sizeof(double));
// 	b1 = (double *)malloc(dim1*sizeof(double));
// 	b2 = (double *)malloc(dim2*sizeof(double));
// 	b3 = (double *)malloc(dim3*sizeof(double));
// 
// 	f0 = fopen(fname_par,"r");
// 	printf("opened file %s\n",fname_par);	
// 	for( j = 0; j < dim1; j++ ) {
// 		for( i = 0; i < dim0; i++ ) {
// 			fscanf(f0,"%le\t",A1+i+j*dim0);
// 			fscanf(f0,"\n");
// 		}
// 	}
// 	for( j = 0; j < dim2; j++ ) {
// 		for( i = 0; i < dim1; i++ ) {
// 			fscanf(f0,"%le\t",A2+i+j*dim1);
// 			fscanf(f0,"\n");
// 		}
// 	}
// 	for( j = 0; j < dim3; j++ ) {
// 		for( i = 0; i < dim2; i++ ) {
// 			fscanf(f0,"%le\t",A3+i+j*dim2);
// 			fscanf(f0,"\n");
// 		}
// 	}
// // 	printf("\n");
// 	for( j = 0; j < dim1; j++ ) {
// 		fscanf(f0,"%le\t",b1+j);
// // 		printf("%.4e\t",b1[j]);
// 	}
// 	fscanf(f0,"\n");
// // 	printf("\n");
// 	for( j = 0; j < dim2; j++ ) {
// 		fscanf(f0,"%le\t",b2+j);
// // 		printf("%.4e\t",b2[j]);
// 	}
// 	fscanf(f0,"\n");
// // 	printf("\n");
// 	for( j = 0; j < dim3; j++ ) {
// 		fscanf(f0,"%le\t",b3+j);
// // 		printf("%.4e\t",b3[j]);
// 	}
// 	fscanf(f0,"\n");
// // 	printf("\n");
// 	fclose(f0);
// 	
// 	
// 	nn->dim0 = dim0;
// 	nn->dim1 = dim1;
// 	nn->dim2 = dim2;
// 	nn->dim3 = dim3;
// 	nn->A1 = A1;
// 	nn->A2 = A2;
// 	nn->A3 = A3;
// 	nn->b1 = b1;
// 	nn->b2 = b2;
// 	nn->b3 = b3;
// }
// 
// 	
// 	
// // //********************************
// int main() {
// 	int n,i;
// 	char fCVname[] = "Data/LDAbasis.txt";
// 	double *LDAmatrix;
// 	int CVdim = CVDIM;
// 	double *conf0,*cnum;
// 	int Natoms = NATOMS;
// 	int dim = Natoms*DIM;
// 	int *jsort;
// 	
// 	printf("In main()\n");
// 	
// 	conf0 = (double *)malloc(dim*sizeof(double));
// 	cnum = (double *)malloc(Natoms*sizeof(double));
// 	jsort = (int *)malloc(Natoms*sizeof(int));
// 	
// 	printf("Right before calling init_conf, dim = %i\n",dim);
// 	
// 	init_conf(conf0,0); 
// 	for( n=0; n < dim; n++ ) printf("iconf[%i] = %.4e\n",n,conf0[n]);
// 	
// 	sort_cnum(conf0,cnum,jsort,Natoms,Natoms);
// 	for( i = 0; i < Natoms; i++ ) {
// 		printf("cnum[%i] = %.4f,jsort[%i] = %i\n",i,cnum[i],i,jsort[i]);
// 	}
// 	printf("Done testing sort_cnum\n");
// 	
// 
// 	LDAmatrix = (double *)malloc(CVdim*Natoms*sizeof(double));
// 	readLDAmatrix(fCVname,Natoms,CVdim,LDAmatrix);
// 	
// 
// 	
// 	// Compute Margot's CV
// 	double *CV;
// 	CV = (double *)malloc(CVdim*sizeof(double));
// 
// 	LDA_CNum_CV(CV,conf0,Natoms,CVdim,LDAmatrix);
// 		
// 	printf("CV = [%.4e, %.4e]\n",CV[0],CV[1]);
// 	
// 	double RCval;
// 	    // NN for the reaction coordinate
// 	char fRCname[100];
// 	sprintf(fRCname,"FEMdataBETA%i/Committor_min01_LDA12_BETA%i/RC_NNdata.txt",
// 			(int)round(BETA),(int)round(BETA)); 
// 	char fRCname_dim[100]; 
// 	sprintf(fRCname_dim,"FEMdataBETA%i/Committor_min01_LDA12_BETA%i/RC_dimensions.txt",
// 			(int)round(BETA),(int)round(BETA)); 
// 	struct NNRC *nnrc;	
// 	nnrc = (struct NNRC *)malloc(sizeof(struct NNRC));
// 	readRCdata(fRCname,fRCname_dim,nnrc);
// 
// 	RCval = reaction_coordinate(CV,nnrc); 
// 	
// 	printf("RCval = %.4e\n",RCval);
// 	
// 
// 	// Test RCgrad
// 	double RCplus,RCminus,*RCgrad;
// 	RCgrad = (double *)malloc(2*sizeof(double));
// 	
// 	reaction_coordinate_grad(&RCval,RCgrad,CV,nnrc);
// 	
// 	printf("Check the gradient of RC by finite difference\n");
// 	
// 	double h=1.0e-6;	
// 	double der;
// 	
// 	for( n = 0; n < 2; n++ ) {
// 		CV[n] += h;
// 		RCplus = reaction_coordinate(CV,nnrc);
// 		CV[n] -= 2.0*h; 
// 		RCminus = reaction_coordinate(CV,nnrc);
// 		CV[n] += h;
// 		der = 0.5*(RCplus-RCminus)/h;
// 		printf("CV %i: RCgrad = %.4e, FD_der = %.4e\n",n,RCgrad[n],der);
// 	}
// 
// 	return 0;
// }
// 
