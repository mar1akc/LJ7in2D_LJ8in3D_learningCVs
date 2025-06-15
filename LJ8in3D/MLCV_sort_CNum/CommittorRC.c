#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timestep_helpers.h"
#include "MargotColVar_CoordNum.h"
#include "macros_and_constants.h"

// compile command: gcc -Wall CommittorRC.c MargotColVar_CoordNum.c timestep_helpers.c -lm -O3

struct NNRC {
	int dim0; 
	int dim1;
	int dim2;
	int dim3;
	double *A1;
	double *b1;
	double *A2;
	double *b2;
	double *A3;
	double *b3;
};


// int main(void);

double ReLU(double z);
double ReLU_grad(double z);
double sigmoid(double z);
double sigmoid_grad(double x);
double reaction_coordinate(double *CV,struct NNRC *nnrc);
void reaction_coordinate_grad(double *RCval,double *RCgrad,double *CV,struct NNRC *nnrc);
void readRCdata(char *fname_par,char *fname_dim,struct NNRC *nn);
//-------------------------------------------------------------




double ReLU(double z) {
	return max(z,0.0);
}

double sigmoid(double z) {
	return 1.0/(1.0 + exp(-z));
}

double ReLU_grad(double z) {
	if (z > 0) return 1.0;
	else return 0.0;
}

double sigmoid_grad(double z) {
	double aux = sigmoid(z);
	return aux*(1.0 - aux);	
}

//-------------------------------------------------------------

double reaction_coordinate(double *CV,struct NNRC *nnrc) {

	int j, dim0 = nnrc->dim0, dim1 = nnrc->dim1, dim2 = nnrc->dim2, dim3 = nnrc->dim3;
	double *w1, *w2, *w3;
	double RCval;
	
	w1 = (double *)malloc(dim1*sizeof(double));
	w2 = (double *)malloc(dim2*sizeof(double));
	w3 = (double *)malloc(dim3*sizeof(double));
	
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
	RCval = sigmoid(*w3);
	if( STARTING_MIN == 0 ) RCval = 1.0 - RCval;

	free(w1);
	free(w2);
	free(w3);
	
	return RCval;
}

//-------------------------------------------------------------

void reaction_coordinate_grad(double *RCval,double *RCgrad,double *CV,struct NNRC *nnrc) {

	int j, dim0 = nnrc->dim0,dim1 = nnrc->dim1,dim2 = nnrc->dim2,dim3 = nnrc->dim3;
	double *w1,*w2,*w3,*D1,*D2,*Jaux1,*Jaux2,*Jaux3;
	double D3;
	int dim01 = dim0*dim1;
	
	w1 = (double *)malloc(dim1*sizeof(double));
	w2 = (double *)malloc(dim2*sizeof(double));
	w3 = (double *)malloc(dim3*sizeof(double));
	D1 = (double *)malloc(dim1*sizeof(double));
	D2 = (double *)malloc(dim2*sizeof(double));
	Jaux1 = (double *)malloc(dim1*dim0*sizeof(double));	
	Jaux2 = (double *)malloc(dim2*dim0*sizeof(double));	
	Jaux3 = (double *)malloc(dim3*dim0*sizeof(double));	


	Axpy(nnrc->A1,CV,nnrc->b1,w1,dim0,dim1); // w1 = A1*CV + b1
	for( j = 0; j < dim1; j++ ) {
		D1[j] = ReLU_grad(w1[j]); // D1 = diag(dReLU(w1))
		w1[j] = ReLU(w1[j]);   // w1 = ReLU(w1)
	}
	// compute D(w1)/Dcv = D1*A1
	for( j = 0; j<dim01; j++ ) Jaux1[j] = nnrc->A1[j];
	matmul_diag_times_full(D1,Jaux1,dim1,dim0); // J1 = D1*A1
	
	Axpy(nnrc->A2,w1,nnrc->b2,w2,dim1,dim2); // w2 = A2*w2 + b2
	for( j = 0; j < dim2; j++ ) {
		D2[j] = ReLU_grad(w2[j]); // D2 = diag(dReLU(w2))
		w2[j] = ReLU(w2[j]);  // w2 = ReLU(w2)
	}
	// compute D(w2)/Dcv = D2*A2*Jaux1
	matmul(nnrc->A2,Jaux1,Jaux2,dim2,dim1,dim0); // J2 = A2*J1
	matmul_diag_times_full(D2,Jaux2,dim2,dim0);  // J2 = D2*J2

	matmul(nnrc->A3,Jaux2,Jaux3,dim3,dim2,dim0); // J3 = A3*J2, size dim3-by-dim
	Axpy(nnrc->A3,w2,nnrc->b3,w3,dim2,dim3); // w3 = A3*w2 + b3
	*RCval = sigmoid(*w3); // RC = sigmoid(w3)
	D3 = sigmoid_grad(*w3); // D2 = diag(dReLU(w3))
	for( j = 0; j < dim0; j++ ) {
		RCgrad[j] = D3*Jaux3[j];   // RC_grad = D3*J3
	}

	free(w1);
	free(w2);
	free(w3);
	free(D1);
	free(D2);
	free(Jaux1);
	free(Jaux2);
	free(Jaux3);
	
}





//********************************
void readRCdata(char *fname_par,char *fname_dim,struct NNRC *nn) {
	
	int i,j;
	FILE *f0, *f_dim;
	int dim0, dim1, dim2, dim3;
	
	printf("In readRCdata\n");
	f_dim = fopen(fname_dim,"r");
	
	fscanf(f_dim,"%i\t",&dim0);
	fscanf(f_dim,"%i\t",&dim1);
	fscanf(f_dim,"%i\t",&dim2);
	fscanf(f_dim,"%i\t",&dim3);
	fclose(f_dim);
	printf("NNRC dimensions: %i -> %i -> %i -> %i\n",dim0,dim1,dim2,dim3);
		
	double *A1, *A2, *A3, *b1, *b2, *b3;

	A1 = (double *)malloc(dim0*dim1*sizeof(double));
	A2 = (double *)malloc(dim1*dim2*sizeof(double));
	A3 = (double *)malloc(dim2*dim3*sizeof(double));
	b1 = (double *)malloc(dim1*sizeof(double));
	b2 = (double *)malloc(dim2*sizeof(double));
	b3 = (double *)malloc(dim3*sizeof(double));

	f0 = fopen(fname_par,"r");
	printf("opened file %s\n",fname_par);	
	for( j = 0; j < dim1; j++ ) {
		for( i = 0; i < dim0; i++ ) {
			fscanf(f0,"%le\t",A1+i+j*dim0);
			fscanf(f0,"\n");
// 			printf("%.4e\t",A1[i+j*dim0]);
// 			printf("\n");
		}
	}
	for( j = 0; j < dim2; j++ ) {
		for( i = 0; i < dim1; i++ ) {
			fscanf(f0,"%le\t",A2+i+j*dim1);
			fscanf(f0,"\n");
		}
	}
	for( j = 0; j < dim3; j++ ) {
		for( i = 0; i < dim2; i++ ) {
			fscanf(f0,"%le\t",A3+i+j*dim2);
			fscanf(f0,"\n");
		}
	}
// 	printf("\n");
	for( j = 0; j < dim1; j++ ) {
		fscanf(f0,"%le\t",b1+j);
// 		printf("%.4e\t",b1[j]);
	}
	fscanf(f0,"\n");
// 	printf("\n");
	for( j = 0; j < dim2; j++ ) {
		fscanf(f0,"%le\t",b2+j);
// 		printf("%.4e\t",b2[j]);
	}
	fscanf(f0,"\n");
// 	printf("\n");
	for( j = 0; j < dim3; j++ ) {
		fscanf(f0,"%le\t",b3+j);
// 		printf("%.4e\t",b3[j]);
	}
	fscanf(f0,"\n");
// 	printf("\n");
	fclose(f0);
	
	
	nn->dim0 = dim0;
	nn->dim1 = dim1;
	nn->dim2 = dim2;
	nn->dim3 = dim3;
	nn->A1 = A1;
	nn->A2 = A2;
	nn->A3 = A3;
	nn->b1 = b1;
	nn->b2 = b2;
	nn->b3 = b3;
}

	
	
// //********************************
// int main() {
// 	int n,i;
// 	char fCVname[] = "MargotCV_NNdata.txt";
// 	char fname_dim[] = "MargotCV_dimensions.txt";
// 	double *conf0,*cnum;
// 	int Natoms = NATOMS;
// 	int dim = Natoms*DIM;
// 	int *jsort;
// 	
// 	printf("In main()\n");
// 	
// 	
// 	
// 	// d2 is a sorted list of interatomic distances
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
// // 	// Read data files for Margot's CV
// // 	FILE *fcv;
// // 	fcv = fopen("MargotCV_2D/MargotCVdata.txt","r");
// // 	printf("opened file MargotCVdata.txt\n");
// // 	// sizes 21 -- 30 -- 30 -- 1
// // 	// A1 30x21, A2 30x30, A3 1x30, b1 1x30, b2 1x30, b3 1x1
// 	struct NN *nncv;
// 	
// 	nncv = (struct NN *)malloc(sizeof(struct NN));
// 	
// 	readCVdata(fCVname,fname_dim,nncv);
// 	
// 
// 	
// 	// Compute Margot's CV
// 	double *CV,*CVgrad;
// 	CV = (double *)malloc((nncv->dim0)*sizeof(double));
// 	CVgrad = (double *)malloc((nncv->dim0)*dim*sizeof(double));
// 
// // 	CVgrad = (double *)malloc(dim4*dim1*sizeof(double)); // to test
// 
// 	
// 	MargotCV(CV,conf0,Natoms,nncv);
// 	MargotCVgrad(CVgrad,CV,conf0,Natoms,nncv);
// 	
// 	printf("CV = [%.4e, %.4e]\n",CV[0],CV[1]);
// 	
// 	double RCval;
// 	    // NN for the reaction coordinate
// 	char fRCname[] = "Committor_NNretrained_BETA10/RC_NNdata.txt";
// 	char fRCname_dim[] = "Committor_NNretrained_BETA10/RC_dimensions.txt";
// 	char fAname[] = "Committor_NNretrained_BETA10/RC_committor_paramsA.txt";
// 	char fBname[] = "Committor_NNretrained_BETA10/RC_committor_paramsB.txt";
// 	struct NNRC *nnrc;	
// 	nnrc = (struct NNRC *)malloc(sizeof(struct NNRC));
// 	readRCdata(fRCname,fAname,fBname,fRCname_dim,nnrc);
// 
// 	RCval = reaction_coordinate(CV,nnrc); 
// 	
// 	printf("RCval = %.4e\n",RCval);
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
