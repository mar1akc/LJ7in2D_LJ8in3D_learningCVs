#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timestep_helpers.h"
#include "LDA_CoordNum_CV.h"
#include "macros_and_constants.h"

// compile command: gcc -Wall CommittorRC.c LDA_CoordNum_CV.c timestep_helpers.c -lm -O3

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
double sigmoid(double z);
double reaction_coordinate(double *CV,struct NNRC *nnrc);
void readRCdata(char *fname_par,char *fname_dim,struct NNRC *nn);
void Axpy(double *A,double *x,double *y,double *w,int d_in,int d_out);

//-------------------------------------------------------------

void Axpy(double *A,double *x,double *y,double *w,int d_in,int d_out) {
// computes w = A*x + y
	int i,j,ishift;
	
	for( i = 0; i < d_out; i++ ) {
		ishift = i*d_in;
		w[i] = y[i];
		for( j = 0; j < d_in; j++ ) {
			w[i] += A[ishift+j]*x[j];
		}
	}	
}



double ReLU(double z) {
	return max(z,0.0);
}

double sigmoid(double z) {
	return 1.0/(1.0 + exp(-z));
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

	free(w1);
	free(w2);
	free(w3);
	
	return RCval;
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
	
	f0 = fopen(fname_par,"r");
	printf("opened file %s\n",fname_par);
	
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
// 	sprintf(fRCname,"FEMdataBETA%i/Committor_min01_LDA23_BETA%i/RC_NNdata.txt",
// 			(int)round(BETA),(int)round(BETA)); 
// 	char fRCname_dim[100]; 
// 	sprintf(fRCname_dim,"FEMdataBETA%i/Committor_min01_LDA23_BETA%i/RC_dimensions.txt",
// 			(int)round(BETA),(int)round(BETA)); 
// 	struct NNRC *nnrc;	
// 	nnrc = (struct NNRC *)malloc(sizeof(struct NNRC));
// 	readRCdata(fRCname,fRCname_dim,nnrc);
// 
// 	RCval = reaction_coordinate(CV,nnrc); 
// 	
// 	printf("RCval = %.4e\n",RCval);
// 	
// 	return 0;
// }
// 
