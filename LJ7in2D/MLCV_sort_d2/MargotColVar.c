#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timestep_helpers.h"
#include "macros_and_constants.h"

// compile command: gcc -Wall MargotColVar_CoordNum.c timestep_helpers.c -lm -O3
#define ELU_PAR 1.0


struct NN {
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
void atom_indices(int ind,int *atom_index);

void quicksort(double *alist,int *isort,int first,int last);
int partition(double *alist,int *isort,int first,int last);

void sortdist2(double *conf,double *d2,int *isort,int Natoms,int Nd2);
void sortdist2_Jac(double *conf,double *d2,double *Jac,int *isort,int Natoms,int Nd2);

double ELU(double z);
double ELUgrad(double z);

void Axpy(double *A,double *x,double *y,double *w,int d_in,int d_out);
void matmul(double *A,double *B,double *AB,int nrowA,int nrowB,int ncolB);
void matmul_diag_times_full(double *D,double *A,int nrowA,int ncolA);

void MargotCV(double *cvval,double *conf,int Natoms,struct NN *nncv);

void MargotCVgrad(double *grad,double *cvval,double *conf,int Natoms,struct NN *nncv);
												
void readCVdata(char *fname,char *fname_dim,struct NN *nncv);

//-------------------------------------------------------------


//-------------------------------------------------------------

void atom_indices(int ind,int *atom_index) {
	switch(ind) {
		case(0):
			atom_index[0] = 1;
			atom_index[1] = 0;
			break;
		case(1):
			atom_index[0] = 2;
			atom_index[1] = 0;
			break;
		case(2):
			atom_index[0] = 2;
			atom_index[1] = 1;
			break;
		case(3):
			atom_index[0] = 3;
			atom_index[1] = 0;
			break;
		case(4):
			atom_index[0] = 3;
			atom_index[1] = 1;
			break;
		case(5):
			atom_index[0] = 3;
			atom_index[1] = 2;
			break;
		case(6):
			atom_index[0] = 4;
			atom_index[1] = 0;
			break;
		case(7):
			atom_index[0] = 4;
			atom_index[1] = 1;
			break;
		case(8):
			atom_index[0] = 4;
			atom_index[1] = 2;
			break;
		case(9):
			atom_index[0] = 4;
			atom_index[1] = 3;
			break;
		case(10):
			atom_index[0] = 5;
			atom_index[1] = 0;
			break;
		case(11):
			atom_index[0] = 5;
			atom_index[1] = 1;
			break;
		case(12):
			atom_index[0] = 5;
			atom_index[1] = 2;
			break;
		case(13):
			atom_index[0] = 5;
			atom_index[1] = 3;
			break;
		case(14):
			atom_index[0] = 5;
			atom_index[1] = 4;
			break;
		case(15):
			atom_index[0] = 6;
			atom_index[1] = 0;
			break;
		case(16):
			atom_index[0] = 6;
			atom_index[1] = 1;
			break;
		case(17):
			atom_index[0] = 6;
			atom_index[1] = 2;
			break;
		case(18):
			atom_index[0] = 6;
			atom_index[1] = 3;
			break;
 		case(19):
			atom_index[0] = 6;
			atom_index[1] = 4;
			break;
		case(20):
			atom_index[0] = 6;
			atom_index[1] = 5;
			break;
		default:
			printf("ind = %i\n",ind);
			exit(1);
			break;			
	}
}



//******************************
void quicksort(double *alist,int *isort,int first,int last){
   int splitpoint;   
	
   if( first < last ) {

       splitpoint = partition(alist,isort,first,last);

       quicksort(alist,isort,first,splitpoint - 1);
       quicksort(alist,isort,splitpoint + 1,last);
    }
}

//******************************
int partition(double *alist,int *isort,int first,int last) {
	double pivotvalue = alist[first];
	double temp,itemp;

	int leftmark = first + 1;
	int rightmark = last;

	char done = 'n';
	while( done == 'n' ){

		while( leftmark <= rightmark && alist[leftmark] <= pivotvalue) {
			leftmark = leftmark + 1;
		}
		while( alist[rightmark] >= pivotvalue && rightmark >= leftmark) {
			rightmark = rightmark - 1;
		}
		if( rightmark < leftmark ) {
			done = 'y';
		}    
		else {
			temp = alist[leftmark];
			alist[leftmark] = alist[rightmark];
			alist[rightmark] = temp;
		   
			itemp = isort[leftmark];
			isort[leftmark] = isort[rightmark];
			isort[rightmark] = itemp;
		}    
	 }
	temp = alist[first];
	alist[first] = alist[rightmark];
	alist[rightmark] = temp;
   
	itemp = isort[first];
	isort[first] = isort[rightmark];
	isort[rightmark] = itemp;


	return rightmark;
}

//------------------------------------------------------------

void sortdist2(double *conf,double *d2,int *isort,int Natoms,int Nd2) {
	int j,k,count = 0;
	double aux1,aux2;
	
	
	// compute pairwise distances
	for( j = 1; j < Natoms; j++) {
		for( k = 0; k < j; k++ ) {
			aux1 = conf[j] - conf[k];
			aux2 = conf[j+Natoms] - conf[k+Natoms];
			d2[count] = aux1*aux1 + aux2*aux2;
			count++;		
		}
	}	
	quicksort(d2,isort,0,Nd2-1);
	
// 	printf("In sortdist2\n");
// 	for(j = 0; j<Nd2;j++) {
// 		printf("isort[%i] = %i, d2[%i] = %.4f\n",j,isort[j],j,d2[j]);
// 	}

}

//-------------------------------------------------------------

void sortdist2_Jac(double *conf,double *d2,double *Jac,int *isort,int Natoms,int Nd2) {
// Computes the Jacobian matrix of the sorted vector of squared distances 
// w.r.t. the atomic coordinates

	int j,k,count = 0;
	int dim = 2*Natoms;
	double aux1,aux2;
	int *atom_index;
	int k1,k2;
	
	atom_index = (int *)malloc(2*sizeof(int));
	
	// compute pairwise distances
	for( j = 1; j < Natoms; j++) {
		for( k = 0; k < j; k++ ) {
			aux1 = conf[j] - conf[k];
			aux2 = conf[j+Natoms] - conf[k+Natoms];
			d2[count] = aux1*aux1 + aux2*aux2;
			count++;		
		}
	}	
	// sort pairwise distances d2
	quicksort(d2,isort,0,Nd2-1);
	// compute the Jacobian matrix rearranged into a vector row-wise
	for( j = 0; j < Nd2; j++ ) {
		atom_indices(isort[j],atom_index);
// 		printf("sortdist2_Jac: index = %i, isort = %i, atom1 = %i, atom2 = %i\n",j,isort[j],atom_index[0],atom_index[1]);
		k = j*dim;
		k1 = k + atom_index[0];
		k2 = k + atom_index[1];
		Jac[k1] = 2.0*(conf[atom_index[0]] - conf[atom_index[1]]);
		Jac[k2] = -Jac[k1];
		Jac[k1 + Natoms] = 2.0*(conf[atom_index[0]+Natoms] - conf[atom_index[1]+Natoms]);
		Jac[k2 + Natoms] = -Jac[k1+Natoms];
	}
	
	free(atom_index);
	
}


//-------------------------------------------------------------


double ELU(double z) {
	return max(z,0.0) + min(ELU_PAR*(exp(z)-1.0),0.0);
}

double ELUgrad(double z) {
	if( z >= 0.0 ) return 1.0;
	else return ELU_PAR*exp(z);
}

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

void matmul(double *A,double *B,double *AB,int nrowA,int nrowB,int ncolB) {
// matrices A and B must be indexed row-wise
// A is nrowA-by-nrowB
// B is nrowB-by-ncolB
// AB is nrowA-by-ncolB
	int i,j,k;
	
	for( i = 0; i < nrowA; i++ ) {		
		for( j = 0; j < ncolB; j++ ) {
			AB[i*ncolB + j] = 0.0;
			for( k = 0; k < nrowB; k++ ) {
				AB[i*ncolB + j] += A[i*nrowB + k]*B[k*ncolB + j];			
			}
		
		}
	}
}

void matmul_diag_times_full(double *D,double *A,int nrowA,int ncolA) {
// diagonal matrix of size nrowA-by-nrowA is multiplied by A and overwritten on A

	int i,j;
	
	for( i = 0; i < nrowA; i++ ) {		
		for( j = 0; j < ncolA; j++ ) {
			A[i*ncolA + j] = D[i]*A[i*ncolA + j];		
		}
	}
}

//-------------------------------------------------------------


void MargotCV(double *cvval,double *conf,int Natoms,struct NN *nncv){
	int i,j,*isort;
	double *d2,*w1,*w2;

	int dim0 = nncv->dim0, dim1 = nncv->dim1, dim2 = nncv->dim2;
	int dim3 = nncv->dim3;
	double *A1 = nncv->A1,*b1 = nncv->b1,*A2 = nncv->A2,*b2 = nncv->b2;
	double *A3 = nncv->A3,*b3 = nncv->b3;
	
	isort = (int *)malloc(dim0*sizeof(int));
	d2 = (double *)malloc(dim0*sizeof(double));
	w1 = (double *)malloc(dim1*sizeof(double));
	w2 = (double *)malloc(dim2*sizeof(double));
	
	for( i = 0; i < dim0; i++ ) {
		isort[i] = i;
	}	

	
	sortdist2(conf,d2,isort,Natoms,dim0);
	
	
	Axpy(A1,d2,b1,w1,dim0,dim1);
	for( j = 0; j < dim1; j++ ) {
		w1[j] = ELU(w1[j]);
	}
	Axpy(A2,w1,b2,w2,dim1,dim2);
	for( j = 0; j < dim2; j++ ) {
		w2[j] = ELU(w2[j]);
	}

	Axpy(A3,w2,b3,cvval,dim2,dim3);
	
	free(isort);
	free(d2);
	free(w1);
	free(w2);

}



void MargotCVgrad(double *grad,double *cvval,double *conf,int Natoms,struct NN *nncv){

	int i,j,*isort;
	int dim = DIM*Natoms;
	double *D1,*D2,*Jaux1,*Jaux2,*Jsortd2;
	double *d2,*w1,*w2;
	
	int dim0 = nncv->dim0, dim1 = nncv->dim1, dim2 = nncv->dim2;
	int dim3 = nncv->dim3; 
	double *A1 = nncv->A1,*b1 = nncv->b1,*A2 = nncv->A2,*b2 = nncv->b2;
	double *A3 = nncv->A3,*b3 = nncv->b3;

	
	d2 = (double *)malloc(dim0*sizeof(double));
	w1 = (double *)malloc(dim1*sizeof(double));
	w2 = (double *)malloc(dim2*sizeof(double));

	
	D1 = (double *)malloc(dim1*sizeof(double));
	D2 = (double *)malloc(dim2*sizeof(double));
	Jaux1 = (double *)malloc(dim1*dim*sizeof(double));	
	Jaux2 = (double *)malloc(dim2*dim*sizeof(double));	

	isort = (int *)malloc(dim0*sizeof(int));
	for( i = 0; i < dim0; i++ ) {
		isort[i] = i;
	}	

	Jsortd2 = (double *)malloc(dim0*dim*sizeof(double)); // D(sortdist2)/Dxy
	
	for( j = 0; j < dim0*dim; j++ ) Jsortd2[j] = 0;
	
	sortdist2_Jac(conf,d2,Jsortd2,isort,Natoms,dim0); 
	// d2 = sorted vector of squared distances 
	// Jsortd2 = D(sortdist2)/Dxy, size dim1-by-dim, stored row-wise
	// compute w1 = A1*d2 + b1
	Axpy(A1,d2,b1,w1,dim0,dim1);
	// compute D1 = ELU'(w1) and w1 = ELU(w1)
	for( j = 0; j < dim1; j++ ) {
		D1[j] = ELUgrad(w1[j]);
		w1[j] = ELU(w1[j]);
	}
	// compute D(w1)/Dxy = D1*A1*Jsortd2
	matmul(A1,Jsortd2,Jaux1,dim1,dim0,dim); // A1*Jsortd2, size dim2-by-dim
	matmul_diag_times_full(D1,Jaux1,dim1,dim); 
	
	// compute w2 = A2*w1 + b2
	Axpy(A2,w1,b2,w2,dim1,dim2);
	// 	 compute D2 = ELU'(w2) and w2 = ELU(w2)
	for( j = 0; j < dim2; j++ ) {
		D2[j] = ELUgrad(w2[j]);
		w2[j] = ELU(w2[j]);
	}
	// compute D(w2)/Dxy = D2*A2*Jaux1
	matmul(A2,Jaux1,Jaux2,dim2,dim1,dim); // A2*Jaux1, size dim3-by-dim
	matmul_diag_times_full(D2,Jaux2,dim2,dim); 
	
	// w3 = A3*w2 + b3
	// grad = A3*Jaux2, size dim4*dim
	matmul(A3,Jaux2,grad,dim3,dim2,dim); // A2*Jaux1, size dim3-by-dim
	Axpy(A3,w2,b3,cvval,dim2,dim3);
	
	free(isort);
	free(d2);
	free(w1);
	free(w2);
	free(D1);
	free(D2);
	free(Jaux1);
	free(Jaux2);
	free(Jsortd2);
	
	
}

//********************************
void readCVdata(char *fname_par,char *fname_dim,struct NN *nncv) {
	
	int i,j;
	FILE *fcv,*f_dim;
	int dim0, dim1, dim2, dim3; 
	
	printf("In readCVdata\n");
	f_dim = fopen(fname_dim,"r");
	if( f_dim == NULL ) {
		printf("Cannot find the file %s\n",fname_dim);
		exit(1);
	}
	
	fscanf(f_dim,"%i\t",&dim0);
	fscanf(f_dim,"%i\t",&dim1);
	fscanf(f_dim,"%i\t",&dim2);
	fscanf(f_dim,"%i\t",&dim3);
	fclose(f_dim);
	printf("NN dimensions: %i -> %i -> %i -> %i\n",dim0,dim1,dim2,dim3);
	
	fcv = fopen(fname_par,"r");
	
	if( fcv == NULL ) {
		printf("Cannot find the file %s\n",fname_par);
		exit(1);
	}
	printf("opened file %s\n",fname_par);
	
	double *A1, *A2, *A3, *b1, *b2, *b3;
	A1 = (double *)malloc(dim0*dim1*sizeof(double));
	A2 = (double *)malloc(dim1*dim2*sizeof(double));
	A3 = (double *)malloc(dim2*dim3*sizeof(double));
	b1 = (double *)malloc(dim1*sizeof(double));
	b2 = (double *)malloc(dim2*sizeof(double));
	b3 = (double *)malloc(dim3*sizeof(double));
	
	for( j = 0; j < dim1; j++ ) {
		for( i = 0; i < dim0; i++ ) {
			fscanf(fcv,"%le\t",A1+i+j*dim0);
		}
		fscanf(fcv,"\n");
	}
	for( j = 0; j < dim2; j++ ) {
		for( i = 0; i < dim1; i++ ) {
			fscanf(fcv,"%le\t",A2+i+j*dim1);
		}
		fscanf(fcv,"\n");
	}
	for( j = 0; j < dim3; j++ ) {
		for( i = 0; i < dim2; i++ ) {
			fscanf(fcv,"%le\t",A3+i+j*dim2);
		}
		fscanf(fcv,"\n");
	}
// 	printf("\n");
	for( j = 0; j < dim1; j++ ) {
		fscanf(fcv,"%le\t",b1+j);
// 		printf("%.4e\t",b1[j]);
	}
	fscanf(fcv,"\n");
// 	printf("\n");
	for( j = 0; j < dim2; j++ ) {
		fscanf(fcv,"%le\t",b2+j);
// 		printf("%.4e\t",b2[j]);
	}
	fscanf(fcv,"\n");
// 	printf("\n");
	for( j = 0; j < dim3; j++ ) {
		fscanf(fcv,"%le\t",b3+j);
// 		printf("%.4e\t",b3[j]);
	}
	fscanf(fcv,"\n");
// 	printf("\n");
	fclose(fcv);
	
	nncv->dim0 = dim0;
	nncv->dim1 = dim1;
	nncv->dim2 = dim2;
	nncv->dim3 = dim3;
	nncv->A1 = A1;
	nncv->A2 = A2;
	nncv->A3 = A3;
	nncv->b1 = b1;
	nncv->b2 = b2;
	nncv->b3 = b3;
	
}

	
	
	
//********************************
// int main() {
// 	int n = 9,i,j,*isort;
// 	double *alist,*a;
// 	char fCVname[] = "MargotCVdata.txt";
// 	
// 	
// 	double *conf0,*d2;
// 	int Natoms = 7;
// 	int dim = Natoms*2;
// 	int dim1 = Natoms*(Natoms-1)/2;
// 	int *jsort;
// 	// d2 is a sorted list of interatomic distances
// 	conf0 = (double *)malloc(dim*sizeof(double));
// 	d2 = (double *)malloc(dim1*sizeof(double));
// 	jsort = (int *)malloc(dim1*sizeof(int));
// 	init_conf(conf0,Natoms); 
// 	for( i = 0; i < dim1; i++ ) {
// 		jsort[i] = i;
// 	}	
// 	
// 	// test sortdist2
// 	sortdist2(conf0,d2,jsort,Natoms,dim1);
// 	for( i = 0; i < dim1; i++ ) {
// 		printf("d2[%i] = %.4f,jsort[%i] = %i\n",i,d2[i],i,jsort[i]);
// 	}
// 	printf("Done testing sortdist2\n");
// 
// 
// // 	// Read data files for Margot's CV
// // 	FILE *fcv;
// // 	fcv = fopen("MargotCV_2D/MargotCVdata.txt","r");
// // 	printf("opened file MargotCVdata.txt\n");
// // 	// sizes 21 -- 30 -- 30 -- 1
// // 	// A1 30x21, A2 30x30, A3 1x30, b1 1x30, b2 1x30, b3 1x1
// 	double *A1,*A2,*A3,*b1,*b2,*b3;
// 	int dim2 = 30,dim3 = 30,dim4 = 2;
// 	
// 	A1 = (double *)malloc(dim1*dim2*sizeof(double));
// 	A2 = (double *)malloc(dim2*dim3*sizeof(double));
// 	A3 = (double *)malloc(dim3*dim4*sizeof(double));
// 	b1 = (double *)malloc(dim2*sizeof(double));
// 	b2 = (double *)malloc(dim3*sizeof(double));
// 	b3 = (double *)malloc(dim4*sizeof(double));
// 	
// 	readCVdata(fCVname,dim1,dim2,dim3,dim4,A1,A2,A3,b1,b2,b3);
// 	
// 
// 	
// 	// Compute Margot's CV
// 	double *CV,*CVgrad;
// 	CV = (double *)malloc(dim4*sizeof(double));
// 	CVgrad = (double *)malloc(dim4*dim*sizeof(double));
// 
// // 	CVgrad = (double *)malloc(dim4*dim1*sizeof(double)); // to test
// 
// 	
// 	MargotCV(CV,conf0,Natoms,A1,A2,A3,b1,b2,b3,dim1,dim2,dim3,dim4);
// 	MargotCVgrad(CVgrad,CV,conf0,Natoms,A1,A2,A3,b1,b2,b3,dim1,dim2,dim3,dim4);
// 	
// 	printf("CV = [%.4e, %.4e]\n",CV[0],CV[1]);
// 	printf("Check the gradient of CVs by finite difference\n");
// 	
// 	double h=1.0e-6,der1,der2,cv1plus,cv1minus,cv2plus,cv2minus;
// 	
// 	for( n = 0; n < dim; n++ ) {
// 		conf0[n] += h;
// 		MargotCV(CV,conf0,Natoms,A1,A2,A3,b1,b2,b3,dim1,dim2,dim3,dim4);
// 		cv1plus = CV[0];
// 		cv2plus = CV[1];
// 		conf0[n] -= 2.0*h;
// 		MargotCV(CV,conf0,Natoms,A1,A2,A3,b1,b2,b3,dim1,dim2,dim3,dim4);
// 		cv1minus = CV[0];
// 		cv2minus = CV[1];
// 		conf0[n] += h;
// 		der1 = 0.5*(cv1plus-cv1minus)/h;
// 		der2 = 0.5*(cv2plus-cv2minus)/h;	
// 		printf("Dim %i: CV1grad = %.4e, CV2grad = %.4e, der1 = %.4e, der2 = %.4e\n",n,CVgrad[n],CVgrad[n+dim],der1,der2);
// 	}
// 
// 	return 0;
// }
// 
