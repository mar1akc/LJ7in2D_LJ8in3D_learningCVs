#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timestep_helpers.h"
#include "macros_and_constants.h"

// compile command: gcc -Wall MargotColVar_CoordNum.c timestep_helpers.c -lm -O3
#define ELU_PAR 1.0
#define SIGMA2 2.25 // 1.5*1.5, the parameter in the coordination numbers

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
void quicksort(double *alist,int *isort,int first,int last);
int partition(double *alist,int *isort,int first,int last);

void sort_cnum(double *conf,double *d2,int *isort,int Natoms,int Nd2);
void sort_cnum_Jac(double *conf,double *d2,double *Jac,int *isort,int Natoms,int Nd2);

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

void sort_cnum(double *conf,double *coord_num,int *isort,int Natoms,int dim0) {
	int j,k;
	double aux_x,aux_y,aux,r2,r8,r16;
	
	
	// compute coordination numbers
	for( k=0; k<Natoms; k++ ) {
		coord_num[k] = 0.0;
	}
	for( k=1; k<Natoms; k++ ) {	
		for( j=0; j<k; j++ ) {
			aux_x = conf[k] - conf[j];
			aux_y = conf[k+Natoms] - conf[j+Natoms];
			r2 = (aux_x*aux_x + aux_y*aux_y)/SIGMA2;
			r8 = pow(r2,4);
			r16 = r8*r8;
			aux = (1.0 - r8)/(1.0 - r16);
			coord_num[k] += aux;
			coord_num[j] += aux;
		}		
	}
// 	// compute coordination numbers
// 	printf("Before sort\n");
// 	for( k=0; k<Natoms; k++ ) {
// 		printf("coord_num[%i] = %.4e\n",k,coord_num[k]);
// 	}

	for( j = 0; j < Natoms; j++ ) {
		isort[j] = j;
	}	

	quicksort(coord_num,isort,0,dim0-1);
	
// 	printf("In sortdist2\n");
// 	for(j = 0; j<Nd2;j++) {
// 		printf("isort[%i] = %i, d2[%i] = %.4f\n",j,isort[j],j,d2[j]);
// 	}

}

//-------------------------------------------------------------

void sort_cnum_Jac(double *conf,double *coord_num,double *Jac,int *isort,int Natoms,int dim0) {
// Computes the Jacobian matrix of the sorted vector of squared distances 
// w.r.t. the atomic coordinates

	int j,k;
	int dim = DIM*Natoms;
	double aux_x,aux_y,aux,r2,r8,r16,iden,fac;
	double grad_coord_num[Natoms][DIM*Natoms];
	int k1,k2;
	


	// initialization	
	for( k=0; k<Natoms; k++ ) {
		coord_num[k] = 0.0;		
		for( j=0; j<dim; j++ ) {
			grad_coord_num[k][j] = 0.0;	
		}
	}

	// compute coordination numbers
	for( k=1; k<Natoms; k++ ) {	
		for( j=0; j<k; j++ ) {
			aux_x = conf[k] - conf[j];
			aux_y = conf[k+Natoms] - conf[j+Natoms];
			r2 = (aux_x*aux_x + aux_y*aux_y)/SIGMA2; // 2.25 = 1.5*1.5
			r8 = pow(r2,4);
			r16 = r8*r8;
			aux = (1.0 - r8)/(1.0 - r16);
			coord_num[k] += aux;
			coord_num[j] += aux;
			iden = 1.0/(1.0 - r16);
			fac = -4.0*pow(r2,3)*iden + aux*8.0*pow(r2,7)*iden;
			aux = fac*2.0*aux_x/SIGMA2;
			grad_coord_num[k][k] += aux;
			grad_coord_num[k][j] -= aux;
			grad_coord_num[j][k] += aux;
			grad_coord_num[j][j] -= aux;
			aux = fac*2.0*aux_y/SIGMA2;
			grad_coord_num[k][k+Natoms] += aux;
			grad_coord_num[k][j+Natoms] -= aux;
			grad_coord_num[j][k+Natoms] += aux;
			grad_coord_num[j][j+Natoms] -= aux;
		}		
	}	

	for( j = 0; j < Natoms; j++ ) {
		isort[j] = j;
	}	

	
	// sort pairwise distances d2
	quicksort(coord_num,isort,0,dim0-1);
	// compute the Jacobian matrix rearranged into a vector row-wise
	for( j = 0; j < dim0; j++ ) {
// 		printf("sortdist2_Jac: index = %i, isort = %i, atom1 = %i, atom2 = %i\n",j,isort[j],atom_index[0],atom_index[1]);		
		k = j*dim;
		k2 = isort[j];
		for( k1 = 0; k1 < dim; k1++ ) {
			Jac[k + k1] = grad_coord_num[k2][k1];
		}
	}
	
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
	int j,*isort;
	double *d2,*w1,*w2;
	int dim0 = nncv->dim0, dim1 = nncv->dim1, dim2 = nncv->dim2;
	int dim3 = nncv->dim3;
	double *A1 = nncv->A1,*b1 = nncv->b1,*A2 = nncv->A2,*b2 = nncv->b2;
	double *A3 = nncv->A3,*b3 = nncv->b3;
	
	isort = (int *)malloc(dim0*sizeof(int));
	d2 = (double *)malloc(dim0*sizeof(double));
	w1 = (double *)malloc(dim1*sizeof(double));
	w2 = (double *)malloc(dim2*sizeof(double));
	
	// compute vector of sorted distances squared	
	sort_cnum(conf,d2,isort,Natoms,dim0);

// 	for( j = 0; j < dim0; j++ ) printf("d2[%i] = %.4e\n",j,d2[j]);
	
	
	Axpy(A1,d2,b1,w1,dim0,dim1);
	for( j = 0; j < dim1; j++ ) {
		w1[j] = ELU(w1[j]);
	}
	Axpy(A2,w1,b2,w2,dim1,dim2);
	for( j = 0; j < dim2; j++ ) {
		w2[j] = ELU(w2[j]);
	}
	Axpy(A3,w2,b3,cvval,dim2,dim3);
	
// 	for( j = 0; j < dim3; j++ ) printf("cvval[%i] = %.4e\n",j,cvval[j]);
	
	free(isort);
	free(d2);
	free(w1);
	free(w2);
}



void MargotCVgrad(double *grad,double *cvval,double *conf,int Natoms,
				struct NN *nncv){

	int j,*isort;
	int dim = DIM*Natoms;
	int dim0 = nncv->dim0, dim1 = nncv->dim1, dim2 = nncv->dim2;
	int dim3 = nncv->dim3;
	double *A1 = nncv->A1,*b1 = nncv->b1,*A2 = nncv->A2,*b2 = nncv->b2;
	double *A3 = nncv->A3,*b3 = nncv->b3;

	double *D1,*D2,*Jaux1,*Jaux2,*Jsort;
	double *d2,*w1,*w2;
	
	d2 = (double *)malloc(dim0*sizeof(double));
	w1 = (double *)malloc(dim1*sizeof(double));
	w2 = (double *)malloc(dim2*sizeof(double));
	
	D1 = (double *)malloc(dim1*sizeof(double));
	D2 = (double *)malloc(dim2*sizeof(double));
	Jaux1 = (double *)malloc(dim1*dim*sizeof(double));	
	Jaux2 = (double *)malloc(dim2*dim*sizeof(double));	

	isort = (int *)malloc(dim0*sizeof(int));

	Jsort = (double *)malloc(dim0*dim*sizeof(double)); // D(sortdist2)/Dxy
	
	for( j = 0; j < dim0*dim; j++ ) Jsort[j] = 0;
	
	sort_cnum_Jac(conf,d2,Jsort,isort,Natoms,dim0); 
	// d2 = sorted vector of squared distances 
	// Jsortd2 = D(sortdist2)/Dxy, size dim0-by-dim, stored row-wise
	// compute w1 = A1*d2 + b1
	
	Axpy(A1,d2,b1,w1,dim0,dim1);
	// compute D1 = ELU'(w1) and w1 = ELU(w1)
	for( j = 0; j < dim1; j++ ) {
		D1[j] = ELUgrad(w1[j]);
		w1[j] = ELU(w1[j]);
	}
	// compute D(w1)/Dxy = D1*A1*Jsortd2
	matmul(A1,Jsort,Jaux1,dim1,dim0,dim); // A1*Jsortd2, size dim2-by-dim
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
	Axpy(A3,w2,b3,cvval,dim2,dim3);
	matmul(A3,Jaux2,grad,dim3,dim2,dim); // A3*Jaux2, size dim3-by-dim
	
	free(isort);
	free(d2);
	free(w1);
	free(w2);
	free(D1);
	free(D2);
	free(Jaux1);
	free(Jaux2);
	free(Jsort);
	
	
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

// //********************************
// int main() {
// 	int n,i;
// 	char fCVname[] = "LJ7_CV_data/MargotCV_CNum_NNdata.txt";
// 	char fname_dim[] = "LJ7_CV_data/MargotCV_CNum_dimensions.txt";
// 	
// 	double *conf0,*cnum;
// 	int Natoms = NATOMS;
// 	int dim = NATOMS*DIM;
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
// 	init_conf(conf0,3); 
// 	for( n=0; n < dim; n++ ) printf("iconf[%i] = %.4e\n",n,conf0[n]);
// 	
// 	for( i = 0; i < Natoms; i++ ) {
// 		jsort[i] = i;
// 		printf("i = %i, jsort = %i\n",i,jsort[i]);
// 	}	
// 	
// 	// test sort_cnum
// 	sort_cnum(conf0,cnum,jsort,Natoms,Natoms);
// 	for( i = 0; i < Natoms; i++ ) {
// 		printf("cnum[%i] = %.4f,jsort[%i] = %i\n",i,cnum[i],i,jsort[i]);
// 	}
// 	printf("Done testing sortdist2\n");
// 	
// 	// Coord_num of atom jsort[place] occupies place "place" in the sorted vector
// 	
// 	
// 	// test sort_cnum_Jac
// 	
// 
// 	printf("Check the Jacobian of sortdist2 by finite difference\n");
// 	
// 	double h=1.0e-6,*d2plus,*d2minus,*Jsortcnum; 
// 	d2plus = (double *)malloc(Natoms*sizeof(double));
// 	d2minus = (double *)malloc(Natoms*sizeof(double));
// 	Jsortcnum = (double *)malloc(Natoms*dim*sizeof(double));
// 
// 	for( i = 0; i < Natoms; i++ ) {
// 		jsort[i] = i;
// 		printf("i = %i, jsort = %i\n",i,jsort[i]);
// 	}	
// 	
// 	sort_cnum_Jac(conf0,cnum,Jsortcnum,jsort,Natoms,Natoms); 
// 
// 	int *inv_jsort;
// 	inv_jsort = (int *)malloc(Natoms*sizeof(int));
// 	for( i = 0; i < Natoms; i++ ) {
// 		n = 0;
// 		while(jsort[n] != i ) n++;
// 		inv_jsort[i] = n;
// 	}
// 	
// 	for( i = 0; i < Natoms; i++ ) {
// 		printf(" i = %i, jsort[%i] = %i, inv_jsort[%i] = %i, jsort[inv_jsort[%i]] = %i\n",
// 			i,i,jsort[i],i,inv_jsort[i],i,jsort[inv_jsort[i]]);
// 	}
// 	
// 	for( n = 0; n < dim; n++ ) {
// 		conf0[n] += h;
// 		sort_cnum(conf0,d2plus,jsort,Natoms,Natoms);
// 		conf0[n] -= 2.0*h;
// 		sort_cnum(conf0,d2minus,jsort,Natoms,Natoms);
// 		conf0[n] += h;
// 		for( i = 0; i < Natoms; i++ ) {
// 			printf("row %i, col %i: Jsort2 = %.4e, FD = %.4e\n",
// 				i,n,Jsortcnum[i*dim+n],0.5*(d2plus[i]-d2minus[i])/h);
// 		}
// 	}
// 
// 	
// 	
// 
//     // NN for the collective variables
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
// 	CV = (double *)malloc(CVDIM*sizeof(double));
// 	CVgrad = (double *)malloc(CVDIM*dim*sizeof(double));
// 
// 	
// 	MargotCV(CV,conf0,Natoms,nncv);
// 	MargotCVgrad(CVgrad,CV,conf0,Natoms,nncv);
// 	
// 	printf("CV = [%.4e, %.4e]\n",CV[0],CV[1]);
// 	printf("Check the gradient of CVs by finite difference\n");
// 	
// 	double der1,der2,cv1plus,cv1minus,cv2plus,cv2minus;
// 	
// 	for( n = 0; n < dim; n++ ) {
// 		conf0[n] += h;
// 		MargotCV(CV,conf0,Natoms,nncv);
// 		cv1plus = CV[0];
// 		cv2plus = CV[1];
// 		conf0[n] -= 2.0*h;
// 		MargotCV(CV,conf0,Natoms,nncv);
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
	
