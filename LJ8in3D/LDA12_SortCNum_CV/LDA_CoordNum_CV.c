#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timestep_helpers.h"

// compile command: gcc -Wall LDA_CoordNum_CV.c timestep_helpers.c -lm -O3
#define mabs(a) ((a) >= 0 ? (a) : -(a))
#define max(a,b) ((a) >= (b) ? (a) : (b))
#define min(a,b) ((a) <= (b) ? (a) : (b))
#define PI 3.141592653589793
#define PI2 6.283185307179586 // 2*PI
#define RSTAR 1.122462048309373 // 2^{1/6}
#define DIM 3
#define SIGMA2 2.25 // 1.5*1.5, the parameter in the coordination numbers


// int main(void);

void quicksort(double *alist,int *isort,int first,int last);
int partition(double *alist,int *isort,int first,int last);

void sort_cnum(double *conf,double *d2,int *isort,int Natoms,int Nd2);
void sort_cnum_Jac(double *conf,double *d2,double *Jac,int *isort,int Natoms,int Nd2);

void rowvec_times_matrix(double *x,double *A,double *xA,int Nrow, int Ncol);
void Atrans_times_J(double *A,double *J, double *AtJ, int d_out, int d_aux, int dim);

void LDA_CNum_CV(double *cvval,double *conf,int Natoms,int CVdim,double *LDAmatrix);
void LDA_CNum_CVgrad(double *CVgrad,double *CVval,double *conf,int Natoms,
				int CVdim,double *LDAmatrix);
																
void readLDAmatrix(char *fname,int Nrow,int Ncol,double *LDAmatrix);

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
	int Na2 = 2*Natoms;
	double aux_x,aux_y,aux_z,aux,r2,r8,r16;
	
	
	// compute coordination numbers
	for( k=0; k<Natoms; k++ ) {
		coord_num[k] = 0.0;
	}
	for( k=1; k<Natoms; k++ ) {	
		for( j=0; j<k; j++ ) {
			aux_x = conf[k] - conf[j];
			aux_y = conf[k+Natoms] - conf[j+Natoms];
			aux_z = conf[k+Na2] - conf[j+Na2];
			r2 = (aux_x*aux_x + aux_y*aux_y + aux_z*aux_z)/SIGMA2;
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
	int Na2 = Natoms*2;
	int dim = DIM*Natoms;
	double aux_x,aux_y,aux_z,aux,r2,r8,r16,iden,fac;
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
			aux_z = conf[k+Na2] - conf[j+Na2];
			r2 = (aux_x*aux_x + aux_y*aux_y + aux_z*aux_z)/SIGMA2; // 2.25 = 1.5*1.5
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
			aux = fac*2.0*aux_z/SIGMA2;
			grad_coord_num[k][k+Na2] += aux;
			grad_coord_num[k][j+Na2] -= aux;
			grad_coord_num[j][k+Na2] += aux;
			grad_coord_num[j][j+Na2] -= aux;
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


void rowvec_times_matrix(double *x,double *A,double *xA,int Nrow, int Ncol) {
	int i,j;
	// the matrix is written row-wise
	for( j = 0; j < Ncol; j++ ) {
		xA[j] = 0.0;
		for( i = 0; i < Nrow; i++ ) {
			xA[j] += x[i]*A[i*Ncol + j];
		}
	}
	
}

void Atrans_times_J(double *A,double *J, double *AtJ, int d_out, int d_aux, int dim) {
	// J and A are stored row-wise
	// compute A^\top * J
	// A is d_aux-by-d_out
	// J is d_aux-by-dim
	int i,j,k;
	
	for( i = 0; i < d_out; i++ ) {
		for( j = 0; j < dim; j++ ) {
			AtJ[i*dim + j] = 0.0;
			for( k = 0; k < d_aux; k++ ) {
				AtJ[i*dim + j] += A[k*d_out + i]*J[k*dim + j];
			}		
		}
	}


}

//-------------------------------------------------------------


void LDA_CNum_CV(double *cvval,double *conf,int Natoms,int CVdim,double *LDAmatrix){
	// CV = sort(CNum)*LDAmatrix
	int *isort;
	double *cnum;
	
	isort = (int *)malloc(Natoms*sizeof(int));
	cnum = (double *)malloc(Natoms*sizeof(double));
	
	// compute vector of sorted distances squared	
	sort_cnum(conf,cnum,isort,Natoms,Natoms);
	
	rowvec_times_matrix(cnum,LDAmatrix,cvval,Natoms,CVdim);

	free(isort);
	free(cnum);

}



void LDA_CNum_CVgrad(double *CVgrad,double *CVval,double *conf,int Natoms,
				int CVdim,double *LDAmatrix){
	// The Jacobian of the CV is LDAmatrix^\top * J, 
	// where J = d sort(cnum) /d xyz, 8-by-24
	int j,*isort;
	double *cnum,*Jsort;
	int dim = DIM*Natoms;

	
	cnum = (double *)malloc(Natoms*sizeof(double));
	isort = (int *)malloc(Natoms*sizeof(int));
	Jsort = (double *)malloc(Natoms*dim*sizeof(double)); // D(cnum)/Dxyz
	
	for( j = 0; j < Natoms*dim; j++ ) Jsort[j] = 0;
	
	sort_cnum_Jac(conf,cnum,Jsort,isort,Natoms,Natoms); 
	rowvec_times_matrix(cnum,LDAmatrix,CVval,Natoms,CVdim);
	Atrans_times_J(LDAmatrix,Jsort,CVgrad,CVdim,Natoms,dim);
	
	free(isort);
	free(cnum);
	free(Jsort);
	
	
}
//********************************
void readLDAmatrix(char *fname,int Nrow,int Ncol,double *LDAmatrix) {
	
	int i,j;
	FILE *fid;
	
	fid = fopen(fname,"r");
	
	printf("In readLDAmatrix\n");
	// save matrix row-wise

	for( j = 0; j < Nrow; j++ ) {
		for( i = 0; i < Ncol; i++ ) {
			fscanf(fid,"%le",LDAmatrix+j*Ncol+i);
			printf("%.4e\t",LDAmatrix[j*Ncol+i]);
		}
		fscanf(fid,"\n");
		printf("\n");
	}
	fclose(fid);
}

	
	
// //********************************
// int main() {
// 	int n,i;
// 	char fname[] = "Data/LDAbasis.txt";
// 	
// 	double *conf0,*cnum;
// 	int Natoms = 8;
// 	int dim = Natoms*DIM;
// 	int CVdim = 2;
// 	int *jsort;
// 	
// 	printf("In main()\n");
// 
// 	// d2 is a sorted list of interatomic distances
// 	conf0 = (double *)malloc(dim*sizeof(double));
// 	cnum = (double *)malloc(Natoms*sizeof(double));
// 	jsort = (int *)malloc(Natoms*sizeof(int));
// 	
// 	printf("Right before calling init_conf, dim = %i\n",dim);
// 	
// 	init_conf(conf0,0); 
// 	for( n = 0; n < dim; n++ ) printf("iconf[%i] = %.4e\n",n,conf0[n]);
// 	
// 	for( i = 0; i < Natoms; i++ ) {
// 		jsort[i] = i;
// 		printf("i = %i, jsort = %i\n",i,jsort[i]);
// 	}	
// 	
// 	// test sortdist2
// 	sort_cnum(conf0,cnum,jsort,Natoms,Natoms);
// 	for( i = 0; i < Natoms; i++ ) {
// 		printf("cnum[%i] = %.4f,jsort[%i] = %i\n",i,cnum[i],i,jsort[i]);
// 	}
// 	printf("Done testing sortdist2\n");
// 	
// 	// Coord_num of atom jsort[place] occupies place "place" in the sorted vector
// 	
// 	
// 	// test sortdist2_Jac
// 	
// 
// 	printf("Check the Jacobian of sort_cnum by finite difference\n");
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
// // test the Jacobian of the LDA CV by finite difference	
// 
// 	double *LDAmatrix;
// 	
// 	LDAmatrix = (double *)malloc(CVdim*Natoms*sizeof(double));
// 	
// 	readLDAmatrix(fname,Natoms,CVdim,LDAmatrix);
// 
// 	double *CV,*CVgrad;
// 	CV = (double *)malloc(CVdim*sizeof(double));
// 	CVgrad = (double *)malloc(CVdim*dim*sizeof(double));
// 	
// 	LDA_CNum_CV(CV,conf0,Natoms,CVdim,LDAmatrix);
// 	LDA_CNum_CVgrad(CVgrad,CV,conf0,Natoms,CVdim,LDAmatrix);
// 	
// 	printf("CV = [%.4e, %.4e]\n",CV[0],CV[1]);
// 	printf("Check the gradient of CVs by finite difference\n");
// 	
// 	double der1,der2,cv1plus,cv1minus,cv2plus,cv2minus;
// 	
// 	for( n = 0; n < dim; n++ ) {
// 		conf0[n] += h;
// 		LDA_CNum_CV(CV,conf0,Natoms,CVdim,LDAmatrix);
// 		cv1plus = CV[0];
// 		cv2plus = CV[1];
// 		conf0[n] -= 2.0*h;
// 		LDA_CNum_CV(CV,conf0,Natoms,CVdim,LDAmatrix);
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
