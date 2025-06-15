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
