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
