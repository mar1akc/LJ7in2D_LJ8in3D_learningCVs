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

double ReLU(double z);
double sigmoid(double z);
double reaction_coordinate(double *CV,struct NNRC *nnrc);
void readRCdata(char *fname_par,char *fname_dim,struct NNRC *nn);
void Axpy(double *A,double *x,double *y,double *w,int d_in,int d_out);
