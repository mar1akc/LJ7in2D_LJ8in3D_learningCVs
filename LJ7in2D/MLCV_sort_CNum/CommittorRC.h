
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

double ReLU(double z);
double ReLU_grad(double z);
double sigmoid(double z);
double sigmoid_grad(double z);
double reaction_coordinate(double *CV,struct NNRC *nnrc);
void reaction_coordinate_grad(double *RCval,double *RCgrad,double *CV,struct NNRC *nnrc);
void readRCdata(char *fname_par,char *fname_dim,struct NNRC *nn);
