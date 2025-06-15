void init_conf(double *conf,int conf_idx);
double LJpot(double *conf,int Natoms);
void LJpot_and_grad(double *conf,double *pot,double *grad,int Natoms);
char MALAstep(double *conf0,double *conf1,int Natoms,double dt, 
			double *Vpot0,double *Vpot1,double *Vgrad1,double *w);
void box_mueller(double *w,int nw); // generates a pair of Gaussian random variables N(0,1)
// aligns configuration by solving Wahba's problem
// https://en.wikipedia.org/wiki/Wahba%27s_problem
void align( double *conf0, double *conf1, int Natoms ); 
// collective variables mu2 and mu3
// void GaussBumps_pot_and_grad(double *conf,int Natoms,double *val1,double *val2,
// 	double *CV,double *CVgrad,double *height,
// 	double sig2,int Nbumps,double *pot,double *grad,double *biasing_pot);
void restraining_pot_and_grad(double *conf,double *pot,double *grad,int Natoms);
//----- bicubic interpolation
void Gpot_and_ders_on_grid(int Nbumps,double *bump_CV1,double *bump_CV2,double *height,
	double *grid_CV1, double *grid_CV2,double h1,double h2,
	double *pot,double *der1,double *der2,double *der12,int N1,int N2);
void compute_bicubic_coeff_matrix(double *pot,double *der1,double *der2,double *der12,
	double *Amatr,int ind,int N1,int N2);
double wsum0(double a,double b,double c,double d);	
double wsum1(double a,double b,double c,double d);	
double wsum2(double a,double b,double c,double d);	
double wsum3(double a,double b,double c,double d);	
void evaluate_Gpot_and_ders(double *grid_CV1, double *grid_CV2,double h1,double h2,
	double *Amatr,double cv1,double cv2,double *FEval,double *FEder1,double *FEder2,int N1,int N2);
void test_CVgrad(double *conf0);
