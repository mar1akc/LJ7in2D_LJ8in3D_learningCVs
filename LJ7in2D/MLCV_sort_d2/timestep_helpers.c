#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "macros_and_constants.h"
#include "MargotColVar.h"

struct vec2 {
	double x;
	double y;
};

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
// struct vec2 CVs(double *conf,int Natoms);
// void CVgrad(double *conf,double *mu2,double *mu3,
// 			double *mu2grad,double *mu3grad,int Natoms);
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
//-------------------------------------------------------------

void init_conf(double *conf,int conf_idx) {
	// configurations must be enumerated starting from zero
	FILE *fid;
	char fname[] = "LJ7in2Dmin/LJ7in2Dminima.txt";
	fid = fopen(fname,"r");
	if( fid == NULL ) {
		printf("Cannot find file LJ7in2Dmin/LJ7in2Dminima.txt\n");
		exit(1);
	}
	int k,j,dim = NATOMS*DIM;
	double aux,amp = 1.0e-2;
	for( k = 0; k < DIM*conf_idx; k++ ) {
		for( j = 0; j < NATOMS; j++ ) {
			fscanf(fid,"%le\t",&aux);
		}
		fscanf(fid,"\n");
	}
	for( k = 0; k < DIM; k++ ) {
		for( j = 0; j < NATOMS; j++ ) {
			fscanf(fid,"%le\t",&aux);
			conf[j + k*NATOMS] = aux;
		}
		fscanf(fid,"\n");
	}
	fclose(fid);
	for( j = 0; j < dim; j++ ) {
	    conf[j] += amp*(((double)random()) / RAND_MAX);
	}
}

void box_mueller(double *w,int nw){
	// generates nw Gaussian random variables
	double x1, y1, p, q;
	int i, j;
	j = (nw%2 == 0) ? nw/2 : nw/2 + 1;
	for( i = 0; i < j; i++ ) {
		do{
			p=random();
			x1 = p/RAND_MAX;
			p=random();
			y1 = p/RAND_MAX;
		}
		while( x1 == 0.0 );
		/* Box-Muller transform */
		p=PI2*y1;
		q=2.0*log(x1);
		w[i]=cos(p)*sqrt(-q);
		if( i + j < nw ) w[i + j]=sin(p)*sqrt(-q);
	}	
}

//------------------------------------------------------------

double LJpot(double *conf,int Natoms) {
	double dist_squared,rm6,dx,dy,pot = 0.0;
	int j,k;
	// pot = 4*sum_{j < k}(r_{jk}^{-12} - r_{jk}^{-6})
	for( k = 1; k < Natoms; k++ ) {
		for( j = 0; j < k; j++ ) {
			dx = conf[k] - conf[j];
			dy = conf[k+Natoms] - conf[j+Natoms];
			dist_squared = dx*dx + dy*dy;
			rm6 = 1.0/(dist_squared*dist_squared*dist_squared);
			pot += rm6*(rm6 - 1.0);
		}
	}
	pot *= 4.0;
	return pot;
}

void LJpot_and_grad(double *conf,double *pot,double *grad,int Natoms) {
	double aux,rm6,rm8,rm14,dx,dy,dist_squared;
	int j,k;
	// grad[k] = 4*sum_{j \neq k}(-12*r_{jk}^{-14} + r_{jk}^{-8})*(conf[j]-conf[k])
	for( k = 0; k < Natoms; k++ ) {
		grad[k] = 0.0;
		grad[k+Natoms] = 0.0;
	}
	*pot = 0.0;
	for( k = 1; k < Natoms; k++ ) {
		for( j = 0; j < k; j++ ) {
			dx = conf[k] - conf[j];
			dy = conf[k+Natoms] - conf[j+Natoms];
			dist_squared = dx*dx + dy*dy;
			rm6 = 1.0/(dist_squared*dist_squared*dist_squared);
			*pot += rm6*(rm6 - 1.0);
			rm8 = rm6/dist_squared;
			rm14 = rm6*rm8;
			aux = (-12.0*rm14 + 6.0*rm8)*dx;
			grad[k] += aux;
			grad[j] -= aux;
			aux = (-12.0*rm14 + 6.0*rm8)*dy;
			grad[k+Natoms] += aux;
			grad[j+Natoms] -= aux;			
		}
	}
	*pot *= 4.0;
	for( k = 0; k < Natoms; k++ ) {
		grad[k] *= 4.0;
		grad[k+Natoms] *= 4.0;
}
}
//------------------------------------------------------------
char MALAstep(double *conf0,double *conf1,int Natoms,double dt, 
			double *Vpot0,double *Vpot1,double *Vgrad1,double *w){
	int k;
	double aux,Q01 = 0.0,Q10 = 0.0; // transition probabilities between configurations 0 and 1
	double alpha,eta;
	char ch;
	int dim = DIM*Natoms;
	// evaluate the transition probabilities Q01 and Q10
	Q01 = 0.0;
	Q10 = 0.0;
	for( k = 0; k < dim; k++ ) {
		Q01 += w[k]*w[k];
		aux = conf0[k]-conf1[k] + dt*Vgrad1[k];
		Q10 += aux*aux;
	}
	alpha = exp(-BETA*((*Vpot1) - (*Vpot0) +(Q10-Q01)*0.25/dt));
	if( alpha >= 1.0 ) { // accept move
		ch = 1;		
	}
	else { // accept move with probability alpha
		eta = (double)random();
		eta /= RAND_MAX; // uniform random variable on (0,1)
		ch = ( eta < alpha ) ? 1 : 0; 
	}
	return ch;	
}

//------------------------------------------------------------

void align( double *conf0, double *conf1, int Natoms ) {
	// shift the center of mass to the origin
	double xc = 0.0, yc = 0.0;
	int k;
	
	// center conf1
	for( k = 0; k < Natoms; k++ ){
		xc += conf1[k];
		yc += conf1[k+Natoms];
	}
	xc /= Natoms;
	yc /= Natoms;
	for( k = 0; k < Natoms; k++ ){
		conf1[k] -= xc;
		conf1[k+Natoms] -= yc;
	}
}
//------------------------------------------------------------
// Compute the gradient of Gaussian bumps
// V(conf) = LJpot(conf) + \sum_{j=0}^{Nbumps-1} h_j*
// exp(-0.5*{[CV1(conf) - val1_j]^2 + (CV2(conf) - val2_j)^2]}/sigma^2 )
// 
// void GaussBumps_pot_and_grad(double *conf,int Natoms,double *val1,double *val2,
// 	double *CV,double *CVgrad,double *height,
// 	double sig2,int Nbumps,double *pot,double *grad,double *biasing_pot) {
// 	double bump,aux1,aux2;
// 	int j,k;
// 	int dim = DIM*Natoms;
// 	
// 	// evaluate CVs and their gradients
// 	MargotCVgrad(conf,CV,CVgrad,NATOMS,nncv);
// 	// compute the output gradient
// 	*biasing_pot = 0.0;
// 	for( j=0; j<Nbumps; j++ ) {
// 		aux1 = CV[0] - val1[j];
// 		aux2 = CV[1] - val2[j];
// 		bump = height[j]*exp(-0.5*(aux1*aux1 + aux2*aux2)/sig2);
// 		*biasing_pot += bump;
// 		for( k=0; k<dim; k++ ) {
// 			grad[k] += -bump*(aux1*CVgrad[k] + aux2*CVgrad[k+dim])/sig2;
// 		}
// 	}
// 	*pot += *biasing_pot;
// } 

//------------------------------------------------------------
// Restraining pot and grad
void restraining_pot_and_grad(double *conf,double *pot,double *grad,int Natoms) {
	double xc = 0.0, yc = 0.0, dist2, aux_x,aux_y;
	int k;
	
	// center conf
	for( k = 0; k < Natoms; k++ ){
		xc += conf[k];
		yc += conf[k+Natoms];
	}
	xc /= Natoms;
	yc /= Natoms;	
	for( k = 0; k < Natoms; k++ ){
		aux_x = conf[k] - xc;
		aux_y = conf[k+Natoms] - yc;
		dist2 = aux_x*aux_x + aux_y*aux_y - 4.0; // 4 = 2^2
		if( dist2 > 0.0 ) {
			*pot += KAPPA*dist2*0.5;
			grad[k] -= KAPPA*aux_x;
			grad[k+Natoms] -= KAPPA*aux_y;		
		}
	}
}

//-------------------------------------------------------------

void Gpot_and_ders_on_grid(int Nbumps,double *bump_CV1,double *bump_CV2,double *height,
	double *grid_CV1, double *grid_CV2,double h1,double h2,
	double *pot,double *der1,double *der2,double *der12,int N1,int N2) {
	int i,j,n,ind;
	double sig2 = SIGMA*SIGMA;
	double aux1,aux2,aux_exp;
	double fac = GAMMA/(GAMMA + 1.0/BETA);
	int n1m1 = N1-1, n2m1 = N2-1;
	
	// the derivatives are computed by finite differences for the purpose of smoothing 
	// while they are available analytically
	// the derivatives are with respect to parameters (x1,x2) that rescale the 
	// cell in the CV space only a unit square:
	// x1 = (CV1- CV10)/h1, x2 = (CV2-CV20)/h2; CV1 = x1*h1 + CV10, x2 = h2*CV2 + CV20
	// d(pot)/d(x1) = d(pot)/d(x1) d(x1)/d(CV1) = h1*d(pot)/d(x1)
	// likewise for d(pot)d(x2)
	
	for( i = 0; i < N1; i++ ) {
		for( j = 0; j < N2; j++ ) {
			ind = i + j*N1;	
			pot[ind] = 0.0;
			if( i == 0 || j == 0 || i == n1m1 || j == n2m1 ) {
				der1[ind] = 0.0;
				der2[ind] = 0.0;
				der12[ind] = 0.0;
			}
			for( n = 0; n < Nbumps; n++ ) {
				aux1 = grid_CV1[i] - bump_CV1[n];
				aux2 = grid_CV2[j] - bump_CV2[n];
				aux_exp = height[n]*exp(-0.5*(aux1*aux1 + aux2*aux2)/sig2);
				pot[ind] += aux_exp;
				if( i == 0 || j == 0 || i == n1m1 || j == n2m1 ) {
					der1[ind] -= aux1*aux_exp/sig2;
					der2[ind] -= aux2*aux_exp/sig2;
					der12[ind] += (aux1/sig2)*(aux2/sig2)*aux_exp;
				}
				
			}
			pot[ind] *= fac;
			if( i == 0 || j == 0 || i == n1m1 || j == n2m1 ) {
				der1[ind] *= fac*h1;
				der2[ind] *= fac*h2;
				der12[ind] *= fac*h1*h2;
			}		
		}
	}
	for( i = 1; i < n1m1; i++ ) {
		for( j = 1; j < n2m1; j++ ) {
			ind = i + j*N1;	
			der1[ind] = 0.5*(pot[ind+1]-pot[ind-1]);
			der2[ind] = 0.5*(pot[ind+N1]-pot[ind-N1]);

		}
	}
	for( i = 1; i < n1m1; i++ ) {
		for( j = 1; j < n2m1; j++ ) {
			ind = i + j*N1;	
			der12[ind]  = 0.5*(der1[ind+N1] - der1[ind-N1]);
		}
	}
}

//----------------------
void compute_bicubic_coeff_matrix(double *pot,double *der1,double *der2,double *der12,
	double *Amatr,int ind,int N1,int N2) {
	int i,j,ind0 = ind,ind1 = ind + 1,ind2 = ind + N1,ind3 = ind + N1 + 1;
	double F[16]; // matrix F defined row-wise
	double B[16];
	// row 1
	F[0] = pot[ind0];
	F[1] = pot[ind2];
	F[2] = der2[ind0];
	F[3] = der2[ind2];
	// row 2
	F[4] = pot[ind1];
	F[5] = pot[ind3];
	F[6] = der2[ind1];
	F[7] = der2[ind3];
	// row 3
	F[8] = der1[ind0];
	F[9] = der1[ind2];
	F[10] = der12[ind0];
	F[11] = der12[ind2];
	// row 4
	F[12] = der1[ind1];
	F[13] = der1[ind3];
	F[14] = der12[ind1];
	F[15] = der12[ind3];
	// Computes A : = M F M^\top where
	// F = [f(0,0),f(0,1),der2(0,0),der2(0,1);
	//		f(1,0),f(1,1),der2(1,0),der2(1,1);
	//		der1(0,0),der1(0,1),der12(0,0),der12(0,1);
	//		der1(1,0),der1(1,1),der12(1,0),der12(1,1)]
	// (0,0) corresponds to ind0 = ind, 
	// (1,0) corresponds to ind1 = ind + 1, 
	// (0,1) corresponds to ind2 = ind + N1,
	// (1,1) corresponds to ind3 = ind + 1 + N1
	
	// B = FM^\top
	for( i = 0; i < 4; i++ ) {
		j = i*4;
		B[j] = wsum0(F[j],F[j+1],F[j+2],F[j+3]);
		B[j+1] = wsum1(F[j],F[j+1],F[j+2],F[j+3]);
		B[j+2] = wsum2(F[j],F[j+1],F[j+2],F[j+3]);
		B[j+3] = wsum3(F[j],F[j+1],F[j+2],F[j+3]);	
	}
	// A = M*B 
	for( i = 0; i < 4; i++ ) {
		Amatr[ind*16 + i] = wsum0(B[i],B[i+4],B[i+8],B[i+12]);
		Amatr[ind*16 + i + 4] = wsum1(B[i],B[i+4],B[i+8],B[i+12]);
		Amatr[ind*16 + i + 8] = wsum2(B[i],B[i+4],B[i+8],B[i+12]);
		Amatr[ind*16 + i + 12] = wsum3(B[i],B[i+4],B[i+8],B[i+12]);
	}	
}

// these functions perform multiplication of the matrix
// M = [1,0,0,0;
//      0,0,1,0;
//     -3,3,-2,1;
//      2,-2,1,1;]
// by vector [a,b,c,d]^\top

double wsum0(double a,double b,double c,double d) {
	return a;
}
double wsum1(double a,double b,double c,double d) {
	return c;
}
double wsum2(double a,double b,double c,double d) {
	return 3.0*(b-a) - 2.0*c - d;
}
double wsum3(double a,double b,double c,double d) {
	return 2.0*(a-b) + c + d;
}

//-------------------
// evaluate the free energy and its gradient at a query point
void evaluate_Gpot_and_ders(double *grid_CV1, double *grid_CV2,double h1,double h2,
	double *Amatr,double cv1,double cv2,double *FEval,double *FEder1,double *FEder2,int N1,int N2) {
	int i,j,ind,ishift;
	double x,y;
	// FEval(x,y) = \sum_{i,j=0}^3 a(i,j)x^i y^j
	// FEder1(x,y) = \sum_{i=1}^3\sum{j=0}^3 a(i,j)ix^{i-1} y^j
	// FEder2(x,y) = \sum_{i=0}^3\sum{j=1}^3 a(i,j)jx^i y^{j-1}
	
	// find the cell
	i = min(max(0,(int)floor((cv1 - grid_CV1[0])/h1)),N1-2);
	j = min(max(0,(int)floor((cv2 - grid_CV2[0])/h2)),N2-2);
	x = (cv1 - grid_CV1[0] - h1*i)/h1;
	y = (cv2 - grid_CV2[0] - h2*j)/h2;
	
	ind = i + N1*j;
	ishift = ind*16;
	*FEval = 0.0;
	*FEder1 = 0.0;
	*FEder2 = 0.0;
	for( i=0; i<4; i++ ) {
		for( j=0; j<4; j++ ) {
			*FEval += Amatr[ishift + i*4 + j]*pow(x,i)*pow(y,j);			
		}
	}
	for( i=1; i<4; i++ ) {
		for( j=0; j<4; j++ ) {
			*FEder1 += Amatr[ishift + i*4 + j]*i*pow(x,i-1)*pow(y,j);			
		}
	}
	for( i=0; i<4; i++ ) {
		for( j=1; j<4; j++ ) {
			*FEder2 += Amatr[ishift + i*4 + j]*j*pow(x,i)*pow(y,j-1);			
		}
	}
}


// //-------------------------------------------------------------
// 
// void test_CVgrad(double *conf0) {
// 	int k,dim = DIM*NATOMS;
// 	struct vec2 CV1,CV2;
// 	double h = 1.e-3;
// 	double *mu2grad,*mu3grad,*mu2,*mu3;
// 	mu2grad = (double *)malloc(dim*sizeof(double));
// 	mu3grad = (double *)malloc(dim*sizeof(double));
// 	mu2 = (double *)malloc(sizeof(double));
// 	mu3 = (double *)malloc(sizeof(double));
// 	CVgrad(conf0,mu2,mu3,mu2grad,mu3grad,NATOMS);
// 	for( k = 0; k < dim; k++ ) {
// 		conf0[k] += h;	
// 		CV1 = CVs(conf0,NATOMS);
// 		conf0[k] -= 2*h;	
// 		CV2 = CVs(conf0,NATOMS);
// 		conf0[k] += h;	
// 		printf("k = %i: mu2der = %.6e, mu3der = %.6e, mu2fd = %.6e, mu3fd = %.6e\n",
// 			k,mu2grad[k],mu3grad[k],0.5*(CV1.x-CV2.x)/h,0.5*(CV1.y-CV2.y)/h);
// 	}
// 	free(mu2grad);
// 	free(mu3grad);
// 	free(mu2);
// 	free(mu3);
// 
// 
// }
// 

