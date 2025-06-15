// Generates sample positions for LJ7 in 2D using metadynamics.
// Biasing with respect to a learned 2D CV represented via a neural network with the following architecture:
// d2sort = sorted vector of pairwise distances squared, size Natoms*(Natoms-1)/2 = 21
// v1 = A1*d2sort + b1, A1 is 30x21
// w1 = ELU(v1)
// v2 = A2*w1 + b2, A2 is 30x30
// w2 = ELU(v2)
// CV = A3*w2 + b3, A3 is 2x30
// MALA algorithm is used.
// The phase space is 14D.
// The positions and the heights of the bumps are saved.
// The samples are recorded at every NSKIP step.
// A total of NSAVE samples are recorded.

// Compile command:  gcc -Wall LJ7_WTMetad_MargotCV_2D.c MargotColVar.c -lm -O3

// Rules-of-thumb for well-tempered metadynamics
// --> gamma should be approximately equal to the max 
// depth that needs to be filled with Gaussian bumps
// --> sigma should >= the size of features that we want to resolve
// --> height should be such that height*Nbumps*(2*pi*sigma^2)^{dim/2} \approx Volume to be filled

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "MargotColVar.h"

#define BETA 5.0
#define GAMMA 1.0 // the artificial temperature for the discount factor
#define NSKIP 500  // save data every 10^6 steps
#define NSAVE 100000 // 10^5 time stes
#define PI 3.141592653589793
#define PI2 6.283185307179586 // 2*PI
#define RSTAR 1.122462048309373 // 2^{1/6}
#define TAU 5.0e-5
#define NATOMS 7 // the number of atoms
#define SIGMA 0.02 // width parameter for Gaussian bumps
#define HEIGHT 0.01 // height of Gaussian bump
#define NSTEPS_BETWEEN_DEPOSITS 500
#define NBUMPS_MAX 50000 // 10000 the number of Gaussian bumps to be deposited
#define KAPPA 100.0 // spring constant for the restraining potential that turns on 
// if an atom is at a distance more than 2 from the center of mass
#define mabs(a) ((a) >= 0 ? (a) : -(a))
#define sgn(a) ((a) == 0 ? 0 : ((a) > 0  ? 1 : -1 ))
#define max(a,b) ((a) >= (b) ? (a) : (b))
#define min(a,b) ((a) <= (b) ? (a) : (b))

struct vec2 {
	double x;
	double y;
};

double LJpot(double *conf,int Natoms);
void LJpot_and_grad(double *conf,double *pot,double *grad,int Natoms);
char MALAstep(double *conf0,double *conf1,int Natoms,double dt, 
			double *Vpot0,double *Vpot1,double *Vgrad1,double *w);
struct vec2	box_mueller(void); // generates a pair of Gaussian random variables N(0,1)
// aligns configuration by solving Wahba's problem
// https://en.wikipedia.org/wiki/Wahba%27s_problem
void align( double *conf0, double *conf1, int Natoms ); 
// collective variables mu2 and mu3
// struct vec2 CVs(double *conf,int Natoms);
// void CVgrad(double *conf,double *mu2,double *mu3,
// 			double *mu2grad,double *mu3grad,int Natoms);
void GaussBumps_pot_and_grad(double *conf,int Natoms,double *val1,double *val2,
	double *CVval,double *CVgrad,double *height,
	double sig2,int Nbumps,double *pot,double *grad,double *biasing_pot,
	double *A1,double *A2,double *A3,double *b1,double *b2,double *b3,
	int dim1,int dim2,int dim3,int dim4);

void total_pot_and_grad(double *conf,int Natoms,double *val1,double *val2,
	double *CVval,double *CVgrad,double *hight,
	double sig2,int Nbumps,double *pot,double *grad,double *biasing_pot,
	double *A1,double *A2,double *A3,double *b1,double *b2,double *b3,
	int dim1,int dim2,int dim3,int dim4);
	
void WTMetadynamics(int Nsteps_between_deposits,int Nbumps_max,int Natoms,
	double *conf0,double dt,double *val1,double *val2,double *height,
	double *A1,double *A2,double *A3,double *b1,double *b2,double *b3,int dim2,int dim3);
	
void restraining_pot_and_grad(double *conf,double *pot,double *grad,int Natoms);
int main(void);

//-------------------------------------------------------------


struct vec2	box_mueller(){
	double x1, y1, p, q;
	struct vec2 g;
	
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
		g.x=cos(p)*sqrt(-q);
		g.y=sin(p)*sqrt(-q);
		return g;
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
	// evaluate the transition probabilities Q01 and Q10
	Q01 = 0.0;
	Q10 = 0.0;
	for( k=0; k < Natoms; k++ ) {
		Q01 += w[k]*w[k] + w[k+Natoms]*w[k+Natoms];
		aux = conf0[k]-conf1[k] + dt*Vgrad1[k];
		Q10 += aux*aux;
		aux = conf0[k+Natoms]-conf1[k+Natoms] + dt*Vgrad1[k+Natoms];
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
	double B11 = 0.0, B12 = 0.0, B21 = 0.0, B22 = 0.0;
	double xc = 0.0, yc = 0.0;
	double alpha = 0.0, c, s, x, y; // the angle
	int k;
	// format: conf = [x0, x1, ..., x_{Natoms}, y0, y1, ..., y_{Natoms}]
	// conf0 must be centered so that its center of mass is at the origin
	// B = Sum(u_i v_i^\top)
	
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
	
	// 	// B = [conf0_x;conf0_y][conf1_x;conf1_y]^\top
	for( k = 0; k < Natoms; k++ ) {
		B11 += conf0[k]*conf1[k];
		B12 += conf0[k]*conf1[k+Natoms];
		B21 += conf0[k+Natoms]*conf1[k];
		B22 += conf0[k+Natoms]*conf1[k+Natoms];
	} 

	// solve 1D optimization problem
	// f(x) = (B11 - cos(x))^2 + (B12 + sin(x))^2 +
	// (B21 - sin(x))^2 + (B22 - cos(x))^2 --> min
	// f(x) = -(B11+B22)*cos(x) - (B21-B12)*sin(x)
	// f'(x) = (B11+B22)*sin(x) - (B21-B12)*cos(x)
	// alpha = atan((B21-B12)/(B11+B22));
	// R = [cos(x),-sin(x); sin(x), cos(x)]
	alpha = atan((B21-B12)/(B11+B22));
	c = cos(alpha);
	s = sin(alpha);
	for( k = 0; k < Natoms; k++ ){
		x = conf1[k];
		y = conf1[k+Natoms];
		conf1[k] = x*c - y*s;
		conf1[k+Natoms] = x*s + y*c;
	}
}
//------------------------------------------------------------
// Compute the gradient of Gaussian bumps
// V(conf) = LJpot(conf) + \sum_{j=0}^{Nbumps-1} h_j*
// exp(-0.5*{[CV1(conf) - val1_j]^2 + (CV2(conf) - val2_j)^2]}/sigma^2 )

void GaussBumps_pot_and_grad(double *conf,int Natoms,double *val1,double *val2,
	double *CVval,double *CVgrad,double *height,
	double sig2,int Nbumps,double *pot,double *grad,double *biasing_pot,
	double *A1,double *A2,double *A3,double *b1,double *b2,double *b3,
	int dim1,int dim2,int dim3,int dim4) {
	
	double bump,aux1,aux2;
	int j,k;
	int dim = 2*Natoms;
	
	// evaluate CVs and their gradients
	MargotCVgrad(CVgrad,CVval,conf,Natoms,A1,A2,A3,b1,b2,b3,dim1,dim2,dim3,dim4);
// 	CVgrad(conf,CV1,CV2,CV1grad,CV2grad,Natoms);
	// compute the output gradient
	*biasing_pot = 0.0;
	for( j=0; j<Nbumps; j++ ) {
		aux1 = CVval[0] - val1[j];
		aux2 = CVval[1] - val2[j];
		bump = height[j]*exp(-0.5*(aux1*aux1 + aux2*aux2)/sig2);
		*biasing_pot += bump;
		for( k=0; k<dim; k++ ) {
			grad[k] += -bump*(aux1*CVgrad[k] + aux2*CVgrad[k+dim])/sig2;
		}
	}
	*pot += *biasing_pot;
} 

//------------------------------------------------------------
// Restraining pot and grad
void restraining_pot_and_grad(double *conf,double *pot,double *grad,int Natoms) {
	double xc = 0.0, yc = 0.0, dist2, aux_x, aux_y;
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
		dist2 = aux_x*aux_x + aux_y*aux_y - 4.0;
		if( dist2 > 0.0 ) {
			*pot += KAPPA*dist2*0.5;
			grad[k] = KAPPA*aux_x;
			grad[k+Natoms] = KAPPA*aux_y;		
		}
	}
}

//------------------------------------------------------------
// Evaluate the total potential energy and its gradient

void total_pot_and_grad(double *conf,int Natoms,double *val1,double *val2,
	double *CVval,double *CVgrad,double *height,
	double sig2,int Nbumps,double *pot,double *grad,double *biasing_pot,
	double *A1,double *A2,double *A3,double *b1,double *b2,double *b3,
	int dim1,int dim2,int dim3,int dim4) {

	LJpot_and_grad(conf,pot,grad,Natoms);
	
	restraining_pot_and_grad(conf,pot,grad,Natoms);
	
	GaussBumps_pot_and_grad(conf,Natoms,val1,val2,CVval,CVgrad,height,
			sig2,Nbumps,pot,grad,biasing_pot,A1,A2,A3,b1,b2,b3,dim1,dim2,dim3,dim4);	


// void GaussBumps_pot_and_grad(double *conf,int Natoms,double *val1,double *val2,
// 	double *CVval,double *CVgrad,double *height,
// 	double sig2,int Nbumps,double *pot,double *grad,double *biasing_pot,
// 	double *A1,double *A2,double *A3,double *b1,double *b2,double *b3,
// 	int dim1,int dim2,int dim3,int dim4)

}

//------------------------------------------------------------
void WTMetadynamics(int Nsteps_between_deposits,int Nbumps_max,int Natoms,
	double *conf0,double dt,double *val1,double *val2,double *height,
	double *A1,double *A2,double *A3,double *b1,double *b2,double *b3,int dim2,int dim3) {

	int n,k,dim = Natoms*2;
	const int dim1 = Natoms*(Natoms - 1)/2,dim4 = 2;
	int Nbumps = 0;
	struct vec2 gauss01;
	double std = sqrt(2.0*dt/BETA);
	double *conf1,*Vpot0,*Vpot1,*Vgrad0,*Vgrad1,*w;
	double *biasing_pot0,*biasing_pot1;
	double *CVval,*CVgrad;
	double sig2 = SIGMA*SIGMA;
	char ch;
	
	conf1 = (double *)malloc(dim*sizeof(double));
	Vgrad0 = (double *)malloc(dim*sizeof(double));
	Vgrad1 = (double *)malloc(dim*sizeof(double));
	Vpot0 = (double *)malloc(sizeof(double));
	Vpot1 = (double *)malloc(sizeof(double));
	biasing_pot0 = (double *)malloc(sizeof(double));
	biasing_pot1 = (double *)malloc(sizeof(double));
	w = (double *)malloc(dim*sizeof(double));
	CVval = (double *)malloc(dim4*sizeof(double));
	CVgrad = (double *)malloc(2*dim*sizeof(double));

	
	total_pot_and_grad(conf0,Natoms,val1,val2,CVval,CVgrad,height,
			sig2,Nbumps,Vpot0,Vgrad0,biasing_pot0,A1,A2,A3,b1,b2,b3,dim1,dim2,dim3,dim4);
			
	for( Nbumps = 0; Nbumps<Nbumps_max; Nbumps++ ) {	
		if( Nbumps%100 == 0) printf("Nbumps = %i\n",Nbumps);	
		for( n=0; n < Nsteps_between_deposits; n++ ) {
			// generate array of random vars N(0,std) of size 2*Natoms
			for( k=0; k<Natoms; k++ ) {
				gauss01 = box_mueller();
				w[k] = std*gauss01.x;
				w[k+Natoms] = std*gauss01.y;			
			}
			// propose move
			for( k = 0; k < Natoms; k++ ) {
				conf1[k] = conf0[k] - dt*Vgrad0[k] + w[k];
				conf1[k+Natoms] = conf0[k+Natoms] - dt*Vgrad0[k+Natoms] + w[k+Natoms];
			}
			// evaluate the potential and the gradient at the proposed point
			total_pot_and_grad(conf1,Natoms,val1,val2,CVval,CVgrad,height,
					sig2,Nbumps,Vpot1,Vgrad1,biasing_pot1,A1,A2,A3,b1,b2,b3,dim1,dim2,dim3,dim4);
			ch = MALAstep(conf0,conf1,Natoms,dt,Vpot0,Vpot1,Vgrad1,w);
			if( ch == 1 ) { // step was accepted
				// align configurations
				align(conf0,conf1,Natoms);
				for( k=0; k<dim; k++ ) {
					conf0[k] = conf1[k];
					Vgrad0[k] = Vgrad1[k];
				}
				*Vpot0 = *Vpot1;
				*biasing_pot0 = *biasing_pot1;		
			}
		}
		height[Nbumps] = HEIGHT*exp(-(*biasing_pot0)/GAMMA);
		MargotCV(CVval,conf0,Natoms,A1,A2,A3,b1,b2,b3,dim1,dim2,dim3,dim4);
		val1[Nbumps] = CVval[0];
		val2[Nbumps] = CVval[1];
	}
}

//------------------------------------------------------------
int main(void){
	int Nsteps_between_deposits = NSTEPS_BETWEEN_DEPOSITS,Nbumps_max = NBUMPS_MAX;
	int Natoms = NATOMS,dim = NATOMS*2,k;
	double dt = TAU;
	double *conf0;
	double *height,*val1,*val2;
	FILE *fpot;
    clock_t CPUbegin; // for measuring CPU time
    double cpu; // for recording CPU time
    char fbump_name[100];
	int i,*isort;
	char fCVname[] = "MargotCVdata.txt";
	double *A1,*A2,*A3,*b1,*b2,*b3;
	double *d2;
	int dim1 = Natoms*(Natoms-1)/2;
	int dim2 = 30,dim3 = 30,dim4 = 2;
	
	
	// d2 is a sorted list of interatomic distances
	conf0 = (double *)malloc(dim*sizeof(double));
	isort = (int *)malloc(dim1*sizeof(int));
	d2 = (double *)malloc(dim1*sizeof(double));
	
	init_conf(conf0,Natoms); 
	
	for( i = 0; i < dim1; i++ ) {
		isort[i] = i;
	}	
	
	// test sortdist2
	sortdist2(conf0,d2,isort,Natoms,dim1);
	for( i = 0; i < dim1; i++ ) {
		printf("d2[%i] = %.4f,isort[%i] = %i\n",i,d2[i],i,isort[i]);
	}
	printf("Done testing sortdist2\n");


	
	A1 = (double *)malloc(dim1*dim2*sizeof(double));
	A2 = (double *)malloc(dim2*dim3*sizeof(double));
	A3 = (double *)malloc(dim3*dim4*sizeof(double));
	b1 = (double *)malloc(dim2*sizeof(double));
	b2 = (double *)malloc(dim3*sizeof(double));
	b3 = (double *)malloc(dim4*sizeof(double));
	
	readCVdata(fCVname,dim1,dim2,dim3,dim4,A1,A2,A3,b1,b2,b3);

    

	val1 = (double *)malloc(Nbumps_max*sizeof(double));
	val2 = (double *)malloc(Nbumps_max*sizeof(double));
	height = (double *)malloc(Nbumps_max*sizeof(double));
	
	
	CPUbegin=clock(); // start time measurement

	WTMetadynamics(Nsteps_between_deposits,Nbumps_max,Natoms,
		conf0,dt,val1,val2,height,A1,A2,A3,b1,b2,b3,dim2,dim3);

	sprintf(fbump_name,"GbumpsData/GaussianBumps_beta%.f.txt",BETA); 
	fpot = fopen(fbump_name,"w");
	for( k=0; k<Nbumps_max; k++ ) {
		fprintf(fpot,"%.10e\t%.10e\t%.10e\n",height[k],val1[k],val2[k]);
	}
	
	cpu = (clock()-CPUbegin)/((double)CLOCKS_PER_SEC);	// end time measurement		
	printf("CPU time = %g\n",cpu);
	
	fclose(fpot);
	
	return 0;
}
