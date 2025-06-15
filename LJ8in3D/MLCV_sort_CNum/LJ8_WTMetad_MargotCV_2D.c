// Generates sample positions for LJ8 in 3D using metadynamics.
// Biasing with respect to a learned 2D CV represented via a neural network with the following architecture:
// cnsort = sorted vector of coordination numbers, size Natoms
// v1 = A1*cnsort + b1, A1 is dim1xdim0
// w1 = ELU(v1)
// v2 = A2*w1 + b2, A2 is dim2xdim1
// w2 = ELU(v2)
// v3 = A3*w2 + b3, A3 is dim3xdim2
// w3 = ELU(v3)
// CV = A4*w3 + b4, A4 is dim4xdim3

// MALA algorithm is used.
// The phase space is 2*Natoms.
// The positions and the heights of the bumps are saved.
// The samples are recorded at every NSKIP step.
// A total of NSAVE samples are recorded.

// Compile command:  gcc -Wall LJ8_WTMetad_MargotCV_2D.c MargotColVar_CoordNum.c timestep_helpers.c -lm -O3

// Rules-of-thumb for well-tempered metadynamics
// --> gamma should be approximately equal to the max 
// depth that needs to be filled with Gaussian bumps
// --> sigma should >= the size of features that we want to resolve
// --> height should be such that height*Nbumps*(2*pi*sigma^2)^{dim/2} \approx Volume to be filled

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "MargotColVar_CoordNum.h"
#include "timestep_helpers.h"
#include "macros_and_constants.h"

#define NSKIP 500  // save data every 10^6 steps
#define NSAVE 100000 // 10^5 time stes

double LJpot(double *conf,int Natoms);
void LJpot_and_grad(double *conf,double *pot,double *grad,int Natoms);
char MALAstep(double *conf0,double *conf1,int Natoms,double dt, 
			double *Vpot0,double *Vpot1,double *Vgrad1,double *w);
// aligns configuration by solving Wahba's problem
// https://en.wikipedia.org/wiki/Wahba%27s_problem
void align( double *conf0, double *conf1, int Natoms ); 
// collective variables mu2 and mu3
// struct vec2 CVs(double *conf,int Natoms);
// void CVgrad(double *conf,double *mu2,double *mu3,
// 			double *mu2grad,double *mu3grad,int Natoms);
void GaussBumps_pot_and_grad_MargotCV(double *conf,int Natoms,double *val1,double *val2,
	double *CVval,double *CVgrad,double *height,
	double sig2,int Nbumps,double *pot,double *grad,double *biasing_pot,
	struct NN *nncv);

void total_pot_and_grad(double *conf,int Natoms,double *val1,double *val2,
	double *CVval,double *CVgrad,double *hight,
	double sig2,int Nbumps,double *pot,double *grad,double *biasing_pot,
	struct NN *nncv);
	
void WTMetadynamics(int Nsteps_between_deposits,int Nbumps_max,int Natoms,
	double *conf0,double dt,double *val1,double *val2,double *height,
	struct NN *nncv);
	
void restraining_pot_and_grad(double *conf,double *pot,double *grad,int Natoms);
int main(void);

//------------------------------------------------------------
// Compute the gradient of Gaussian bumps
// V(conf) = LJpot(conf) + \sum_{j=0}^{Nbumps-1} h_j*
// exp(-0.5*{[CV1(conf) - val1_j]^2 + (CV2(conf) - val2_j)^2]}/sigma^2 )

void GaussBumps_pot_and_grad_MargotCV(double *conf,int Natoms,double *val1,double *val2,
	double *CVval,double *CVgrad,double *height,
	double sig2,int Nbumps,double *pot,double *grad,double *biasing_pot,
	struct NN *nncv) {
	
	double bump,aux1,aux2;
	int j,k;
	int dim = DIM*Natoms;
	
	// evaluate CVs and their gradients
	MargotCVgrad(CVgrad,CVval,conf,Natoms,nncv);
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
// Evaluate the total potential energy and its gradient

void total_pot_and_grad(double *conf,int Natoms,double *val1,double *val2,
	double *CVval,double *CVgrad,double *height,
	double sig2,int Nbumps,double *pot,double *grad,double *biasing_pot,
	struct NN *nncv) {

	LJpot_and_grad(conf,pot,grad,Natoms);
	
	restraining_pot_and_grad(conf,pot,grad,Natoms);
	
	GaussBumps_pot_and_grad_MargotCV(conf,Natoms,val1,val2,CVval,CVgrad,height,
			sig2,Nbumps,pot,grad,biasing_pot,nncv);	


// void GaussBumps_pot_and_grad(double *conf,int Natoms,double *val1,double *val2,
// 	double *CVval,double *CVgrad,double *height,
// 	double sig2,int Nbumps,double *pot,double *grad,double *biasing_pot,
// 	double *A1,double *A2,double *A3,double *b1,double *b2,double *b3,
// 	int dim1,int dim2,int dim3,int dim4)

}

//------------------------------------------------------------
void WTMetadynamics(int Nsteps_between_deposits,int Nbumps_max,int Natoms,
	double *conf0,double dt,double *val1,double *val2,double *height,
	struct NN *nncv) {

	int n,k,dim = Natoms*DIM;
	int dim4 = nncv->dim4;
	double *gauss01;
	int Ngauss = 2*DIM*Natoms, jgauss;
	int Nbumps = 0;
	int kshift;
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
	CVgrad = (double *)malloc(dim4*dim*sizeof(double));
	gauss01 = (double *)malloc(Ngauss*sizeof(double));

	
	total_pot_and_grad(conf0,Natoms,val1,val2,CVval,CVgrad,height,
			sig2,Nbumps,Vpot0,Vgrad0,biasing_pot0,nncv);

	jgauss = 0;
	for( Nbumps = 0; Nbumps<Nbumps_max; Nbumps++ ) {	
		if( Nbumps%100 == 0) printf("Nbumps = %i\n",Nbumps);	
		for( n=0; n < Nsteps_between_deposits; n++ ) {
			// generate array of random vars N(0,std) of size 2*DIM*Natoms
			if( jgauss == 0 ) box_mueller(gauss01,Ngauss);
			kshift = jgauss*dim;
			for( k=0; k<dim; k++ ) {
				w[k] = std*gauss01[k+kshift];
			}
			if( jgauss == 1 ) jgauss = 0;
			else jgauss = 1;
			// propose move
			for( k = 0; k < dim; k++ ) {
				conf1[k] = conf0[k] - dt*Vgrad0[k] + w[k];
			}
			// evaluate the potential and the gradient at the proposed point
			total_pot_and_grad(conf1,Natoms,val1,val2,CVval,CVgrad,height,
					sig2,Nbumps,Vpot1,Vgrad1,biasing_pot1,nncv);
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
		MargotCV(CVval,conf0,Natoms,nncv);
		val1[Nbumps] = CVval[0];
		val2[Nbumps] = CVval[1];
	}
}


//------------------------------------------------------------
int main(void){
	int Nsteps_between_deposits = NSTEPS_BETWEEN_DEPOSITS,Nbumps_max = NBUMPS_MAX;
	int Natoms = NATOMS,dim,k;
	double dt = TAU;
	double *conf0;
	double *height,*val1,*val2;
	FILE *fpot;
    clock_t CPUbegin; // for measuring CPU time
    double cpu; // for recording CPU time
    char fbump_name[100];
	char fCVname[] = "MargotCV_NNdata.txt";
	char fname_dim[] = "MargotCV_dimensions.txt";
	struct NN *nncv;

	dim = DIM*Natoms;
	conf0 = (double *)malloc(dim*sizeof(double));
	val1 = (double *)malloc(Nbumps_max*sizeof(double));
	val2 = (double *)malloc(Nbumps_max*sizeof(double));
	height = (double *)malloc(Nbumps_max*sizeof(double));
	nncv = (struct NN *)malloc(sizeof(struct NN));
	
	init_conf(conf0,0);
	
	
	readCVdata(fCVname,fname_dim,nncv);

    

	val1 = (double *)malloc(Nbumps_max*sizeof(double));
	val2 = (double *)malloc(Nbumps_max*sizeof(double));
	height = (double *)malloc(Nbumps_max*sizeof(double));
	
	
	CPUbegin=clock(); // start time measurement

	WTMetadynamics(Nsteps_between_deposits,Nbumps_max,Natoms,
		conf0,dt,val1,val2,height,nncv);

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
