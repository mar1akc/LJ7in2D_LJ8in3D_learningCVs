// Generates sample positions for LJ7 in 2D using metadynamics.
// Biasing with respect to (mu2, mu3), the 2nd and 3rd moments of coordination numbers.
// MALA algorithm is used.
// The phase space is 14D.
// The positions and the heights of the bumps are saved.
// The samples are recorded at every NSKIP step.
// A total of NSAVE samples are recorded.

// Compile command:  gcc -Wall LJ8in3D_WTMetad_mu2mu3.c CV_helpers.c timestep_helpers.c -lm -O3

// Rules-of-thumb for well-tempered metadynamics
// --> gamma should be approximately equal to the max 
// depth that needs to be filled with Gaussian bumps
// --> sigma should >= the size of features that we want to resolve
// --> height should be such that height*Nbumps*(2*pi*sigma^2)^{dim/2} \approx Volume to be filled

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "macros_and_constants.h"
#include "timestep_helpers.h"
#include "CV_helpers.h"

#define NSKIP 500  // save data every 10^6 steps
#define NSAVE 100000 // 10^5 time stes
#define NSTEPS_BETWEEN_DEPOSITS 500
#define NBUMPS_MAX 50000 // 10000 the number of Gaussian bumps to be deposited

// struct vec2 {
// 	double x;
// 	double y;
// };

void total_pot_and_grad(double *conf,int Natoms,double *val1,double *val2,
	double *CV,double *CVgrad,double *hight,
	double sig2,int Nbumps,double *pot,double *grad,double *biasing_pot);
void WTMetadynamics(int Nsteps_between_deposits,int Nbumps_max,int Natoms,
	double *conf0,double dt,double *val1,double *val2,double *height);
int main(void);

//------------------------------------------------------------
// Evaluate the total potential energy and its gradient

void total_pot_and_grad(double *conf,int Natoms,double *val1,double *val2,
	double *CV,double *CVgrad,double *hight,
	double sig2,int Nbumps,double *pot,double *grad,double *biasing_pot) {

	LJpot_and_grad(conf,pot,grad,Natoms);
	restraining_pot_and_grad(conf,pot,grad,Natoms);
	GaussBumps_pot_and_grad(conf,Natoms,val1,val2,CV,CVgrad,hight,
			sig2,Nbumps,pot,grad,biasing_pot);	
}

//------------------------------------------------------------
void WTMetadynamics(int Nsteps_between_deposits,int Nbumps_max,int Natoms,
	double *conf0,double dt,double *val1,double *val2,double *height) {
	int n,k,dim = Natoms*DIM,kshift;
	int Nbumps = 0;
	double *gauss01;
	int Ngauss = 2*DIM*Natoms, jgauss;
	double std = sqrt(2.0*dt/BETA);
	double *conf1,*Vpot0,*Vpot1,*Vgrad0,*Vgrad1,*w;
	double *biasing_pot0,*biasing_pot1;
	double *CV,*CVgrad;
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
	CV = (double *)malloc(CVDIM*sizeof(double));
	CVgrad = (double *)malloc(CVDIM*dim*sizeof(double));
	gauss01 = (double *)malloc(Ngauss*sizeof(double));
	
	total_pot_and_grad(conf0,Natoms,val1,val2,CV,CVgrad,height,
			sig2,Nbumps,Vpot0,Vgrad0,biasing_pot0);
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
			total_pot_and_grad(conf1,Natoms,val1,val2,CV,CVgrad,height,
					sig2,Nbumps,Vpot1,Vgrad1,biasing_pot1);
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
		mu2mu3(conf0,CV);
		val1[Nbumps] = CV[0];
		val2[Nbumps] = CV[1];
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

	dim = DIM*Natoms;
	conf0 = (double *)malloc(dim*sizeof(double));
	val1 = (double *)malloc(Nbumps_max*sizeof(double));
	val2 = (double *)malloc(Nbumps_max*sizeof(double));
	height = (double *)malloc(Nbumps_max*sizeof(double));
	
	
	init_conf(conf0,0);
// 	printf("dim = %i\n",dim);
// 	for( k = 0; k < dim; k++ ) printf("k = %i, conf0 = %.6e\n",k,conf0[k]);
	
	// test CVgrad
	double h = 1.e-3;
	double *mugrad,*mu23,*CV1,*CV2;
	mugrad = (double *)malloc(CVDIM*dim*sizeof(double));
	mu23 = (double *)malloc(CVDIM*sizeof(double));
	CV1 = (double *)malloc(CVDIM*sizeof(double));
	CV2 = (double *)malloc(CVDIM*sizeof(double));
	mu2mu3grad(conf0,mu23,mugrad,Natoms);
	for( k = 0; k < dim; k++ ) {
		conf0[k] += h;	
		mu2mu3(conf0,CV1);
		conf0[k] -= 2*h;	
		mu2mu3(conf0,CV2);
		conf0[k] += h;	
		printf("k = %i: mu2der = %.6e, mu3der = %.6e, mu2fd = %.6e, mu3fd = %.6e\n",
			k,mugrad[k],mugrad[k+dim],0.5*(CV1[0]-CV2[0])/h,0.5*(CV1[1]-CV2[1])/h);
	}
	
	
	sprintf(fbump_name,"GbumpsData/GaussianBumps_beta%.f.txt",BETA); 

	fpot = fopen(fbump_name,"w");	
	
	CPUbegin=clock(); // start time measurement

	WTMetadynamics(Nsteps_between_deposits,Nbumps_max,Natoms,
		conf0,dt,val1,val2,height);
	for( k=0; k<Nbumps_max; k++ ) {
		fprintf(fpot,"%.10e\t%.10e\t%.10e\n",height[k],val1[k],val2[k]);
	}
	
	cpu = (clock()-CPUbegin)/((double)CLOCKS_PER_SEC);	// end time measurement		
	printf("CPU time = %g\n",cpu);
	
	fclose(fpot);
	
	return 0;
}
