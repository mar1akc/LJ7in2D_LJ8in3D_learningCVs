// This code does two things.
//
// (1) Reads well-tempered metadynamics data (Gaussian bumps) and approximates 
// the biasing potential with a bicubic spline. The derivatives are evaluated 
// by finite differences for smoothing purposes.
//
// (2) Runs a long trajectory in the biased force field and find the free energy 
// using the binning approach.

// Compile command:  gcc -Wall LJ7in2D_bicubicFE_binning.c CV_helpers.c timestep_helpers.c -lm -O3

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "macros_and_constants.h"
#include "timestep_helpers.h"
#include "CV_helpers.h"


#define N1 129 // the number of grid points along mu2-axis
#define N2 129 // the number of grid points along mu3-axis

#define NSTEPS 1e9  // the length of the stochastic trajectory that we bin


struct vec2 {
	double x;
	double y;
};

void total_pot_and_grad(double *conf,int Natoms,double *pot,double *grad,
	double *CV,double *CVgrad,
	double *grid_CV1,double *grid_CV2,double h1,double h2,
	double *bicubic_matrix,double *FEpot,double *FEder1,double *FEder2);
//----- running long trajectory in the biased potential and binning it
void binning_trajectory(long *bins,double *grid_CV1,double *grid_CV2,
	int Nsteps,int Natoms,double *conf0,double dt,double h1,double h2,
	double *bicubic_matrix,double *bin_confs);
void FEders_on_grid(double *pot,double *der1,double *der2,double *der12);

//----- main	
int main(void);

//-------------------------------------------------------------



//------------------------------------------------------------
// Evaluate the total potential energy and its gradient

void total_pot_and_grad(double *conf,int Natoms,double *pot,double *grad,
	double *CV,double *CVgrad,
	double *grid_CV1,double *grid_CV2,double h1,double h2,
	double *bicubic_matrix,double *FEpot,double *FEder1,double *FEder2) {
	
	int j,dim = Natoms*DIM;

	LJpot_and_grad(conf,pot,grad,Natoms);
	restraining_pot_and_grad(conf,pot,grad,Natoms);
	mu2mu3grad(conf,CV,CVgrad,CVDIM);			
	evaluate_Gpot_and_ders(grid_CV1,grid_CV2,h1,h2,bicubic_matrix,CV[0],CV[1],
				FEpot,FEder1,FEder2,N1,N2);
	// need to divide by h as FEder is the derivative w.r.t a parameter in a\in(0,1)
	// d(FE)/dCV = d(FE)/da * da/dCV, a(CV) = (CV - CV0)/h, da/dCV = 1/h	
	for( j=0; j<dim; j++ ) {
		grad[j] += *FEder1*CVgrad[j]/h1 + *FEder2*CVgrad[j+dim]/h2;
	}	
	*pot += *FEpot;	
	
// 	printf("CV1 = %.4e, CV2 = %.4e\n",*CV1,*CV2);
// 	printf("FEpot = %.4e,FEder1 = %.4e,FEder2 = %.4e\n",*FEpot,*FEder1,*FEder2);
		
}


//------------------------------------------------------------
void binning_trajectory(long *bins,double *grid_CV1,double *grid_CV2,
	int Nsteps,int Natoms,double *conf0,double dt,double h1,double h2,
	double *bicubic_matrix,double *bin_confs) {

	int j,n,k,dim = Natoms*DIM,j1,j2,kshift;
	int ind;
	double *gauss01;
	int Ngauss = 2*DIM*Natoms, jgauss;
	double std = sqrt(2.0*dt/BETA);
	double *conf1,*Vpot0,*Vpot1,*Vgrad0,*Vgrad1,*w;
	double *CV,*CVgrad,*FEpot,*FEder1,*FEder2;
	char ch;
	
	conf1 = (double *)malloc(dim*sizeof(double));
	Vgrad0 = (double *)malloc(dim*sizeof(double));
	Vgrad1 = (double *)malloc(dim*sizeof(double));
	Vpot0 = (double *)malloc(sizeof(double));
	Vpot1 = (double *)malloc(sizeof(double));
	w = (double *)malloc(dim*sizeof(double));
	CV = (double *)malloc(CVDIM*sizeof(double));
	CVgrad = (double *)malloc(CVDIM*dim*sizeof(double));
	FEpot = (double *)malloc(sizeof(double));
	FEder1 = (double *)malloc(sizeof(double));
	FEder2 = (double *)malloc(sizeof(double));
	gauss01 = (double *)malloc(Ngauss*sizeof(double));
	
	total_pot_and_grad(conf0,Natoms,Vpot0,Vgrad0,CV,CVgrad,
		grid_CV1,grid_CV2,h1,h2,bicubic_matrix,FEpot,FEder1,FEder2);

	jgauss = 0;				
	for( j = 0; j < Nsteps; j++ ) {				
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
		total_pot_and_grad(conf1,Natoms,Vpot1,Vgrad1,CV,CVgrad,
			grid_CV1,grid_CV2,h1,h2,bicubic_matrix,FEpot,FEder1,FEder2);
		ch = MALAstep(conf0,conf1,Natoms,dt,Vpot0,Vpot1,Vgrad1,w);
		if( ch == 1 ) { // step was accepted
			// align configurations
			align(conf0,conf1,Natoms);
			for( k=0; k<dim; k++ ) {
				conf0[k] = conf1[k];
				Vgrad0[k] = Vgrad1[k];
			}
			*Vpot0 = *Vpot1;
		}
		// bin the current position 
		// the grid points are the centers of the bins
		
		j1 = min(max(0,(int)floor((CV[0] - grid_CV1[0])/h1+0.5)),N1-1);
		j2 = min(max(0,(int)floor((CV[1] - grid_CV2[0])/h2 +0.5)),N2-1);
		ind = j1 + j2*N1;
// 		printf("%i, %i\n",j1,j2);
// 		printf("%.4f, %.4e\n",(*CV1 - grid_CV1[0])/h1+0.5,(*CV2 - grid_CV2[0])/h2 +0.5);
// 		exit(1);
		
		if( bins[ind] == 0 ) {
			n = ind*dim;
			for( k=0; k < dim; k++ ) {
				bin_confs[n + k] = conf0[k];
			}
		}		
		bins[ind]++;
		
		if( j%100000 == 0 ) printf("binning: %i steps\n",j);
	}
}

//-----------------------------------------------------------

void FEders_on_grid(double *pot,double *der1,double *der2,double *der12) {

	int n1m1 = N1-1, n2m1 = N2-1;
	int i,j,ind;

	// interior grid points
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
	// borders i = 0 and i = n1m1
	for( j = 1; j < n2m1; j++ ) {
		// i = 0
		ind = j*N1;
		der1[ind] = pot[ind+1] - pot[ind];
		der2[ind] = 0.5*(pot[ind+N1]-pot[ind-N1]);
		// i = n1m1
		ind = j*N1+n1m1;
		der1[ind] = pot[ind] - pot[ind-1];
		der2[ind] = 0.5*(pot[ind+N1]-pot[ind-N1]);
	}
	for( j = 1; j < n2m1; j++ ) {
		// i = 0
		ind = N1*j;
		der12[ind] = 0.5*(der1[ind+N1] - der1[ind-N1]);
		// i = n1m1
		ind = j*N1+n1m1;
		der12[ind] = 0.5*(der1[ind+N1] - der1[ind-N1]);		
	}
	// borders j = 0 and j = n2m1
	for( i = 1; i < n1m1; i++ ) {
		// j = 0
		ind = i;
		der1[ind] = 0.5*(pot[ind+1]-pot[ind-1]);
		der2[ind] = pot[ind+N1]-pot[ind];
		// j = n2m1
		ind = n2m1*N1+i;
		der1[ind] = 0.5*(pot[ind+1]-pot[ind-1]);
		der2[ind] = pot[ind]-pot[ind-N1];
	}
	for( i = 1; i < n1m1; i++ ) {
		// j = 0
		ind = i;
		der12[ind] = der1[ind+N1] - der1[ind];
		// j = n2m1
		ind = n2m1*N1+i;
		der12[ind] = der1[ind]-der1[ind-N1];
	}
	// corners
	// i = 0; j = 0;
	der1[0] = pot[1] - pot[0];
	der2[1] = pot[N1] - pot[0];
	der12[0] = der1[N1] - der1[0];
	// i = n1m1; j = 0;
	der1[n1m1] = pot[n1m1] - pot[n1m1-1];
	der2[n1m1] = pot[N1+n1m1] - pot[n1m1];
	der12[n1m1] = der1[N1+n1m1] - der1[n1m1];
	// i = 0; j = n2m1;
	ind = N1*n2m1;
	der1[ind] = pot[ind+1] - pot[ind];
	der2[ind] = pot[ind] - pot[ind-N1];
	der12[ind] = der1[ind] - der1[ind-N1];
	// i = n1m1; j = n2m1;
	ind =  n1m1 + N1*n2m1;
	der1[ind] = pot[ind] - pot[ind-1];
	der2[ind] = pot[ind] - pot[ind-N1];
	der12[ind] = der1[ind] - der1[ind-N1];
}






//------------------------------------------------------------
int main(void){
	int Nbumps, Ngrid = N1*N2, Nsteps = NSTEPS, Natoms = NATOMS;
	int i,j,ind,n,dim=NATOMS*DIM;
	double h1,h2;
	double *height,*val1,*val2;
	double *val1_min,*val1_max,*val2_min,*val2_max;
	double *grid_CV1,*grid_CV2,*grid_pot,*grid_der1,*grid_der2,*grid_der12;
	double *bicubic_matrix;
	FILE *fpot,*fpar,*fconf;
    clock_t CPUbegin; // for measuring CPU time
    double cpu; // for recording CPU time
    char fpot_name[] = "GBumpsData/GaussianBumps_beta5.txt";
    double dt = TAU;

	val1 = (double *)malloc(NBUMPS_MAX*sizeof(double));
	val2 = (double *)malloc(NBUMPS_MAX*sizeof(double));
	height = (double *)malloc(NBUMPS_MAX*sizeof(double));
	val1_min = (double *)malloc(sizeof(double));
	val1_max = (double *)malloc(sizeof(double));
	val2_min = (double *)malloc(sizeof(double));
	val2_max = (double *)malloc(sizeof(double));
	
// read the metadynamics data for the Gauusian bumps
	fpot = fopen(fpot_name,"r");
	
	n = 0;
	*val1_min = INFTY;
	*val1_max = -INFTY;
	*val2_min = INFTY;
	*val2_max = -INFTY;
	while( !feof(fpot) && n < NBUMPS_MAX ) {
		fscanf( fpot,"%le\t%le\t%le\n",height+n,val1+n,val2+n);
		*val1_min = min(*val1_min,val1[n]);
		*val1_max = max(*val1_max,val1[n]);
		*val2_min = min(*val2_min,val2[n]);
		*val2_max = max(*val2_max,val2[n]);
		n++;
	}
	fclose(fpot);
	printf("The total number of bumps is n = %i\n",n);
	printf("val1_min = %.4e\n",*val1_min);
	printf("val1_max = %.4e\n",*val1_max);
	printf("val2_min = %.4e\n",*val2_min);
	printf("val2_max = %.4e\n",*val2_max);
	Nbumps = n;
	
// Compute data for constructing the bicubic spline
	grid_CV1 = (double *)malloc(N1*sizeof(double));
	grid_CV2 = (double *)malloc(N2*sizeof(double));
	h1 = (*val1_max-*val1_min)/(N1-1);
	h2 = (*val2_max-*val2_min)/(N2-1);
	
	// extend the domain
	*val1_min -= 5.0*h1;
	*val1_max += 5.0*h1;
	*val2_min -= 5.0*h2;
	*val2_max += 5.0*h2;
	
	h1 = (*val1_max-*val1_min)/(N1-1);
	h2 = (*val2_max-*val2_min)/(N2-1);
	printf("h1 = %.4e\n",h1);
	printf("h2 = %.4e\n",h2);
	
	for( i=0; i<N1; i++ ) {
		grid_CV1[i] = *val1_min + h1*i;		
	}
	for( i=0; i<N2; i++ ) {
		grid_CV2[i] = *val2_min + h2*i;		
	}
	printf("Computing WTMetad data on the grid");
	grid_pot = (double *)malloc(Ngrid*sizeof(double));
	grid_der1 = (double *)malloc(Ngrid*sizeof(double));
	grid_der2 = (double *)malloc(Ngrid*sizeof(double));
	grid_der12 = (double *)malloc(Ngrid*sizeof(double));

	Gpot_and_ders_on_grid(Nbumps,val1,val2,height,
		grid_CV1,grid_CV2,h1,h2,grid_pot,grid_der1,grid_der2,grid_der12,N1,N2);
	printf(" ... done!\n");
	

// Compute coefficient matrices for the bicubic spline	
	printf("Computing the bicubic matrix on the grid");

	bicubic_matrix = (double *)malloc(Ngrid*16*sizeof(double));	
	
	for( j = 0; j < N2; j++ ) {
		for( i=0; i < N1; i++ ) {
			ind = i + j*N1;
			compute_bicubic_coeff_matrix(grid_pot,grid_der1,grid_der2,grid_der12,
				bicubic_matrix,ind,N1,N2);
		}
	}
	printf(" ... done!\n");
	
	
	// save the parameters and the bicubic matrix 
	fpar = fopen("Data/bicubic_params.txt","w");
	fprintf(fpar,"%i\n",N1);
	fprintf(fpar,"%i\n",N2);
	fprintf(fpar,"%.10e\n",h1);
	fprintf(fpar,"%.10e\n",h2);
	fprintf(fpar,"%.10e\n",*val1_min);
	fprintf(fpar,"%.10e\n",*val1_max);
	fprintf(fpar,"%.10e\n",*val2_min);
	fprintf(fpar,"%.10e\n",*val2_max);
	fclose(fpar);
	
	// run a long trajectory in the biased potential and bin it
	long *bins;
	bins = (long *)malloc(Ngrid*sizeof(long));
	for( j=0; j<Ngrid; j++ ) bins[j] = 0;
	
	double *conf0,*bin_confs;
	conf0 = (double *)malloc(dim*sizeof(double));
	bin_confs = (double *)malloc(dim*Ngrid*sizeof(double));
	
	init_conf(conf0,0); 
	
 	CPUbegin=clock(); // start time measurement
 	binning_trajectory(bins,grid_CV1,grid_CV2,Nsteps,Natoms,
		conf0,dt,h1,h2,bicubic_matrix,bin_confs);
	cpu = (clock()-CPUbegin)/((double)CLOCKS_PER_SEC);	// end time measurement		
	printf("CPU time = %g\n",cpu);
	
	// record configurations
	fconf = fopen("Data/LJ7bins_confs.txt","w");
	for( i=0; i<N1; i++ ) {
		for( j=0; j<N2; j++ ) {
			ind = (i + j*N1)*dim;
			for( n = 0; n < dim; n++ ) {
				fprintf(fconf,"%.4e\t",bin_confs[ind + n]);
			}
			fprintf(fconf,"\n");
		}
	}
	fclose(fconf);

	// restore the invariant measure
	// FE = free energy, BP = biasing potential
	// bins[i,j] \propto exp( - beta*(FE[i,j] + BP[i,j]))
	// p[i,j] \propto exp( -beta*FE[i,j] ) \propto bins[i,j]*exp( beta BP[i,j])
	
	
	// record the free energy
	char fname[100];	
	sprintf(fname,"Data/LJ7bins_beta%.0f.txt",BETA);
	fpot = fopen(fname,"w");
	for( i=0; i<N1; i++ ) {
		for( j=0; j<N2; j++ ) {
			ind = i + j*N1;
			fprintf(fpot,"%li\t",bins[ind]);
		}
		fprintf(fpot,"\n");
	}
	fclose(fpot);
	
	
	double *FE,FEmin,FEmax;
	int *empty_bins_ind,empty_bins_count = 0;;
	
	FE = (double *)malloc(Ngrid*sizeof(double));
	empty_bins_ind = (int *)malloc(Ngrid*sizeof(int));
	
	FEmin = 1.0e12;
	FEmax = -1.0e12;
	for( j=0; j<Ngrid; j++ ) {
		if( bins[j] > 0 ) {
			FE[j] = -log((double)bins[j])/BETA - grid_pot[j];
			FEmin = min(FEmin,FE[j]);
			FEmax = max(FEmax,FE[j]);
		}
		else {
			empty_bins_ind[empty_bins_count] = j;
			empty_bins_count++;
		}
	}
	for( j=0; j<Ngrid; j++ ) FE[j] -= FEmin;
	FEmax -= FEmin;
	for( j=0; j < empty_bins_count; j++ ) {
		FE[empty_bins_ind[j]] = FEmax;
	}
	
	sprintf(fname,"Data/LJ7free_energy_beta%.0f.txt",BETA);
	fpot = fopen(fname,"w");
	for( j=0; j<N2; j++ ) {
		for( i=0; i<N1; i++ ) {
			ind = i + j*N1;
			fprintf(fpot,"%.4e\t",FE[ind]);
		}
		fprintf(fpot,"\n");
	}
	fclose(fpot);
	
	// smooth free energy
	sprintf(fname,"Data/LJ7free_energy_bicubic_matrix_beta%.0f.txt",BETA);
	fpot = fopen(fname,"w");
	FEders_on_grid(FE,grid_der1,grid_der2,grid_der12);
	for( j = 0; j < N2; j++ ) {
		for( i=0; i < N1; i++ ) {
			ind = i + j*N1;
			compute_bicubic_coeff_matrix(FE,grid_der1,grid_der2,grid_der12,
				bicubic_matrix,ind,N1,N2);
			for( n=0; n < 16; n++ ) {
				fprintf(fpot,"%.4e\t",bicubic_matrix[ind*16 + n]);
			}	
			fprintf(fpot,"\n");
		}
	}
	fclose(fpot);
	
	return 0;
}


