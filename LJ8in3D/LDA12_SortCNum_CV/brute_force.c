// Calculates the transition rate between the hexagon and the trapezoid in LJ7 in 2D 
// by brute force. 
// Calculates the escape rate from minimum 1 to minimum 0 in LJ8 in 3D 
// by the Forward Flux Sampling. The reaction coordinate is the committor in collective variables
// represented as a neural network.
// 
// Sets A and B are, respectively:
// NN(x) = sigmoid(A3*w2 + b3)
// w2 = ReLU(A2*w1 + b2)
// w1 = ReLU(A1*CV + b1)
// Q = (1 - chiA)*(NN(x)*(1-chiB) + chiB)
// 
// A = [1.5607, 1.5657]
// B = [-0.8985, 0.5363]
// rA = 1.5
// rB = 1.0
// 
// chiA(x) = 0.5 - 0.5*tanh[0.5(dist(x, A)^2 - (rA + 0.3)^2]
// chiB(x) = 0.5 - 0.5*tanh[0.5(dist(x, B)^2 - (rB + 0.2)^2]
//
// MALA algorithm is used.

// Compile command:  gcc -Wall brute_force.c CommittorRC.c LDA_CoordNum_CV.c timestep_helpers.c -lm -O3

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "LDA_CoordNum_CV.h"
#include "CommittorRC.h"
#include "timestep_helpers.h"
#include "macros_and_constants.h"

#define NSTEPS 1e9  // the length of the stochastic trajectory that we bin

#define INFTY 1.0e6

void run_trajectory(int Nsteps,int Natoms,double *conf0,double dt,double t_start,
	int *AB_event_counter,struct NNRC *nnrc,double *LDAmatrix);
void total_pot_and_grad(double *conf,int Natoms,double *pot,double *grad);
char Acondition(double RCval);
char Bcondition(double RCval);
//----- main	
int main(void);

//-------------------------------------------------------------
// Evaluate the total potential energy and its gradient

void total_pot_and_grad(double *conf,int Natoms,double *pot,double *grad) {
	
	LJpot_and_grad(conf,pot,grad,Natoms);
	restraining_pot_and_grad(conf,pot,grad,Natoms);
		
}

char Acondition(double RCval) {
	if( RCval < MSTONE_MIN_VAL)
	{
		return 1;
	}
	else {
		return 0;
	}
}

char Bcondition(double RCval) {
    if( RCval >= MSTONE_MAX_VAL ) {
    	return 1;
    }
    else {
    	return 0;
    }
}


//------------------------------------------------------------
void run_trajectory(int Nsteps,int Natoms,double *conf0,double dt,double t_start,
	int *AB_event_counter,struct NNRC *nnrc,double *LDAmatrix) {

	int j,k,dim = Natoms*DIM,CVdim = nnrc->dim0;
	double *gauss01;
	int kshift;
	int Ngauss = 2*DIM*NATOMS, jgauss;
	double std = sqrt(2.0*dt/BETA);
	double *conf1,*Vpot0,*Vpot1,*Vgrad0,*Vgrad1,*w;
	char ch;
	char AB_event;
	double *CVval;
	double TA = 0.0, TB = 0.0, T_OmegaAB = 0.0,TlastA = 0.0,TlastB = 0.0;
	double RCval;
	
	CVval = (double *)malloc(CVdim*sizeof(double));

	conf1 = (double *)malloc(dim*sizeof(double));
	Vgrad0 = (double *)malloc(dim*sizeof(double));
	Vgrad1 = (double *)malloc(dim*sizeof(double));
	Vpot0 = (double *)malloc(sizeof(double));
	Vpot1 = (double *)malloc(sizeof(double));
	w = (double *)malloc(dim*sizeof(double));
	gauss01 = (double *)malloc(Ngauss*sizeof(double));
	
	// initialization	
	total_pot_and_grad(conf0,NATOMS,Vpot0,Vgrad0);
	
	LDA_CNum_CV(CVval,conf0,Natoms,CVdim,LDAmatrix);
	RCval = reaction_coordinate(CVval,nnrc); 
		
	printf("In run_trajectory\n");	
		
	if( Acondition(RCval) == 1 ) AB_event = 'A';
	else{ // not in B
		if( Bcondition(RCval) == 1 ) AB_event = 'B';
		else AB_event = 0;
	}
	printf("In run_trajectory, RCval = %.4e\n",RCval);	

	// start the trajectory
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
		total_pot_and_grad(conf1,Natoms,Vpot1,Vgrad1);
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
		// track transitions
		LDA_CNum_CV(CVval,conf0,Natoms,CVdim,LDAmatrix);
		RCval = reaction_coordinate(CVval,nnrc); 
// 		printf("In run_trajectory, j = %i, RCval = %.4e\n",j,RCval);	
		
		//-------
		
		switch(AB_event) {
			case 0:
				if( Acondition(RCval) == 1 ) {
					AB_event = 'A';
					TA +=dt;
				}
				else {
					if( Bcondition(RCval) == 1 ) {
						AB_event = 'B';
						TB += dt;
					}
					else T_OmegaAB += dt;				
				}
				break;
			case 'A':
				TlastA += dt;
				if( Acondition(RCval) == 1 ) {
					AB_event = 'A';
					TA +=dt;
				}
				else {
					if( Bcondition(RCval) == 1 ) {
						AB_event = 'B';
						TB += dt;
						(*AB_event_counter)++;
					}
					else T_OmegaAB += dt;				
				}
				break;
			case 'B':
				TlastB += dt;
				if( Acondition(RCval) == 1 ) {
					AB_event = 'A';
					TA +=dt;
				}
				else {
					if( Bcondition(RCval) == 1 ) {
						AB_event = 'B';
						TB += dt;
					}
					else T_OmegaAB += dt;				
				}
				break;
			default:
				printf("AB_event = %c\n",AB_event);
				exit(1);
				break;
		}
				
	}
	printf("Done! Beta = %.1f\n",BETA);
	printf("TinA = %.4e, TinB = %.4e, T_OmegaAB = %.4e, rateAB = %.4e\n",
		TA,TB,T_OmegaAB,(*AB_event_counter)/(j*dt));
	printf("TlastA(j) = %.4e; TlastB(j) = %.4e;\n",TlastA,TlastB);
	printf("rhoA(j) = %.4e; rhoB(j) = %.4e;\n",TlastA/(Nsteps*dt),TlastB/(Nsteps*dt) );
	printf("kA(j) = %.4e; kB(j) = %.4e;\n",(*AB_event_counter)/TlastA,(*AB_event_counter)/TlastB );
	
	free(CVval);
	free(conf1);
	free(Vgrad0);
	free(Vgrad1);
	free(Vpot0);
	free(Vpot1);
	free(w);
	
}


//------------------------------------------------------------
int main(void){
	int Nsteps = NSTEPS;
	int n;
	int dim=NATOMS*DIM;
	int Natoms = NATOMS;
// 	FILE *fid;
    clock_t CPUbegin; // for measuring CPU time
    double cpu; // for recording CPU time
    double dt,t_start;
    int *AB_event_counter;
//     char fname[100];
    
    // LDA CV
	char fCVname[] = "Data/LDAbasis.txt";
	double *LDAmatrix;
	int CVdim = CVDIM;
	LDAmatrix = (double *)malloc(CVdim*Natoms*sizeof(double));
	readLDAmatrix(fCVname,Natoms,CVdim,LDAmatrix);

    // NN for the reaction coordinate
// 	char fRCname[100];
// 	sprintf(fRCname,"FEMdataBETA%i/Committor_min01_LDA12_BETA%i/RC_NNdata.txt",
// 			(int)round(BETA),(int)round(BETA)); 
// 	char fRCname_dim[100]; 
// 	sprintf(fRCname_dim,"FEMdataBETA%i/Committor_min01_LDA12_BETA%i/RC_dimensions.txt",
// 			(int)round(BETA),(int)round(BETA)); 
	char fRCname[] = "FEMdataBETA15/Committor_min01_LDA12_BETA15/RC_NNdata.txt";
	char fRCname_dim[] ="FEMdataBETA15/Committor_min01_LDA12_BETA15/RC_dimensions.txt";
	struct NNRC *nnrc;	
	nnrc = (struct NNRC *)malloc(sizeof(struct NNRC));
	readRCdata(fRCname,fRCname_dim,nnrc);
				
	srand(time(NULL));
	
	double *conf0;
	dt = TAU;
	
	conf0 = (double *)malloc(dim*sizeof(double));
	init_conf(conf0,1); 
	dt = TAU;
	
	
// 	sprintf(fname,"UnbiasedTrajectoryData/LJ7_events_beta%.0f.txt",BETA);
// 	f_events = fopen(fname,"w");
	
 	CPUbegin=clock(); // start time measurement
 	
	t_start = 0.0;
	AB_event_counter = (int *)malloc(sizeof(int));
	for( n = 0; n < 10; n++ ) {
		printf("Beta = %.1f, Run # %i\n",BETA,n);
		*AB_event_counter = 0;
 		run_trajectory(Nsteps,NATOMS,conf0,dt,t_start,AB_event_counter,nnrc,LDAmatrix);
		printf("AB_event_counter(j) = %i;\n",*AB_event_counter);	
	}		
	cpu = (clock()-CPUbegin)/((double)CLOCKS_PER_SEC);	// end time measurement
// 	fclose(f_events);
	printf("CPU time = %g\n",cpu);	
	
	return 0;
}


