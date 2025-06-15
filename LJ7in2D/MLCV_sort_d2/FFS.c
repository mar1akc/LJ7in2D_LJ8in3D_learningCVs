// Calculates the escape rate from minimum 1 to minimum 0 in LJ8 in 3D 
// by the Forward Flux Sampling. The reaction coordinate is the committor in collective variables
// represented as a neural network.
// 
// The reaction coordinate approximates the committor and is defined by
// RC(x) = sigmoid(A3*w2 + b3)
// w2 = ReLU(A2*w1 + b2)
// w1 = ReLU(A1*CV + b1)

// MALA algorithm is used.


// Compile command:  gcc -Wall FFS.c CommittorRC.c MargotColVar.c timestep_helpers.c -lm -O3

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "MargotColVar.h"
#include "CommittorRC.h"
#include "timestep_helpers.h"
#include "macros_and_constants.h"

#define NEXIT 10000  
#define N_MSTONES 20
// #define MSTONE_MIN_VAL 3.0e-2 // 1.0e-3 for LJ7 in 2D
#define N_REPEAT 10

#define INFTY 1.0e6


// char PRINT = 'n';
int main(void);
void total_pot_and_grad(double *conf,int Natoms,double *pot,double *grad);
double exit_flux(double *init_conf,double *exit_conf,struct NNRC *nnrc,struct NN *nncv,
				 double dt,double *milestones, int Nmilestones);
double prob_next_milestone(double *init_conf,double *exit_conf,struct NNRC *nnrc,struct NN *nncv,
				 double dt,double *milestones, int mstone_idx);


//------------------------------------
void total_pot_and_grad(double *conf,int Natoms,double *pot,double *grad) {
	
	LJpot_and_grad(conf,pot,grad,Natoms);
	restraining_pot_and_grad(conf,pot,grad,Natoms);
			
}

//------------------------------------
double exit_flux(double *init_conf,double *exit_conf,struct NNRC *nnrc,struct NN *nncv,
				 double dt,double *milestones, int Nmilestones) {

	int exit_count = 0;
	int step = 0;
	double eflux;

	int j,k,dim = NATOMS*DIM,CVdim = nnrc->dim0;
	double *gauss01;
	int kshift;
	int Ngauss = 2*DIM*NATOMS, jgauss;
	double std = sqrt(2.0*dt/BETA);
	double *conf0,*conf1,*Vpot0,*Vpot1,*Vgrad0,*Vgrad1,*w;
	double *CVval,*CVprev;
	double RCval,RCprev;
	double last_milestone = milestones[Nmilestones-1];
	char ch;
	CVval = (double *)malloc(CVdim*sizeof(double));
	CVprev = (double *)malloc(CVdim*sizeof(double));

	conf0 = (double *)malloc(dim*sizeof(double));
	conf1 = (double *)malloc(dim*sizeof(double));
	Vgrad0 = (double *)malloc(dim*sizeof(double));
	Vgrad1 = (double *)malloc(dim*sizeof(double));
	Vpot0 = (double *)malloc(sizeof(double));
	Vpot1 = (double *)malloc(sizeof(double));
	w = (double *)malloc(dim*sizeof(double));
	gauss01 = (double *)malloc(Ngauss*sizeof(double));
	
	for( j = 0; j < dim; j++ ) conf0[j] = init_conf[j];
	
	// initialization	
	total_pot_and_grad(conf0,NATOMS,Vpot0,Vgrad0);
	
	MargotCV(CVval,conf0,NATOMS,nncv);
	RCval = reaction_coordinate(CVval,nnrc); 
		
	if( RCval >= milestones[0] ) {
		printf("The initial configuration lies outside A: RCval(conf0) = %.4e\n",RCval);
		exit(1);
	}
	// start simulation
	jgauss = 0;				
	while( exit_count < NEXIT ) {		
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
		total_pot_and_grad(conf1,NATOMS,Vpot1,Vgrad1);
		ch = MALAstep(conf0,conf1,NATOMS,dt,Vpot0,Vpot1,Vgrad1,w);
		if( ch == 1 ) { // step was accepted
			// align configurations
			align(conf0,conf1,NATOMS);
			for( k=0; k<dim; k++ ) {
				conf0[k] = conf1[k];
				Vgrad0[k] = Vgrad1[k];
			}
			*Vpot0 = *Vpot1;
		}
		// track transitions
		CVprev[0] = CVval[0];
		CVprev[1] = CVval[1];
		MargotCV(CVval,conf0,NATOMS,nncv);
		RCprev = RCval;
		RCval = reaction_coordinate(CVval,nnrc); 
		if( RCval >= milestones[0] ) {
			if( RCprev < milestones[0] ) {
				for( j = 0; j < dim; j++ ) { // record exit configuration
					exit_conf[exit_count*dim + j] = conf0[j];
				}
				exit_count++;
			}	
			if( RCval >= last_milestone ) {
				// reset to the initial configuration
				for( j = 0; j < dim; j++ ) {
					conf0[j] = init_conf[j];
				}	
				total_pot_and_grad(conf0,NATOMS,Vpot0,Vgrad0);	
				MargotCV(CVval,conf0,NATOMS,nncv);
				RCval = reaction_coordinate(CVval,nnrc); 	
				RCprev = RCval;
			}
		}
		step++;
	}
	printf("Finished computing exit flux\n");
	printf("step = %i, time = %.4e, exit_count = %i\n",step,dt*step,exit_count);
	eflux = exit_count/(dt*step);
	printf("Exit flux = %.4e\n",eflux);
	
	free(CVval);
	free(CVprev);
	free(conf0);
	free(conf1);
	free(Vgrad0);
	free(Vgrad1);
	free(Vpot0);
	free(Vpot1);
	free(w);
	
	return eflux;	
}

//------------------------------------
double prob_next_milestone(double *init_conf,double *exit_conf,struct NNRC *nnrc,struct NN *nncv,
				 double dt,double *milestones, int mstone_idx) {

	int next_mstone_count = 0,event_count = 0;
	int step = 0;
	double prob_next_mstone;
	int iconf_idx;

	int j,k,dim = NATOMS*DIM,CVdim = nnrc->dim0;;
	double *gauss01;
	int kshift;
	int Ngauss = 2*DIM*NATOMS, jgauss;
	double std = sqrt(2.0*dt/BETA);
	double *conf0,*conf1,*Vpot0,*Vpot1,*Vgrad0,*Vgrad1,*w;
	char ch;
	double *CVval;
	double RCval;
	double mstone0 = MSTONE_MIN_VAL,next_mstone = milestones[mstone_idx];
	
	CVval = (double *)malloc(CVdim*sizeof(double));
// 	CVprev = (double *)malloc(dim4*sizeof(double));

	conf0 = (double *)malloc(dim*sizeof(double));
	conf1 = (double *)malloc(dim*sizeof(double));
	Vgrad0 = (double *)malloc(dim*sizeof(double));
	Vgrad1 = (double *)malloc(dim*sizeof(double));
	Vpot0 = (double *)malloc(sizeof(double));
	Vpot1 = (double *)malloc(sizeof(double));
	w = (double *)malloc(dim*sizeof(double));
	gauss01 = (double *)malloc(Ngauss*sizeof(double));
	
	iconf_idx = random()%NEXIT;
// 	printf("iconf_idx = %i\n",iconf_idx);
	for( j = 0; j < dim; j++ ) {
		conf0[j] = init_conf[iconf_idx*dim + j];
// 		printf("conf0[%i] = %.4e\n",j,conf0[j]);
	}
	// initialization	
	total_pot_and_grad(conf0,NATOMS,Vpot0,Vgrad0);
	
	MargotCV(CVval,conf0,NATOMS,nncv);
	RCval = reaction_coordinate(CVval,nnrc); 
		
	// start simulation
	jgauss = 0;				
	while( next_mstone_count < NEXIT ) {		
		// generate array of random vars N(0,std) of size 2*Natoms
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
		total_pot_and_grad(conf1,NATOMS,Vpot1,Vgrad1);
		ch = MALAstep(conf0,conf1,NATOMS,dt,Vpot0,Vpot1,Vgrad1,w);
// 		printf("Vpot0 = %.10e, Vpot1 = %.10e\n",*Vpot0,*Vpot1);
		if( ch == 1 ) { // step was accepted
			// align configurations
			align(conf0,conf1,NATOMS);
			for( k=0; k<dim; k++ ) {
				conf0[k] = conf1[k];
				Vgrad0[k] = Vgrad1[k];
			}
			*Vpot0 = *Vpot1;
		}
		// track transitions
// 		CVprev[0] = CVval[0];
// 		CVprev[1] = CVval[1];
		MargotCV(CVval,conf0,NATOMS,nncv);
// 		RCprev = RCval;
		RCval = reaction_coordinate(CVval,nnrc); 
		if( RCval >= next_mstone ) {
			for( j = 0; j < dim; j++ ) { // record exit configuration
				exit_conf[next_mstone_count*dim + j] = conf0[j];
			}
			next_mstone_count++;
// 			printf("next_mstone_count = %i\n",next_mstone_count);
		}
		if( RCval >= next_mstone || RCval <= mstone0 ) {
			event_count++;
			// reset to the initial configuration
// 			printf("before reset: CV0 = %.4e, CV1 = %.4e, RCval = %.4e\n",CVval[0],CVval[1],RCval);		
			iconf_idx = random()%NEXIT;
			for( j = 0; j < dim; j++ ) conf0[j] = init_conf[iconf_idx*dim + j];
			total_pot_and_grad(conf0,NATOMS,Vpot0,Vgrad0);	
			MargotCV(CVval,conf0,NATOMS,nncv);
// 			PRINT = 'y';
			RCval = reaction_coordinate(CVval,nnrc); 	
// 			printf("reset: iconf_idx = %i, CV0 = %.4e, CV1 = %.4e, RCval = %.4e\n",
// 					iconf_idx,CVval[0],CVval[1],RCval);	
// 			exit(1);	
		}
// 		printf("step = %i, ch = %i, RCval = %.4e\n",step,ch,RCval);				
		step++;
	}
	printf("Finished computing P(milestone_next|milestone_current)\n");
	printf("step = %i, time = %.4e, exit_count = %i\n",step,dt*step,next_mstone_count);
	prob_next_mstone = (double)next_mstone_count/(double)event_count;
	printf("P(milestone_next|milestone_current) = %.4e\n",prob_next_mstone);
	
	free(CVval);
// 	free(CVprev);
	free(conf0);
	free(conf1);
	free(Vgrad0);
	free(Vgrad1);
	free(Vpot0);
	free(Vpot1);
	free(w);
	
// 	exit(1);
	return prob_next_mstone;	
}

//------------------------------------------------------------
int main(void){
// 	int run_id = 1;
	int k;
	int j,dim=NATOMS*DIM;
	FILE *fid;
    clock_t CPUbegin; // for measuring CPU time
    double cpu; // for recording CPU time
    double dt = TAU;
    char fname[100];
    
    // Margot's CV defined on the vector of sorted coordination numbers
	char fCVname[] = "MargotCV_NNdata.txt";
	char fname_dim[] = "MargotCV_dimensions.txt";
	struct NN *nncv;	
	nncv = (struct NN *)malloc(sizeof(struct NN));
	readCVdata(fCVname,fname_dim,nncv);

    // NN for the reaction coordinate
	char fRCname[100];
	sprintf(fRCname,"FEMdataBETA%i/Committor_NN_BETA%i/RC_NNdata.txt",
			(int)round(BETA),(int)round(BETA)); 
	char fRCname_dim[100]; 
	sprintf(fRCname_dim,"FEMdataBETA%i/Committor_NN_BETA%i/RC_dimensions.txt",
			(int)round(BETA),(int)round(BETA)); 
	struct NNRC *nnrc;	
	nnrc = (struct NNRC *)malloc(sizeof(struct NNRC));
	readRCdata(fRCname,fRCname_dim,nnrc);
	
	printf("Back to FFS\n");
	printf("dim = %i, CVdim = %i\n",dim,CVDIM);
	
	double *conf0,*CV,RCval;	
	conf0 = (double *)malloc(dim*sizeof(double));
	CV = (double *)malloc(CVDIM*sizeof(double));

	for( k = 0; k < N_MINIMA; k++ ) {
		init_conf(conf0,k); // initial minimum is minimum 1
		MargotCV(CV,conf0,NATOMS,nncv);
		RCval = reaction_coordinate(CV,nnrc);
		printf("MIN %i: RCval = %.4e\n",k,RCval);
	}
//    	exit(1);
		
	double *milestones,*eflux_and_pvector;
	int Nmilestones = N_MSTONES;
	double *conf_data1, *conf_data2;
	int mstone_idx;
	double mstone_step = (MSTONE_MAX_VAL-MSTONE_MIN_VAL)/(N_MSTONES - 1);
	double esc_rate = 1.;
	double esc_rate_list[N_REPEAT];
		
	srand(time(NULL));
	
	
	conf_data1 = (double *)malloc(NEXIT*dim*sizeof(double));
	conf_data2 = (double *)malloc(NEXIT*dim*sizeof(double));
	milestones = (double *)malloc(Nmilestones*sizeof(double));
	eflux_and_pvector = (double *)malloc(Nmilestones*sizeof(double));
	sprintf(fname,"Data/FFS&BF/FFS_fromM%itoM%i_BETA%i.txt",STARTING_MIN,FINISH_MIN,(int)round(BETA));
	printf("%s\n",fname);
	
	// uniform milestones
	milestones[0] = MSTONE_MIN_VAL;
	milestones[Nmilestones-1] = MSTONE_MAX_VAL;
	for( j = 1; j < Nmilestones; j++ ) {
		milestones[j] = MSTONE_MIN_VAL + j*mstone_step;
	}
	// make log scale in milestones
// 	milestones[Nmilestones/2] = 0.6;
// 	mstone_step = (log(milestones[Nmilestones/2]) - log(milestones[0]))/(Nmilestones/2);
// 	
// 	
// 	for( j = 1; j < Nmilestones/2; j++ ) {
// 		milestones[j] = exp(log(MSTONE_MIN_VAL) + j*mstone_step);
// 	}
// 	mstone_step = (milestones[Nmilestones-1] - milestones[Nmilestones/2])/(Nmilestones - Nmilestones/2 - 1);
// 	for( j = 1 + Nmilestones/2; j < Nmilestones; j++ ) {
// 		milestones[j] = milestones[Nmilestones/2] + (j-Nmilestones/2)*mstone_step;
// 	}
	
	fid = fopen(fname,"w");
	for( k = 0; k < N_MINIMA; k++ ) {
		init_conf(conf0,k); // initial minimum is minimum 1
		MargotCV(CV,conf0,NATOMS,nncv);
		RCval = reaction_coordinate(CV,nnrc);
		printf("MIN %i: RCval = %.4e\n",k,RCval);
		fprintf(fid,"MIN %i: RCval = %.4e\n",k,RCval);
	}
	
	printf("BETA = %.2f\n",BETA);
	fprintf(fid,"BETA = %.2f\n",BETA);
	fprintf(fid,"NEXIT = %i\n",NEXIT);
	fprintf(fid,"Milestones:\n");
	for( j = 0; j < Nmilestones; j++ ) {
		fprintf(fid,"%.4e\n",milestones[j]);
		printf("%.4e\n",milestones[j]);
	}


	
	for( k = 0; k < N_REPEAT; k++ ) {
		printf("Run %i\n", k);
		fprintf(fid,"Run %i: from min %i to min %i\n", k,STARTING_MIN,FINISH_MIN);
		CPUbegin=clock(); // start time measurement
		
		init_conf(conf0,STARTING_MIN); // initial minimum is minimum 1
		eflux_and_pvector[0] = exit_flux(conf0,conf_data1,nnrc,nncv,dt,milestones,Nmilestones);
		fprintf(fid,"0\t%.4e\n",eflux_and_pvector[0]);
		for( mstone_idx = 1; mstone_idx < Nmilestones; mstone_idx++ ) {
			if( mstone_idx%2 == 1 ) {
				printf("Prob(mstone[%i]|mstone[%i]):\n",mstone_idx,mstone_idx-1);
				eflux_and_pvector[mstone_idx] = prob_next_milestone(conf_data1,conf_data2,
												nnrc,nncv,dt,milestones,mstone_idx);
			}
			else {
				printf("Prob(mstone[%i]|mstone[%i]):\n",mstone_idx,mstone_idx-1);
				eflux_and_pvector[mstone_idx] = prob_next_milestone(conf_data2,conf_data1,
												nnrc,nncv,dt,milestones,mstone_idx); 		
			}
			fprintf(fid,"%i\t%.4e\n",mstone_idx,eflux_and_pvector[mstone_idx]);
		}
		esc_rate = 1.;
		for( j = 0; j < Nmilestones; j++ ) esc_rate*=eflux_and_pvector[j];
		fprintf(fid,"Escape_rate = %.4e\n",esc_rate); 	
		printf("Escape_rate = %.4e\n",esc_rate); 	
		esc_rate_list[k] = esc_rate;
		cpu = (clock()-CPUbegin)/((double)CLOCKS_PER_SEC);	// end time measurement
	// 	fclose(f_events);

		printf("CPU time = %g\n",cpu);
		fprintf(fid,"CPU time = %g\n",cpu);
	}	
	double erate_mean = 0.,erate_std = 0.;
	
	for( k = 0; k < N_REPEAT; k++ ) {
		printf("%.4e\n",esc_rate_list[k]);
		erate_mean += esc_rate_list[k];
	}	
	erate_mean /= N_REPEAT;
	printf("Mean escape rate = %.4e\n",erate_mean);
	for( k = 0; k < N_REPEAT; k++ ) {
		erate_std += pow(esc_rate_list[k] - erate_mean,2);
	}
	erate_std /= (N_REPEAT - 1);
	erate_std = sqrt(erate_std);
	printf("Standard deviation = %.4e\n",erate_std);
	
	fprintf(fid,"Escape rate vector:\n");
	for( k = 0; k < N_REPEAT; k++ ) fprintf(fid,"%.4e\n",esc_rate_list[k]);
	fprintf(fid,"Mean escape rate = %.4e\n",erate_mean);
	fprintf(fid,"Standard deviation = %.4e\n",erate_std);
	fclose(fid);
	return 0;
}


