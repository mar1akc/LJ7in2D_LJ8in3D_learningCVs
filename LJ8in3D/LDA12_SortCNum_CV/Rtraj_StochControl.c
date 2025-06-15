// Samples stochastic trajectories using stochastic optimal controll.
// The controller is 2\beta^{-1}\nabla log(q(CV1(x),CV2(x))). 
// The committor in collective variables is represented as a neural network.
// 
// Sets A and B are, respectively:
// A = { (CV1,CV2) | F(CV1,CV2) < 0.6 } // the hexagon configuration
// B = { (CV1,CV2) | (CV1,CV2) \in ellipse } // the trapezoid configuration
// The ellipse defining the region B has the following parameters
// x0 =  4.82226  y0 =  2.55284
// r0 =  1.6441554343948148  r1 =  0.4
// vec0 =  0.5692071323806902 0.8221941622554562
// vec1 =  -0.8221941622554562 0.5692071323806902


// The free energy is approximated by a bicubic interpolant
//
// MALA algorithm is used.
// The phase space is 14D.
//
// The trajectory is saved into files.
// The exit and entrance times to A and B are saved into a file.

// Compile command:  gcc -Wall Rtraj_StochControl.c  CommittorRC.c LDA_CoordNum_CV.c timestep_helpers.c -lm -O3

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "LDA_CoordNum_CV.h"
#include "CommittorRC.h"
#include "timestep_helpers.h"
#include "macros_and_constants.h"

#define NEXIT 10000

#define INFTY 1.0e6


// char PRINT = 'n';
int main(void);
void exit_confs(double *init_conf,double *exit_conf,struct NNRC *nnrc,double *LDAmatrix,double dt);
void generate_reactive_trajectories(double *init_conf,double *exit_conf,
					struct NNRC *nnrc,double *LDAmatrix,double dt,double *crossover_time,
					long *bins,double *grid_CV1,double *grid_CV2,double h1,double h2,
					int N1,int N2);
void total_pot_and_grad(double *conf,int Natoms,double *pot,double *grad);
void controlled_pot_and_grad(double *conf,struct NNRC *nnrc,double *LDAmatrix,
							double *pot,double *grad,double *CVval,double *q);
//------------------------------------

void total_pot_and_grad(double *conf,int Natoms,double *pot,double *grad) {
	
	LJpot_and_grad(conf,pot,grad,Natoms);
	restraining_pot_and_grad(conf,pot,grad,Natoms);
			
}
//------------------------------------

void exit_confs(double *init_conf,double *exit_conf,struct NNRC *nnrc,double *LDAmatrix,double dt) {

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
	char ch;
	double mstone0 = MSTONE_MIN_VAL,mstone1 = MSTONE_MAX_VAL;

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
	
	LDA_CNum_CV(CVval,conf0,NATOMS,CVdim,LDAmatrix);
	RCval = reaction_coordinate(CVval,nnrc); 
		
	if( RCval >= mstone0 ) {
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
		LDA_CNum_CV(CVval,conf0,NATOMS,CVdim,LDAmatrix);
		RCprev = RCval;
		RCval = reaction_coordinate(CVval,nnrc); 
		if( RCval >= mstone0 ) {
			if( RCprev < mstone0 ) {
				for( j = 0; j < dim; j++ ) { // record exit configuration
					exit_conf[exit_count*dim + j] = conf0[j];
				}
				exit_count++;
			}	
			if( RCval >= mstone1 ) {
				// reset to the initial configuration
				for( j = 0; j < dim; j++ ) {
					conf0[j] = init_conf[j];
				}	
				total_pot_and_grad(conf0,NATOMS,Vpot0,Vgrad0);	
				LDA_CNum_CV(CVval,conf0,NATOMS,CVdim,LDAmatrix);
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
	
}

//------------------------------------
void controlled_pot_and_grad(double *conf,struct NNRC *nnrc,double *LDAmatrix,
							double *pot,double *grad,double *CVval,double *q) {
	int j,dim = NATOMS*DIM,CVdim = nnrc->dim0;
	double *qgrad,*CVgrad;
	double binv2 = 2.0/BETA;

	qgrad = (double *)malloc(CVdim*sizeof(double));
	CVgrad = (double *)malloc(CVdim*DIM*sizeof(double));

	LJpot_and_grad(conf,pot,grad,NATOMS);
	restraining_pot_and_grad(conf,pot,grad,NATOMS);
	
// 	void MargotCVgrad(double *grad,double *cvval,double *conf,int Natoms,struct NN *nncv);
// 	void reaction_coordinate_grad(double *RCval,double *RCgrad,double *CV,struct NNRC *nnrc);

	LDA_CNum_CVgrad(CVgrad,CVval,conf,NATOMS,CVdim,LDAmatrix);
	reaction_coordinate_grad(q,qgrad,CVval,nnrc);
	// add the effect of the biasing potential
	*pot -= binv2*log(*q);
	for( j = 0; j < dim; j++ ) {
		grad[j] -= binv2*(qgrad[0]*CVgrad[j] + qgrad[1]*CVgrad[j+dim])/(*q);
	}
	
	free(qgrad);
	free(CVgrad);
}




//------------------------------------
void generate_reactive_trajectories(double *init_conf,double *exit_conf,
					struct NNRC *nnrc,double *LDAmatrix,double dt,double *crossover_time,
					long *bins,double *grid_CV1,double *grid_CV2,double h1,double h2,
					int N1,int N2) {
	printf("In generate_reactive_trajectories\n");
	int crossover_count = 0,return_count = 0,event_count = 0;
	long step = 0;
	int iconf_idx;
	int j,k,dim = NATOMS*DIM,CVdim = nnrc->dim0;
	double *gauss01;
	int kshift;
	int Ngauss = 2*DIM*NATOMS, jgauss;
	double std = sqrt(2.0*dt/BETA);
	double *conf0,*conf1,*Vpot0,*Vpot1,*Vgrad0,*Vgrad1,*w;
	char ch;
	double *CVval,*q,CVsave[2],qsave;
	double mstone0 = MSTONE_MIN_VAL,mstone1 = MSTONE_MAX_VAL;
	int j1,j2,ind;

	long *bins_tmp;
	int Ngrid = N1*N2;
	
	CVval = (double *)malloc(CVdim*sizeof(double));
	conf0 = (double *)malloc(dim*sizeof(double));
	conf1 = (double *)malloc(dim*sizeof(double));
	Vgrad0 = (double *)malloc(dim*sizeof(double));
	Vgrad1 = (double *)malloc(dim*sizeof(double));
	Vpot0 = (double *)malloc(sizeof(double));
	Vpot1 = (double *)malloc(sizeof(double));
	w = (double *)malloc(dim*sizeof(double));
	q = (double *)malloc(sizeof(double));
	gauss01 = (double *)malloc(Ngauss*sizeof(double));
	bins_tmp = (long *)malloc(Ngrid*sizeof(long));

	for( j=0; j<Ngrid; j++ ) bins_tmp[j] = 0;

	iconf_idx = random()%NEXIT;
// 	printf("iconf_idx = %i\n",iconf_idx);
	for( j = 0; j < dim; j++ ) {
		conf0[j] = init_conf[iconf_idx*dim + j];
// 		printf("conf0[%i] = %.4e\n",j,conf0[j]);
	}
	// initialization	
// 	total_pot_and_grad(conf0,NATOMS,Vpot0,Vgrad0);
// 	MargotCV(CVval,conf0,NATOMS,A1,A2,A3,b1,b2,b3,dim1,dim2,dim3,dim4);
// 	RCval = committor(CVval,nnrc); 
	controlled_pot_and_grad(conf0,nnrc,LDAmatrix,Vpot0,Vgrad0,CVval,q);
	CVsave[0] = CVval[0];
	CVsave[1] = CVval[1];
	qsave = *q;	
	// start simulation
	jgauss = 0;				
	while( crossover_count < NEXIT ) {		
		// generate array of random vars N(0,std) of size 2*Natoms
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
// 		total_pot_and_grad(conf1,NATOMS,Vpot1,Vgrad1);
		controlled_pot_and_grad(conf1,nnrc,LDAmatrix,Vpot1,Vgrad1,CVval,q);
		ch = MALAstep(conf0,conf1,NATOMS,dt,Vpot0,Vpot1,Vgrad1,w);
// 		printf("Vpot0 = %.10e, Vpot1 = %.10e\n",*Vpot0,*Vpot1);
		step++;
// 		printf("step = %li\n",step);
		if( ch == 1 ) { // step was accepted
			// align configurations
			align(conf0,conf1,NATOMS);
			for( k=0; k<dim; k++ ) {
				conf0[k] = conf1[k];
				Vgrad0[k] = Vgrad1[k];
			}
			*Vpot0 = *Vpot1;
			CVsave[0] = CVval[0];
			CVsave[1] = CVval[1];
			qsave = *q;				
		}
		// bin the trajectory
		j1 = min(max(0,(int)floor((CVsave[0] - grid_CV1[0])/h1 + 0.5)),N1-1);
		j2 = min(max(0,(int)floor((CVsave[1] - grid_CV2[0])/h2 + 0.5)),N2-1);
		ind = j1 + j2*N1;
		bins_tmp[ind]++;
		if( qsave >= mstone1 ) {
			for( j = 0; j < dim; j++ ) { // record exit configuration
				exit_conf[crossover_count*dim + j] = conf0[j];
			}
			crossover_time[crossover_count] = step*dt;
			crossover_count++;
			printf("crossover #%i: %li steps\n",crossover_count,step);
			for( j=0; j<Ngrid; j++ ) bins[j] += bins_tmp[j];
		}
		else if( qsave <= mstone0 ) {
			return_count++;
		}
		if( qsave >= mstone1 || qsave <= mstone0 ) {
			event_count++;
			// reset to the initial configuration
// 			printf("before reset: CV0 = %.4e, CV1 = %.4e, RCval = %.4e\n",CVval[0],CVval[1],RCval);		
			iconf_idx = random()%NEXIT;
			for( j = 0; j < dim; j++ ) conf0[j] = init_conf[iconf_idx*dim + j];
			controlled_pot_and_grad(conf0,nnrc,LDAmatrix,Vpot0,Vgrad0,CVval,q);	
			CVsave[0] = CVval[0];
			CVsave[1] = CVval[1];
			qsave = *q;
			step = 0;	
			for( j=0; j<Ngrid; j++ ) bins_tmp[j] = 0;
		}

	}
	
	free(CVval);
	free(conf0);
	free(conf1);
	free(Vgrad0);
	free(Vgrad1);
	free(Vpot0);
	free(Vpot1);
	free(w);
	free(q);
	printf("crossover_count = %i, return_count = %i, event_count = %i\n",
				crossover_count,return_count,event_count);				

	free(bins_tmp);
	
}

//------------------------------------------------------------
int main(void){
	int run_id = 0;
	int j,dim=NATOMS*DIM,Natoms = NATOMS;
// 	double h1,h2;
// 	double *val1_min,*val1_max,*val2_min,*val2_max;
// 	double *grid_CV1,*grid_CV2;
// 	double *bicubic_matrix;
	FILE *fid;
    clock_t CPUbegin; // for measuring CPU time
    double cpu; // for recording CPU time
    double dt = TAU;
    char fname[100];
    // LDA CV
	char fCVname[] = "Data/LDAbasis.txt";
	double *LDAmatrix;
	int CVdim = CVDIM;
	LDAmatrix = (double *)malloc(CVdim*Natoms*sizeof(double));
	readLDAmatrix(fCVname,Natoms,CVdim,LDAmatrix);

    // NN for the reaction coordinate
	char fRCname_dim[100]; 
	double beta = min(BETA,15.0);
	sprintf(fRCname_dim,"FEMdataBETA%i/Committor_min01_LDA12_BETA%i/RC_dimensions.txt",
			(int)round(beta),(int)round(beta)); 
	char fRCname[100];
	sprintf(fRCname,"FEMdataBETA%i/Committor_min01_LDA12_BETA%i/RC_NNdata.txt",
			(int)round(beta),(int)round(beta)); 
	struct NNRC *nnrc;	
	nnrc = (struct NNRC *)malloc(sizeof(struct NNRC));
	readRCdata(fRCname,fRCname_dim,nnrc);
	
	double *A_conf, *B_conf;
	double *crossover_time;
	double *conf0;
	
	// bin transition trajectories
	FILE *fpar;
	int N1,N2;
	double h1,h2;
	double *val1_min,*val1_max,*val2_min,*val2_max;
	double *grid_CV1,*grid_CV2;

	val1_min = (double *)malloc(sizeof(double));
	val1_max = (double *)malloc(sizeof(double));
	val2_min = (double *)malloc(sizeof(double));
	val2_max = (double *)malloc(sizeof(double));

	fpar = fopen("Data/bicubic_params.txt","r");
	fscanf(fpar,"%i\n",&N1);
	fscanf(fpar,"%i\n",&N2);
	fscanf(fpar,"%le\n",&h1);
	fscanf(fpar,"%le\n",&h2);
	fscanf(fpar,"%le\n",val1_min);
	fscanf(fpar,"%le\n",val1_max);
	fscanf(fpar,"%le\n",val2_min);
	fscanf(fpar,"%le\n",val2_max);
	fclose(fpar);
	// print read values
	printf("val1_min = %.4e\n",*val1_min);
	printf("val1_max = %.4e\n",*val1_max);
	printf("val2_min = %.4e\n",*val2_min);
	printf("val2_max = %.4e\n",*val2_max);
	printf("h1 = %.4e\n",h1);
	printf("h2 = %.4e\n",h2);
	printf("N1 = %i\n",N1);
	printf("N2 = %i\n",N2);

	grid_CV1 = (double *)malloc(N1*sizeof(double));
	grid_CV2 = (double *)malloc(N2*sizeof(double));

	int i;
	for( i=0; i<N1; i++ ) {
		grid_CV1[i] = *val1_min + h1*i;		
	}
	for( i=0; i<N2; i++ ) {
		grid_CV2[i] = *val2_min + h2*i;		
	}

	long *bins;
	int Ngrid = N1*N2;
	bins = (long *)malloc(Ngrid*sizeof(long));
	for( j=0; j<Ngrid; j++ ) bins[j] = 0;
	
	srand(time(0));

	A_conf = (double *)malloc(NEXIT*dim*sizeof(double));
	B_conf = (double *)malloc(NEXIT*dim*sizeof(double));
	crossover_time = (double *)malloc(NEXIT*sizeof(double));
	
	
	sprintf(fname,"Data/Rtraj_CrossoverTime_BETA%i_run%i.txt",(int)round(BETA),run_id);	
	
	printf("BETA = %.2f\n",BETA);
	
	CPUbegin=clock(); // start time measurement
		
	conf0 = (double *)malloc(dim*sizeof(double));
	init_conf(conf0,1); 
	exit_confs(conf0,A_conf,nnrc,LDAmatrix,dt);
	generate_reactive_trajectories(A_conf,B_conf,nnrc,LDAmatrix,dt,crossover_time,
				bins,grid_CV1,grid_CV2,h1,h2,N1,N2);

	double mean_crossover_time = 0.0, std_crossover_time = 0.0;
	fid = fopen(fname,"w");
	for( j = 0; j < NEXIT; j++ ) {
		mean_crossover_time += crossover_time[j];
		fprintf(fid,"%.6e\n",crossover_time[j]);	
	}
	fclose(fid);
	mean_crossover_time /= NEXIT;
	printf("mean crossover time = %.6e\n",mean_crossover_time); 
	double aux;	
	for( j = 0; j< NEXIT; j++ ) {
		aux = mean_crossover_time - crossover_time[j];
		std_crossover_time += aux*aux;
	}
	std_crossover_time /= (NEXIT - 1);
	std_crossover_time = sqrt(std_crossover_time);
	printf("std crossover time = %.6e\n",std_crossover_time); 
	cpu = (clock()-CPUbegin)/((double)CLOCKS_PER_SEC);	// end time measurement
	printf("CPU time = %g\n",cpu);
	fprintf(fid,"CPU time = %g\n",cpu);


	// record the bins
	char fname1[100];
	int ind;
	sprintf(fname1,"Data/Rtraj_bins_beta%.i.txt",(int)round(BETA));
	FILE *fpot;
	fpot = fopen(fname1,"w");
	for( i=0; i<N1; i++ ) {
		for( j=0; j<N2; j++ ) {
			ind = i + j*N1;
			fprintf(fpot,"%li\t",bins[ind]);
		}
		fprintf(fpot,"\n");
	}
	fclose(fpot);
	
	return 0;
}


