// This code does the following three things.
//
// (1) Reads parameter file and binning data for computing free energy.
//
// (2) Computes the diffusion matrix at the centers of all nonempty bins 
// using harmonic biasing potential U = U0 + 0.5*Kdiffmatr(||CV - CV(cell_center)||^2)
// and the formula 
// M_{i,j} = (1/Nsteps)sum_steps sum_k=1^{dim} (dCV_i(step)/dx_k)(dCV_j(step)/dx_k).
//
// (3) Builds a bicubic interpolant for M and its derivatives

// Compile command:  gcc -Wall LJ7in2D_diffusion_matrix.c MargotColVar_CoordNum.c timestep_helpers.c -lm -O3

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "macros_and_constants.h"
#include "timestep_helpers.h"
#include "MargotColVar_CoordNum.h"

#define TAU_DMATRIX 1.0e-5;
#define BKAPPA 500.0 // the spring constant for the biasing potential 
// attaching the configuration to particular values of CVs
#define INFTY 1.0e6
#define NSTEPS 1e6


void derivatives_on_grid(double *fun,double *der1,double *der2,double *der12,int N1,int N2);
void total_pot_and_grad(double *conf,int Natoms,double *pot,double *grad,
	double *CV,double *CVgrad,double CV1star,double CV2star,double Bkappa,struct NN *nncv);
void diffusion_matrix(int Natoms,double *conf0,double dt,int Nsteps,
	double CV1val,double CV2val,int ind,double *M11,double *M12,double *M22,
	struct NN *nncv);
void prepare_conf(int Natoms,double *conf0,double CV1val,double CV2val,
		double dt,int Nsteps);

//----- main	
int main(void);

//-------------------------------------------------------------
//------------------------------------------------------------
// Evaluate the total potential energy and its gradient

void total_pot_and_grad(double *conf,int Natoms,double *pot,double *grad,
	double *CV,double *CVgrad,
	double CV1star,double CV2star,double Bkappa,struct NN *nncv) {
	
	int j,dim = Natoms*DIM;
	double CV1diff,CV2diff;

	LJpot_and_grad(conf,pot,grad,Natoms);
	restraining_pot_and_grad(conf,pot,grad,Natoms);
	MargotCVgrad(CVgrad,CV,conf,Natoms,nncv);			
	// add the effect of the biasing potential
	// B(x) = 0.5*Bkappa*[(CV1(x) - CV1val)^2 + (CV2(x) - CV2val)^2 )
	CV1diff = CV[0] - CV1star;
	CV2diff = CV[1] - CV2star;
	for( j=0; j<dim; j++ ) {
		grad[j] += Bkappa*(CV1diff*CVgrad[j] + CV2diff*CVgrad[j+dim]);
	}	
	*pot += 0.5*Bkappa*(CV1diff*CV1diff + CV2diff*CV2diff);	
// 	printf("CV1 = %.4e, CV2 = %.4e\n",*CV1,*CV2);
// 	printf("FEpot = %.4e,FEder1 = %.4e,FEder2 = %.4e\n",*FEpot,*FEder1,*FEder2);
		
}
//------------------------------------------------------------
// 	diffusion_matrix(Natoms,conf,dt,Nsteps,grid_CV1[i],grid_CV2[j],ind,M11,M12,M22)

void diffusion_matrix(int Natoms,double *conf0,double dt,int Nsteps,
	double CV1val,double CV2val,int ind,double *M11,double *M12,double *M22,
	struct NN *nncv) {

	int j,k,dim = Natoms*DIM,kshift;
	double *gauss01;
	int Ngauss = 2*DIM*Natoms, jgauss;
	double std = sqrt(2.0*dt/BETA);
	double *conf1,*Vpot0,*Vpot1,*Vgrad0,*Vgrad1,*w;
	double *CV,*CVgrad;
	double M11update,M12update,M22update;
	char ch;
	double Bkappa = BKAPPA;

	conf1 = (double *)malloc(dim*sizeof(double));
	Vgrad0 = (double *)malloc(dim*sizeof(double));
	Vgrad1 = (double *)malloc(dim*sizeof(double));
	Vpot0 = (double *)malloc(sizeof(double));
	Vpot1 = (double *)malloc(sizeof(double));
	w = (double *)malloc(dim*sizeof(double));
	CV = (double *)malloc(CVDIM*sizeof(double));
	CVgrad = (double *)malloc(CVDIM*dim*sizeof(double));
	gauss01 = (double *)malloc(Ngauss*sizeof(double));
		
	total_pot_and_grad(conf0,Natoms,Vpot0,Vgrad0,CV,CVgrad,
		CV1val,CV2val,Bkappa,nncv);
	
// 	for( k = 0; k < dim; k++ ) printf("%.4e\n",conf0[k]);
// 	test_CVgrad(conf0);
	
				
	jgauss = 0;	
	M11[ind] = 0.0;
	M12[ind] = 0.0;
	M22[ind] = 0.0;			
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
			CV1val,CV2val,Bkappa,nncv);
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
		// update the values of the diffusion matrix
		M11update = 0.0;
		M12update = 0.0;
		M22update = 0.0;
		for( k = 0; k < dim; k++ ) {
			M11update += CVgrad[k]*CVgrad[k];
			M12update += CVgrad[k]*CVgrad[k+dim];
			M22update += CVgrad[k+dim]*CVgrad[k+dim];
		}
		M11[ind] = (M11[ind]*j + M11update)/(j+1);
		M12[ind] = (M12[ind]*j + M12update)/(j+1);
		M22[ind] = (M22[ind]*j + M22update)/(j+1);
		if( j%100000 == 0 ) {
			printf("M11 = %.4e, M12 = %.4e, M22 = %.4e\n",M11[ind],M12[ind],M22[ind]);
// 			for( k = 0; k < dim; k++ ) printf("%.4e\n",conf0[k]);
// 			test_CVgrad(conf0);
		}
	}
}
//-----------------------------------------------------------

//-----------------------------------------------------------

void derivatives_on_grid(double *fun,double *der1,double *der2,double *der12,int N1,int N2) {

	int n1m1 = N1-1, n2m1 = N2-1;
	int i,j,ind;

	// interior grid points
	for( i = 1; i < n1m1; i++ ) {
		for( j = 1; j < n2m1; j++ ) {
			ind = i + j*N1;	
			der1[ind] = 0.5*(fun[ind+1]-fun[ind-1]);
			der2[ind] = 0.5*(fun[ind+N1]-fun[ind-N1]);

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
		der1[ind] = fun[ind+1] - fun[ind];
		der2[ind] = 0.5*(fun[ind+N1]-fun[ind-N1]);
		// i = n1m1
		ind = j*N1+n1m1;
		der1[ind] = fun[ind] - fun[ind-1];
		der2[ind] = 0.5*(fun[ind+N1]-fun[ind-N1]);
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
		der1[ind] = 0.5*(fun[ind+1]-fun[ind-1]);
		der2[ind] = fun[ind+N1]-fun[ind];
		// j = n2m1
		ind = n2m1*N1+i;
		der1[ind] = 0.5*(fun[ind+1]-fun[ind-1]);
		der2[ind] = fun[ind]-fun[ind-N1];
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
	der1[0] = fun[1] - fun[0];
	der2[1] = fun[N1] - fun[0];
	der12[0] = der1[N1] - der1[0];
	// i = n1m1; j = 0;
	der1[n1m1] = fun[n1m1] - fun[n1m1-1];
	der2[n1m1] = fun[N1+n1m1] - fun[n1m1];
	der12[n1m1] = der1[N1+n1m1] - der1[n1m1];
	// i = 0; j = n2m1;
	ind = N1*n2m1;
	der1[ind] = fun[ind+1] - fun[ind];
	der2[ind] = fun[ind] - fun[ind-N1];
	der12[ind] = der1[ind] - der1[ind-N1];
	// i = n1m1; j = n2m1;
	ind =  n1m1 + N1*n2m1;
	der1[ind] = fun[ind] - fun[ind-1];
	der2[ind] = fun[ind] - fun[ind-N1];
	der12[ind] = der1[ind] - der1[ind-N1];
}






//------------------------------------------------------------
int main(void){
	int N1, N2, Nsteps = NSTEPS, Natoms = NATOMS, Ngrid;
	int i,j,k,ind,n,dim=NATOMS*DIM;
	double h1,h2;
	double *val1_min,*val1_max,*val2_min,*val2_max;
	double *grid_CV1,*grid_CV2;
	double *conf0;
	double *bicubic_matrix;
	double *M11,*M12,*M22;
	double *grid_der1,*grid_der2,*grid_der12;
	FILE *fM11,*fM12,*fM22,*fpar,*fconf;
    clock_t CPUbegin; // for measuring CPU time
    double cpu; // for recording CPU time
    double dt = TAU_DMATRIX;
	char *bins;
	char fname[100];
	char fCVname[] = "LJ7_CV_data/MargotCV_CNum_NNdata.txt";
	char fname_dim[] = "LJ7_CV_data/MargotCV_CNum_dimensions.txt";
	struct NN *nncv;

	nncv = (struct NN *)malloc(sizeof(struct NN));
	readCVdata(fCVname,fname_dim,nncv);

	val1_min = (double *)malloc(sizeof(double));
	val1_max = (double *)malloc(sizeof(double));
	val2_min = (double *)malloc(sizeof(double));
	val2_max = (double *)malloc(sizeof(double));

	// save the parameters and the bicubic matrix 
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

	for( i=0; i<N1; i++ ) {
		grid_CV1[i] = *val1_min + h1*i;		
	}
	for( i=0; i<N2; i++ ) {
		grid_CV2[i] = *val2_min + h2*i;		
	}
	
	conf0 = (double *)malloc(dim*sizeof(double));
	
	Ngrid = N1*N2;
	M11 = (double *)malloc(Ngrid*sizeof(double));
	M12 = (double *)malloc(Ngrid*sizeof(double));
	M22 = (double *)malloc(Ngrid*sizeof(double));
	
	// Compute the diffusion matrix at all nonempty bin centers
 	CPUbegin=clock(); // start time measurement
 	double confrad = 0.0,TOL = 1.0e-10;
 	fconf = fopen("Data/LJ7bins_confs.txt","r");
 	bins = (char *)malloc(Ngrid*sizeof(char)); 	 	
	for( i=0; i<N1; i++ ) {
		for( j=0; j<N2; j++ ) {
			ind = i + j*N1;
			confrad = 0.0;
			bins[ind] = 0;
			for( k = 0; k < dim; k++ ) {
				fscanf(fconf,"%le\t",conf0+k);
				confrad += conf0[k]*conf0[k];				
			}
			fscanf(fconf,"\n");
			if( confrad > TOL ) {
				bins[ind] = 1;
				printf("bin (%i,%i)\n",i,j);
				// Compute the diffusion matrix
				diffusion_matrix(Natoms,conf0,dt,Nsteps,grid_CV1[i],grid_CV2[j],
					ind,M11,M12,M22,nncv);
			}
		}
	}
	fclose(fconf);
	cpu = (clock()-CPUbegin)/((double)CLOCKS_PER_SEC);	// end time measurement		
	printf("Computation of the diffusion matrix: CPU time = %g\n",cpu);

	// Do nearest-neighbor interpolation
 	CPUbegin=clock(); // start time measurement
	double aux1,aux2,d0,d1;
	d0 = INFTY;
	d1 = min(h1,h2);
	int ind1;
	
	for( i=0; i<N1; i++ ) {
		for( j=0; j<N2; j++ ) {
			ind = i + j*N1;
			if( bins[ind] == 0 ) {				
				d0 = INFTY;
				d1 = min(h1,h2);
				for( k = 0; k < N1; k++ ) {
					for( n = 0; n < N2; n++ ) {
						ind1 = k+n*N1;
						if( bins[ind1] > 0 ) {
							aux1 = h1*(i-k);
							aux2 = h2*(j-n);
							d1 = sqrt(aux1*aux1+aux2*aux2);
							if( d1 < d0 ) {
								d0 = d1;
								M11[ind] = M11[ind1];
								M12[ind] = M12[ind1];
								M22[ind] = M22[ind1];
							}						
						}
					}
				}			
			}
		}
	}
	cpu = (clock()-CPUbegin)/((double)CLOCKS_PER_SEC);	// end time measurement		
	printf("Nearest neighbor interpolation: CPU time = %g\n",cpu);
	
	// Save the diffusion matrix
	sprintf(fname,"Data/LJ7_M11_beta%.0f.txt",BETA);
	fM11 = fopen(fname,"w");
	sprintf(fname,"Data/LJ7_M12_beta%.0f.txt",BETA);
	fM12 = fopen(fname,"w");
	sprintf(fname,"Data/LJ7_M22_beta%.0f.txt",BETA);
	fM22 = fopen(fname,"w");
		
	for( j= 0; j<N2; j++ ){
		for( i=0; i<N1; i++ ) {
			ind = i + N1*j;
			fprintf(fM11,"%.4e\t",M11[ind]);
			fprintf(fM12,"%.4e\t",M12[ind]);
			fprintf(fM22,"%.4e\t",M22[ind]);
		}
		fprintf(fM11,"\n");
		fprintf(fM12,"\n");
		fprintf(fM22,"\n");
	}	
	fclose(fM11);
	fclose(fM12);
	fclose(fM22);
		
	// Compute the bicubic matrices for the components of the diffusion matrix
	sprintf(fname,"Data/LJ7_M11_bicubic_matrix_beta%.0f.txt",BETA);
	fM11 = fopen(fname,"w");
	sprintf(fname,"Data/LJ7_M12_bicubic_matrix_beta%.0f.txt",BETA);
	fM12 = fopen(fname,"w");
	sprintf(fname,"Data/LJ7_M22_bicubic_matrix_beta%.0f.txt",BETA);
	fM22 = fopen(fname,"w");

	grid_der1 = (double *)malloc(Ngrid*sizeof(double));
	grid_der2 = (double *)malloc(Ngrid*sizeof(double));
	grid_der12 = (double *)malloc(Ngrid*sizeof(double));
	bicubic_matrix = (double *)malloc(Ngrid*dim*sizeof(double));

	derivatives_on_grid(M11,grid_der1,grid_der2,grid_der12,N1,N2);
	for( j = 0; j < N2; j++ ) {
		for( i=0; i < N1; i++ ) {
			ind = i + j*N1;
			compute_bicubic_coeff_matrix(M11,grid_der1,grid_der2,grid_der12,
				bicubic_matrix,ind,N1,N2);
			for( n=0; n < 16; n++ ) {
				fprintf(fM11,"%.4e\t",bicubic_matrix[ind*16 + n]);
			}	
			fprintf(fM11,"\n");
		}
	}
	fclose(fM11);

	derivatives_on_grid(M12,grid_der1,grid_der2,grid_der12,N1,N2);
	for( j = 0; j < N2; j++ ) {
		for( i=0; i < N1; i++ ) {
			ind = i + j*N1;
			compute_bicubic_coeff_matrix(M12,grid_der1,grid_der2,grid_der12,
				bicubic_matrix,ind,N1,N2);
			for( n=0; n < 16; n++ ) {
				fprintf(fM12,"%.4e\t",bicubic_matrix[ind*16 + n]);
			}	
			fprintf(fM12,"\n");
		}
	}
	fclose(fM12);

	derivatives_on_grid(M22,grid_der1,grid_der2,grid_der12,N1,N2);
	for( j = 0; j < N2; j++ ) {
		for( i=0; i < N1; i++ ) {
			ind = i + j*N1;
			compute_bicubic_coeff_matrix(M22,grid_der1,grid_der2,grid_der12,
				bicubic_matrix,ind,N1,N2);
			for( n=0; n < 16; n++ ) {
				fprintf(fM22,"%.4e\t",bicubic_matrix[ind*16 + n]);
			}	
			fprintf(fM22,"\n");
		}
	}
	fclose(fM22);
	
	return 0;
}

