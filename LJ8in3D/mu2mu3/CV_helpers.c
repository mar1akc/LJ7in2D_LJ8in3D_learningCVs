// This code computes the CVs mu2, mu3, 
// the second and third central moments of the coordination numbers
// and the gradients of the CVs
//

// Compile command:  gcc -Wall CV_helpers.c timestep_helpers.c -lm -O3

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "timestep_helpers.h"
#include "macros_and_constants.h"


void mu2mu3(double *conf,double *CVval);
void mu2mu3grad(double *conf,double *CVval,double *grad,int CVdim);


//-------------------------------------------------------------
//------------------------------------------------------------
// c_i(x) = sum_{j\neq i} (1 - (r_{ij}/1.5)^8) / (1 - (r_{ij}/1.5)^{16})
// mean(c_i) = [sum_{i=1}^{Natoms} c_i] / Natoms
// \mu_2(x) = [\sum_{i} (c_i - mean(c_i))^2] / Natoms
// \mu_3(x) = [\sum_{i} (c_i - mean(c_i))^3] / Natoms

void mu2mu3(double *conf,double *CVval) {
	double coord_num[NATOMS],mean_coord_num = 0.0;
	double aux_x, aux_y, aux_z,r2,r8,r16,aux;
	int j,k,Na2 = NATOMS*2;
	
	// compute coordination numbers
	for( k=0; k<NATOMS; k++ ) {
		coord_num[k] = 0.0;
	}
	for( k=1; k<NATOMS; k++ ) {	
		for( j=0; j<k; j++ ) {
			aux_x = conf[k] - conf[j];
			aux_y = conf[k+NATOMS] - conf[j+NATOMS];
			aux_z = conf[k+Na2] - conf[j+Na2];
			r2 = (aux_x*aux_x + aux_y*aux_y + aux_z*aux_z)/2.25;
			r8 = pow(r2,4);
			r16 = r8*r8;
			aux = (1.0 - r8)/(1.0 - r16);
			coord_num[k] += aux;
			coord_num[j] += aux;
		}		
	}
	// compute mean coordination number
	for( k=0; k<NATOMS; k++ ) mean_coord_num += coord_num[k];
	mean_coord_num /= NATOMS;

	// compute mu2 and mu3
	for( k=0; k<NATOMS; k++ ) coord_num[k] -= mean_coord_num;
	CVval[0] = 0.0;
	CVval[1] = 0.0;
	for( k=0; k<NATOMS; k++ ) {
		aux = coord_num[k]*coord_num[k];
		CVval[0] += aux;
		CVval[1] += aux*coord_num[k];
	}
	CVval[0] /= NATOMS;
	CVval[1] /= NATOMS;
}

//-----------------------------------------------------------
// Compute the CVs mu2 and mu3 and their gradients 
void mu2mu3grad(double *conf,double *CVval,double *grad,int CVdim) {
	double coord_num[NATOMS],mean_coord_num = 0.0;
	double aux_x,aux_y,aux_z,r2,r8,r16,aux,iden,fac;
	int j,k,Na2 = NATOMS*2;
	double grad_coord_num[NATOMS][DIM*NATOMS],mean_grad[DIM*NATOMS];
	int dim = DIM*NATOMS;
	const double sigma2 = 2.25; // 1.5^2

	// initialization	
	for( k=0; k<NATOMS; k++ ) {
		coord_num[k] = 0.0;		
		for( j=0; j<dim; j++ ) {
			grad_coord_num[k][j] = 0.0;	
		}
	}
	for( j=0; j<dim; j++ ) {
		mean_grad[j] = 0.0;
		grad[j] = 0.0; // mu2grad
		grad[j+dim] = 0.0; // mu3grad
	}	
	
	// compute coordination numbers
	for( k=1; k<NATOMS; k++ ) {	
		for( j=0; j<k; j++ ) {
			aux_x = conf[k] - conf[j];
			aux_y = conf[k+NATOMS] - conf[j+NATOMS];
			aux_z = conf[k+Na2] - conf[j+Na2];
			r2 = (aux_x*aux_x + aux_y*aux_y + aux_z*aux_z)/sigma2;
			r8 = pow(r2,4);
			r16 = r8*r8;
			aux = (1.0 - r8)/(1.0 - r16);
			coord_num[k] += aux;
			coord_num[j] += aux;
			iden = 1.0/(1.0 - r16);
			fac = -4.0*pow(r2,3)*iden + aux*8.0*pow(r2,7)*iden;
			aux = fac*2.0*aux_x/sigma2;
			grad_coord_num[k][k] += aux;
			grad_coord_num[k][j] -= aux;
			grad_coord_num[j][k] += aux;
			grad_coord_num[j][j] -= aux;
			aux = fac*2.0*aux_y/sigma2;
			grad_coord_num[k][k+NATOMS] += aux;
			grad_coord_num[k][j+NATOMS] -= aux;
			grad_coord_num[j][k+NATOMS] += aux;
			grad_coord_num[j][j+NATOMS] -= aux;
			aux = fac*2.0*aux_z/sigma2;
			grad_coord_num[k][k+Na2] += aux;
			grad_coord_num[k][j+Na2] -= aux;
			grad_coord_num[j][k+Na2] += aux;
			grad_coord_num[j][j+Na2] -= aux;
		}		
	}	
	// compute mean coordination number and its grad
	for( k=0; k<NATOMS; k++ ) mean_coord_num += coord_num[k];
	mean_coord_num /= NATOMS;
	for( j=0; j<dim; j++ ) {
		for( k=0; k<NATOMS; k++ ) mean_grad[j] += grad_coord_num[k][j];
		mean_grad[j] /= NATOMS;
	}

	// compute mu2 and mu3 and their gradients
	for( k=0; k<NATOMS; k++ ) coord_num[k] -= mean_coord_num;
	CVval[0] = 0.0;
	CVval[1] = 0.0;
	for( k=0; k<NATOMS; k++ ) {
		aux = coord_num[k]*coord_num[k];
		CVval[0] += aux;
		CVval[1] += aux*coord_num[k];
		for( j=0; j<dim; j++ ) {
			aux = coord_num[k]*(grad_coord_num[k][j] - mean_grad[j]);
			grad[j] += 2.0*aux;
			grad[j + dim] += 3.0*coord_num[k]*aux;
		}
	}
	CVval[0] /= NATOMS;
	CVval[1] /= NATOMS;
	for( j=0; j<dim; j++ ) {
		grad[j] /= NATOMS;
		grad[j + dim] /= NATOMS;
	}
}



