#define NATOMS 7 // the number of atoms
#define DIM 2 // the dimension of each particle

#define BETA 9.0

// Parameters for the WTMETAD
#define GAMMA 1.0 // the artificial temperature for the discount factor
#define NBUMPS_MAX 50000 // 50000 10000
#define SIGMA 0.02 // width parameter for Gaussian bumps
#define HEIGHT 2.0e-2 // height of Gaussian bump

#define CVDIM 2 // the dimension of the collective variable

#define TAU 5.0e-5 // time step

#define KAPPA 100.0 // spring constant for the restraining potential that turn on 
// if an atom is at distance more than 2 from the center of mass

#define STARTING_MIN 3 // minima 0,1,2,3
#define FINISH_MIN 0

#define N_MINIMA 4
// for FFS
#define MSTONE_MIN_VAL 1e-4 
#define MSTONE_MAX_VAL 0.9999

#define PI 3.141592653589793
#define PI2 6.283185307179586 // 2*PI
#define RSTAR 1.122462048309373 // 2^{1/6}
#define mabs(a) ((a) >= 0 ? (a) : -(a))
#define sgn(a) ((a) == 0 ? 0 : ((a) > 0  ? 1 : -1 ))
#define max(a,b) ((a) >= (b) ? (a) : (b))
#define min(a,b) ((a) <= (b) ? (a) : (b))
#define INFTY 1.0e6
