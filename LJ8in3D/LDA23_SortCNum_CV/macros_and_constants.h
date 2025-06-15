#define DIM 3
#define BETA 20.0
#define NATOMS 8 // the number of atoms

// dimension of CV
#define CVDIM 2

// restraining potential
#define KAPPA 100.0 // spring constant for the restraining potential that turns on

// time step
#define TAU 5.0e-5

// metadynamics parameters 
#define GAMMA 1.0 // the artificial temperature for the discount factor in WTMetad
#define SIGMA 0.02 // width parameter for Gaussian bumps in WTMetad
#define HEIGHT 0.01 // height of Gaussian bump
#define NSTEPS_BETWEEN_DEPOSITS 500
#define NBUMPS_MAX 50000 // 10000 the number of Gaussian bumps to be deposited

// for FFS
#define MSTONE_MIN_VAL 1.0e-2
#define MSTONE_MAX_VAL 0.99

#define STARTING_MIN 1 // minima 0,1,2,3,4,5,6,7
#define FINISH_MIN 0

// general and LJ
#define PI 3.141592653589793
#define PI2 6.283185307179586 // 2*PI
#define RSTAR 1.122462048309373 // 2^{1/6}

// macros
#define mabs(a) ((a) >= 0 ? (a) : -(a))
#define sgn(a) ((a) == 0 ? 0 : ((a) > 0  ? 1 : -1 ))
#define max(a,b) ((a) >= (b) ? (a) : (b))
#define min(a,b) ((a) <= (b) ? (a) : (b))

struct vec2 {
	double x;
	double y;
};



