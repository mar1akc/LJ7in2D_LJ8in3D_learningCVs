# LJ7in2D_LJ8in3D_learningCVs
This package contains a collection of C codes for the paper

[1] Jiaxin Yuan, Shashank Sule, Yeuk Yin Lam, and Maria Cameron, *"Learning collective variables that respect permutational symmetry".* ArXiv:<TBA>.

This package is designed for computing the free energy and diffusion matrix with respect to the standard and given machine-learned collective variables (CVs) for two systems: Lennard-Jones-7 in 2D and Lennard-Jones-8 in 3D. The machine-learned collective variables are obtained by Algorithm 1 in [1] implemented in the codes in the GitHub Repository

[https://github.com/margotyjx/OrthogonalityCVLearning](https://github.com/margotyjx/OrthogonalityCVLearning)

Furthermore, this package offers C codes for forward flux sampling, generating transition trajectories using stochastic control, brute force sampling. 

The directories LJ7in2D and LJ8in3D contain the collections of codes for the respective systems. The subdirectories of these two main directories contain codes for different collective variables.

Subdirectories of LJ7in2D:
- MLCV_sort_CNum: codes for the CVs learned by Algorithm 1 in [1] with the feature map sort[c], the sorted vector of coordination numbers of the atomic coordinates (the coordination number of an atom $$i$$ is a continuous function of atomic coordinates approximately equal to the number of nearest neighbors of atom $$i$$:\
   $$c_i = \sum_{j\neq i} (1-r_{i,j}^8)/(1 - r_{i,j}^{16})$$;
- MLCV_sort_d2: codes for the CVs learned by Algorithm 1 in [1] with the feature map sort[d^2], the sorted vector of pairwise distances squared.
- mu2mu3: codes for the standard set of CVs $$(\mu_2,\mu_3)$$, the second and the third central moments of the coordination numbers.
