# LJ7in2D_LJ8in3D_learningCVs
This package contains a collection of C codes for the paper

[1] Jiaxin Yuan, Shashank Sule, Yeuk Yin Lam, and Maria Cameron, *"Learning collective variables that respect permutational symmetry".* ArXiv:<TBA>.

This package is designed for computing the free energy and diffusion matrix with respect to the standard and given machine-learned collective variables (CVs) for two systems: Lennard-Jones-7 in 2D and Lennard-Jones-8 in 3D. The machine-learned collective variables are obtained by Algorithm 1 in [1] implemented in the codes in the GitHub Repository

[https://github.com/margotyjx/OrthogonalityCVLearning/tree/main/LrCV_permsym](https://github.com/margotyjx/OrthogonalityCVLearning/tree/main/LrCV_permsym)

Furthermore, this package offers C codes for forward flux sampling, generating transition trajectories using stochastic control, brute force sampling. 

The directories LJ7in2D and LJ8in3D contain the collections of codes for the respective systems. The subdirectories of these two main directories contain codes for different collective variables.

### Subdirectories of LJ7in2D:
- **MLCV\_sort\_CNum**: codes for the CVs learned by Algorithm 1 in [1] with the feature map sort[c], the sorted vector of coordination numbers of the atomic coordinates. The coordination number of an atom $$i$$ is a continuous function of atomic coordinates approximately equal to the number of the nearest neighbors of atom $$i$$:\
   $$c_i = \sum_{j\neq i} (1-r_{i,j}^8)/(1 - r_{i,j}^{16}).$$
- **MLCV\_sort\_d2**: codes for the CVs learned by Algorithm 1 in [1] with the feature map sort[d^2], the sorted vector of pairwise distances squared.
- **mu2mu3**: codes for the standard set of CVs $$(\mu_2,\mu_3)$$, the second and the third central moments of the coordination numbers.
- **Drivers4FEMmesh**: Matlab codes for triangulation working with Darren Engwirda's mesh generator  [https://github.com/dengwirda/mesh2d](https://github.com/dengwirda/mesh2d)

### Subdirectories of LJ8in3D:
- **LDA12\_SortCNum\_CV**: codes for (LDA1,LDA2) CVs, the first and second dominant eigenvectors of the generalized eigenvalue problem in the Linear  Discriminant Analysis (see e.g. _R. O. Duda, P. E. Hart, D. G. Stork "Pattern Classification", Section 3.8.3_). The CVs (LDA1,LDA2) separate the union of the two lowest minima, 1 and 2, of LJ8 in 3D from the rest.
- **LDA23\_SortCNum\_CV**: codes for (LDA2,LDA3) CVs, the second third dominant eigenvectors of the generalized eigenvalue problem in the Linear  Discriminant Analysis. These CVs separate minima 1 and 2.
- **MLCV\_sort\_CNum**: codes for the CVs learned by Algorithm 1 in [1] with the feature map sort[c], the sorted vector of coordination numbers of the atomic coordinates. 
- **mu2mu3**: codes for CVs $$(\mu_2,\mu_3)$$, the second and the third central moments of the coordination numbers.
- **Drivers4FEMmesh**: Matlab codes for triangulation working with Darren Engwirda's mesh generator  [https://github.com/dengwirda/mesh2d](https://github.com/dengwirda/mesh2d)
- **LJ8_drawconf**: Matlab codes for visualizing the LJ8 cluster in 3D using balls and rods. The driver file is `LJ8_drawconf.m`. The drawing file is `DrawConf_BallsStix_Bicolor.m`. The data file with atomic coordinates of the minima of LJ8 in 3D is `LJ8min_xyz.m`. 

### Getting started
- If you want to reproduce figures from [1], it suffices to open the file **LJ{a description of the system and the CVs}.ipynb**. The provided data allow you to visualize the free energy, the diffusion matrix, the committor, and the probability density of transition trajectories for LJ7 in 2D at $$\beta \in\{5,7,9\}$$ and for LJ8 in 3D at $$\beta \in\{10,15,20\}$$.
- If you want to run C codes, you can use a command line tool. We ran all C codes in the Terminal on Mac OS. The compile command is given in a commented line at the top of each C code. For example, to run metadynamics with the machine-learned CV with the feature map sort[c], the compile command is\
`gcc -Wall LJ8_WTMetad_MargotCV_2D.c MargotColVar_CoordNum.c timestep_helpers.c -lm -O3`\
As the compilation is completed, type\
`./a.out`\
in the Terminal to run the code.
  

### The workflow of codes in each directory:
1. `macros_and_contants.h` contains MACROs and important parameters:
- `BETA` ($$\beta = 1/(k_BT)$$)
- `NATOMS` (the number of atoms)
- `DIM` (2, or 3, the dimension of the space where the atoms move)
- `CVDIM` (2, the dimension of the CV)
- `GAMMA`, `SIGMA`, `HEIGHT`, `NBUMPS_MAX`, `NSTEPS_BETWEEN_DEPOSITS` (the parameters for the well-tempered metadynamics)
- `N_MINIMA` (the number of local minima in the system)
- `KAPPA` (the spring force parameter for the restraining potential)
- `STARTING_MIN`, `FINISH_MIN`, `MSTONE_MIN_VAL`, `MSTONE_MAX_VAL` (the parameters for the forward flux sapling (Allen et al., 2005, 2009)
- `TAU` (the time step size)
2. `LJ{NATOMS}_WTMetad_{description of the CVs}.c` runs the metadynamics algorithm. It may take files as input specified in the declaration of main() with the parameters and the dimensions of the CVs. If the CVs are $$(\mu_2,\mu_3)$$, no input files are required. If the CVs are LDAs, one input file is required. If CVs are given by a neural network, two input files are required. The first file contains the parameters of the neural network, and the other one contains its dimensions. All these files are found in the corresponding directories and generated by Matlab codes `read_NN_data.m` (or a similar name). This code outputs the file
  - `GBumpsData/GaussianBumps_beta{BETA}.txt`. You must have the directory named `GBumpsData`. You also must have a directory `Data`.
Note that it suffices to deposit the Gaussian bumps at one value of `BETA` and use them for all other values of `BETA`.
3. `LJ{NATOMS}in{DIM}D_bicubicFE_binning.c` computes the free energy. It requires the file `GBumpsData/GaussianBumps_beta{BETA}.txt` as an input plus the input files (if any) of `LJ{NATOMS}_WTMetad_{description of the CVs}.c` describing the CVs. It firsts creates a bounding box in the CV space containing all centers of the Gaussian bumps and meshes it into N2-by-N1 grid cells. It approximates the biasing potential given by the Gaussian bumps with a bicubic inperpolant. Then it runc a long trajectory of `NSTEPS` (set to 1e9) and computes the free energy by the binning approach. More details are found in the appendix of [1]. The output files are saved in the directory `Data`:
- `bicubic_params.txt` (N1,N2,h1,h2,CV1 min value, CV1 max value, CV2 min value, CV2 max value) -- a column vector
- `LJ{NATOMS}bins_confs.txt` -- atomic coordinates of representative configurations in all visited bins. If a bin was not visited, the configuration has all zero coordinates.
- `LJ{NATOMS}bins_beta{BETA}.txt` -- the number of visits to each bin.
- `LJ{NATOMS}free_energy_beta{BETA}.txt` is the free energy
- `LJ{BETA}free_energy_bicubic_matrix_beta{BETA}.txt` is a file of ~3Mb with data to for the bicubic interpolant of the free energy allowing to calculate it on any grid of a different size or at any point. We do not include it in the GitHub version due to its large size.
4. `LJ{NATOMS}in{DIM}D_diffusion_matrix.c` computes the diffusion matrix. It requires the files `Data/bicubic_params.txt`, `Data/LJ{NATOMS}bins_confs.txt` and the input files (if any) of `LJ{NATOMS}_WTMetad_{description of the CVs}.c` describing the CVs. The calculation of the diffusion matrix is described in the appendix of [1]. It outputs the following files to the directory Data:
- `LJ{NATOMS}_M11_beta{BETA}.txt` -- the (1,1) component of the diffusion matrix
- `LJ{NATOMS}_M12_beta{BETA}.txt` -- the (1,2) component  of the diffusion matrix (this matrix is symmetric)
- `LJ{NATOMS}_M22_beta{BETA}.txt` -- the (2,2) component of the diffusion matrix
- `LJ{NATOMS}_M11_bicubic_matrix_beta{BETA}.txt` -- the data for the bicubic interpolant for the (1,1) component of the diffusion matrix that allow to evaluate $$M_{11}$$ on any mesh or at any point.
- `LJ{NATOMS}_M12_bicubic_matrix_beta{BETA}.txt` -- the similar data for (1,2)-component
- `LJ{NATOMS}_M22_bicubic_matrix_beta{BETA}.txt` -- the similar data for (2,2)-component
5. `LJ{a description of the system and the CVs}.ipynb` is a Python notebook that reads the files output by  `LJ{NATOMS}in{DIM}D_bicubicFE_binning.c` and `LJ{NATOMS}in{DIM}D_diffusion_matrix.c` and visualizes the free energy and the diffusion matrix. The package contains all necessary files allowing you to run this file at $$\beta = 5$$, $$7$$, and $$9$$ for LJ7 in 2D and $$\beta = 10$$, $$15$$, and $$20$$ for LJ8 in 3D. In most cases, it also contains the code lines for computing the committor using finite element method (FEM). The FEM codes are provided in the file FEM_TPT.py. The mesh data, are contained in the directories `FEMdataBETA{BETA}`. The mesh is generated using Darren Engwirda's mesh generator [https://github.com/dengwirda/mesh2d](https://github.com/dengwirda/mesh2d) written in Matlab. We provide drivers for them and `Tarjan.m`. See the directory Drivers4FEMmesh. To run, copy the files from there to Engwirda's `mesh2d_main` directory.
6. Next, the committor must be approximated by a neural network. This is done in [https://github.com/margotyjx/OrthogonalityCVLearning](https://github.com/margotyjx/OrthogonalityCVLearning). After that, we convert the neural network data to two input files, one with the network parameters and the other one with network dimensions, by a provided Matlab codes `read_committorNN_data.m` (or a similar name).
7. `FFS.c` runs the forward flux sampling algorithm _(R. J. Allen, C. Valeriani, and P. Rein ten Wolde, “Forward flux sampling for
rare event simulations,” Journal of Physics: Condensed Matter 21, 463102 (2009).)_. It reads the files with the neural network parameters and dimensions of the committor and of the CVs. If the CVs are LDAs, it requires the same LDA imput files as the metadynamics code. If the CVs are $$(\mu_2,\mu_3)$$, no files for them are requred. You must check that the minimal milestone $$\lambda_0$$ and the maximal milestone $$\lambda_{N-1}$$ are such that the neural network value at the starting minimum is smaller than $$\lambda_0$$ and the neural network value at the finish minimum is larger than $$\lambda_{N-1}$$.  If necessary, adjust the parameters `STARTING_MIN`  `FINISH_MIN` in `macros_and_contants.h`. You must make a directory  named `FFS&BF` in the `Data` directory for the output files with statistics.
8. `Rtraj_StochControl.c` generates samples of transition trajectories using stochastic control with the controller made out of the committor. The drift in the controlled process is modified as (_J. Yuan, A. Shah, C. Bentz, and M. Cameron, “Optimal control for sampling the transition path process and estimating rates,” Communications in Nonlinear Science and Numerical Simulation 129, 107701 (2024)_):
  $$b(x) \mapsto b(x) + 2\beta^{-1}\nabla\log q_{NN}(CV(\phi(x)))$$\
  where CV is represented by a neural network or an analytic formula and $$\phi$$ is the feature map (e.g. sort [c]). The unput files are the same as for FFS.c. You must make a directory  named `RtrajStochControl` in the `Data` directory for the output files with statistics and the binned probability density of transition trajectories.
9. `brute_force.c` runs brute force sampling of the transition process. The input files are the same as for `FFS.c`. The output files are written to the directory `Data/FFS&BF`.

  
