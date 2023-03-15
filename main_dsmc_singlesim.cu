/* _________________________________________________________________________________________
FILE:           main_dsmc_singlesim.cu
AUTHOR:         Lachlan Schilling
DATE:           12-Mar-2023
DESCRIPTION:    Primitive DSMC collision routine for a rectangular tube of heated argon gas.
                Inputs can be adjusted below. The solver assumes one block per cell.

LIMITATIONS:
- The code is optimized for my GPU (NVIDIA GeForce GTX 1650 Ti, 2GB) and must be manually 
  checked if other GPUs are used
- No feature to upload simulation data to host mid-sim. This limits the product of 
  simultaneous simulations, particle count, and timesteps.

TAGS (INCLUSIONS)
- Reductions, parallel sorting, nested kernels, random numbers, constant memory, DSMC,
  simulations, shared memory

TAGS (EXCLUSIONS)
- Streams, spatial locality, texture memory

IMPROVEMENTS TO BE MADE / REFLECTION
- Due to the issues discussed below, the code executes slowly. The next attempt will
  incorporate adaptive cells to increase speed.
- DSMC posed a challenge I did not foresee relating to dynamic shared memory allocation 
  (and, as a result, occupancy). As the temperature fluctuates throughout a simulation, the
  particles per cell can fluctuate by an order of magnitude. This algorithm initialises 
  shared memory arrays statically in each timestep. Since they are statically allocated, 
  they are always of size threadsPerBlock. This has two problems:

    1. It cannot be guaranteed that threadsPerBlock will be large enough to contain all
       particles in a particular cell for all temperatures encountered. This is due to 
       the random nature of DSMC.

    2. Since threadsPerBlock must be large to account for density changes, it means that 
       the cells with far fewer particles (an order of magnitude less) will have 
       many threads not doing any work. This lowers occupancy considerably.

  The solution to this would be to dynamically allocate shared memory in each block, which
  is not suuported in CUDA (as far as I know).

- Even if the arrays could be dynamically allocated, this would not improve the computation
  speed greatly, as each thread would likely still be doing multiple loops of work to account
  for all particles in a cell. Instead, the cell would need to be divided into smaller sub-
  -cells where each thread accounts for a single particle each. This is similar to adaptive
  meshing in CFD. This solution would also allow static allocation of shared memory. If I 
  were to try DSMC again, this is the route I would pursue.

- These issues are not hurdles in CFD because the density in a cell is represented by a single
  float, rather than being physically represented by many particles.
*******************************************************************************************/

#include "../common/book.h"
#include <cuda_device_runtime_api.h>
#include <ios>
#include <iostream>
#include <fstream>
#include <cmath>

#include "cuda.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#define imin(a,b) (a<b?a:b)
#define beta sqrt(k / m)
#define rnd( x ) (x * (float) rand() / RAND_MAX)

// _____________________________________________________________________________________
// VARIABLES
// INPUTS (h refers to host)
// Independent Variables
#define timesteps   6000
#define DIMx        10
#define DIMy        10
#define DIMz        10
#define particles   25000
#define frequency   60                    //        controls frequency of output to terminal
static const float  h_n         = 1e18;   //  /m3   molecules per cubic meter
static const float  h_wallTemp  = 1000;   //  K
static const float  h_minTemp   = 293;    //  K
static const float  h_MBP       = 1e-6;   //  -     probability cutoff for vMax, 1 in 1,000,000
static const float  h_Lx        = 1e-3;   //  m     domain size x, y, z
static const float  h_Ly        = 1e-3;   //  m
static const float  h_Lz        = 1e-3;   //  m
static const float  h_N_dt_tau  = 5e6;    // -      number of timesteps per mean collision time

// CONSTANTS
static const float h_PI   = 3.1415926535897932;
static const float h_k    = 1.380649e-23; // J/K    Boltzmann's constant 
static const float h_m    = 6.63e-26;     // kg     Argon particle mass
static const float h_d    = 3.66e-10;     // m      Argon particle diameter

// CALCULATED CONSTANTS
// Dependent Variables
static const int   h_cells   = DIMx*DIMy*DIMz;                          // numbers of cells
static const float h_maxTemp = 1.3 * h_wallTemp;                        // max temp estimate assuming some collisions
static const float h_dLx     = h_Lx / DIMx;                             // length of a cell (x)
static const float h_dLy     = h_Ly / DIMy;                             // length of a cell (y)
static const float h_dLz     = h_Lz / DIMz;                             // length of a cell (z)
static const float h_volCELL = h_dLx * h_dLy * h_dLz;                   // volume of a cell
static const float h_nPARTeff= h_n * h_Lx * h_Ly * h_Lz / particles;    // effective number of atoms
static const float h_lambda  = 1/sqrt(2.0f) * h_PI * pow(h_d, 2) * h_n; // mean free path
static const float h_vMP     = sqrt((double)2 * h_k * h_minTemp / h_m); // most probable velocity
static const float h_vAVE    = 2.0f / sqrt(h_PI) * h_vMP;               // average velocity
static const float h_tau     = h_lambda / h_vAVE;                       // mean collision time
static const float h_dt      = h_tau / h_N_dt_tau;                      // timestep duration
static const float h_sigma   = h_PI * pow(h_d, 2);                      // collision cross section
static const float h_coefNTC = 0.5f * h_sigma * h_nPARTeff * h_dt / h_volCELL; // term for efficient calculation
struct wall
{
    // temperatures
    float lefT = h_wallTemp;
    float rigT = h_wallTemp;
    float botT = h_wallTemp;
    float topT = h_wallTemp;
    float aftT = h_minTemp;
    float froT = h_minTemp;

    // particle reflection directions (relative to axes)
    int L =  1;
    int R = -1;
    int B =  1;
    int T = -1;
    int A =  1;
    int F = -1;

    // wall locations
    float LW = 0.0f;
    float RW = h_Lx;
    float BW = 0.0f;
    float TW = h_Ly;
    float AW = 0.0f;
    float FW = h_Lz;
};
struct data {
    // host
    float   *u;
    float   *v;
    float   *w;
    float   *T;
    float   *rho;
    // device
    float   *dev_x;
    float   *dev_y;
    float   *dev_z;
    float   *dev_u;
    float   *dev_v;
    float   *dev_w;
    float   *dev_T;
    float   *dev_rho;

    int     *c_idx;  // cell indicies of particles
    int     *c_adrs; // address of first particle in cell
    int     *c_ppc;  // particles per cell
    int     *c_key;  // reference key to find particles in cells
    float   *c_rem;  // remaining collisions in cell
    float   *c_dVM;  // maximum relative velocity in a cell
};
// CUDA
#define threadsPerBlock 512 // ~20*(particles/cells), must be power of 2 for reductions, must be ~20x greater than average ppc
                             // this is because density increases by a factor of 20 in some cells as equilibrium is being reached.
#define blocksPerGrid DIMx*DIMy*DIMz

// CONSTANT MEMORY VARIABLES DECLARATION
int __constant__   cells, MAXppc;
float __constant__ n, wallTemp, minTemp, maxTemp, MBP, Lx, Ly, Lz, N_dt_tau, PI, k, m, d, dLx, dLy, dLz;
float __constant__ volCell, nPARTeff, vMP, vAVE, dt, coefNTC, vMAX, vACT;
// *********************************************************************************************


// _____________________________________________________________________________________________
// FUNCTIONS
// HOST
float vmax() {

    // This function generates the initial maximum velocity of a particle according to a Maxwell-Boltzman distribution.

    // FINDING VMAX
    int length    = 5e3;
    int index     = -1;
    float *vSWEEP = new float[length];
    float *mbPopA = new float[length];
    float vMAX;

    // generate vSWEEP (m/s)
    for (int i = 0; i < length; i++) {
        vSWEEP[i] = (float) i;
    }

    // generate maxwell boltzman distribution based on vSWEEP && wallTemp
    for (int i = 0; i < length; i++) {
        mbPopA[i] = 4.0f * h_PI * pow(h_m/(2.0f *h_PI * h_k * h_maxTemp), 1.5) * pow(vSWEEP[i], 2) * exp(-1 * pow(vSWEEP[i], 2) * h_m /(2.0f * h_k * h_maxTemp));
    }

    // determine vMAX according to probability cutoff
    for (int i = 1; i < length; i++) {
        mbPopA[i] += mbPopA[i - 1];
        if (mbPopA[i] > (1.0f - h_MBP)) {
            index = i;
            break;
        }
    }

    // Error checking
    if (index < 0) {
        vMAX = 1.0f;
    }
    else if (index > 0) {
        vMAX = vSWEEP[index];
    }

    delete [] vSWEEP;
    delete [] mbPopA;

    return vMAX;
}
float vactual(float vMAX) {

    // This function generates the initial "actual" velocity of a particle according to a Maxwell-Boltzman distribution.

    // generate vVEC from vMAX
    int length    = (int) vMAX;
    int index     = -1;
    float *vVEC   = new float[length];
    float *mbPopA = new float[length];
    float rand0, vACT;

    // generate vVec (m/s)
    for (int i = 0; i < length; i++) {
        vVEC[i] = (float) i;
    }

    // generate maxwell boltzman distribution based on vVEC && minTemp
    for (int i = 0; i < length; i++) {
        mbPopA[i] = 4.0f * h_PI * pow(h_m/(2.0f * h_PI * h_k * h_minTemp), 1.5) * pow(vVEC[i], 2) * exp(-1 * pow(vVEC[i], 2) * h_m /(2.0f * h_k * h_minTemp));
    }

    // generate random number for current particle
    srand(time(NULL));
    rand0 = rnd( 1.0f );

    // determine vVEC according to probability cutoff
    for (int i = 1; i < length; i++) {
        mbPopA[i] += mbPopA[i - 1];
        if (rand0 < mbPopA[i]) {
            index = i;
            break;
        }
    }
    vACT = vVEC[index];
    delete [] vVEC;
    delete [] mbPopA;

    return vACT;
}
// DEVICE
__device__ int particleCellIndex(float x, float y, float z) {

    // This function outputs the cell ID of a particle based on its cartesian position.

    float gx, gy, gz;

    // grid locations
    gx = (int) (x / dLx);
    gy = (int) (y / dLy);
    gz = (int) (z / dLz);

    // 1D grid location
    int c = gx + DIMx * gy + DIMx * DIMy * gz;
    return c;

}
__device__ bool boundaryConditions(float *newpos, float *newvel, float old_x, float old_y, float old_z, wall WALL, int idx) {
    
    // This function updates the position and velocity of particle who escape the boundaries of the domain during a timestep.

    //VARIABLES
    float rand1;
    float randn1;
    float randn2;
    float tof;      // time of flight (see documentation)
    float x_at_wall;
    float y_at_wall;
    float z_at_wall;

    // initialising random number state
    curandState state;
    curand_init((unsigned long long)clock() + idx, 0, 0, &state);

    // bools to determine which boundary has been cross
    bool updateBoundary = false;
    bool left   = false;
    bool right  = false;
    bool bottom = false;
    bool top    = false;
    bool aft    = false;
    bool front  = false;
    
    // determine which boundaries the particle has escaped througgh
    if (newpos[1] <= WALL.LW)   left    = true;
    if (newpos[1] >= WALL.RW)   right   = true;
    if (newpos[2] <= WALL.BW)   bottom  = true;
    if (newpos[2] >= WALL.TW)   top     = true;
    if (newpos[3] <= WALL.AW)   aft     = true;
    if (newpos[3] >= WALL.FW)   front   = true;

    // update boundaries
    if (left){

        updateBoundary = true;
        // THERMAL WALL
        // Velocity update
        rand1 = curand_uniform(&state);
        randn1= curand_normal(&state);
        randn2= curand_normal(&state);
        newvel[1] = WALL.L * beta * sqrt(-2.0f * WALL.lefT * log(rand1));
        newvel[2] = beta * sqrt(WALL.lefT) * randn1;
        newvel[3] = beta * sqrt(WALL.lefT) * randn2;
        // House keeping
        tof         = dt * (newpos[1] - WALL.LW) / (newpos[1] - old_x);
        y_at_wall   = newpos[2] - (tof/dt) * (newpos[2] - old_y); 
        z_at_wall   = newpos[3] - (tof/dt) * (newpos[3] - old_z); 
        // Position update
        newpos[1] = WALL.LW   + newvel[1] * tof;
        newpos[2] = y_at_wall + newvel[2] * tof;
        newpos[3] = z_at_wall + newvel[3] * tof;

    }
    if (right) {

        updateBoundary = true;
        // THERMAL WALL
        // Velocity update
        rand1 = curand_uniform(&state);
        randn1= curand_normal(&state);
        randn2= curand_normal(&state);
        newvel[1] = WALL.R * beta * sqrt(-2.0f * WALL.rigT * log(rand1));
        newvel[2] = beta * sqrt(WALL.rigT) * randn1;
        newvel[3] = beta * sqrt(WALL.rigT) * randn2;
        // House keeping
        tof         = dt * (newpos[1] - WALL.RW) / (newpos[1] - old_x);
        y_at_wall   = newpos[2] - (tof/dt) * (newpos[2] - old_y); 
        z_at_wall   = newpos[3] - (tof/dt) * (newpos[3] - old_z); 
        // Position update
        newpos[1] = WALL.RW   + newvel[1] * tof;
        newpos[2] = y_at_wall + newvel[2] * tof;
        newpos[3] = z_at_wall + newvel[3] * tof;

    }
    if (bottom) {

        updateBoundary = true;
        // THERMAL WALL
        // Velocity update
        rand1 = curand_uniform(&state);
        randn1= curand_normal(&state);
        randn2= curand_normal(&state);
        newvel[2] = WALL.B * beta * sqrt(-2.0f * WALL.botT * log(rand1));
        newvel[3] = beta * sqrt(WALL.botT) * randn1;
        newvel[1] = beta * sqrt(WALL.botT) * randn2;
        // House keeping
        tof         = dt * (newpos[2] - WALL.BW) / (newpos[2] - old_y);
        z_at_wall   = newpos[3] - (tof/dt) * (newpos[3] - old_z); 
        x_at_wall   = newpos[1] - (tof/dt) * (newpos[1] - old_x); 
        // Position update
        newpos[2] = WALL.BW   + newvel[2] * tof;
        newpos[3] = z_at_wall + newvel[3] * tof;
        newpos[1] = x_at_wall + newvel[1] * tof;
        
    }
    if (top) {

        updateBoundary = true;
        // THERMAL WALL
        // Velocity update
        rand1 = curand_uniform(&state);
        randn1= curand_normal(&state);
        randn2= curand_normal(&state);
        newvel[2] = WALL.T * beta * sqrt(-2.0f * WALL.topT * log(rand1));
        newvel[3] = beta * sqrt(WALL.topT) * randn1;
        newvel[1] = beta * sqrt(WALL.topT) * randn2;
        // House keeping
        tof         = dt * (newpos[2] - WALL.TW) / (newpos[2] - old_y);
        z_at_wall   = newpos[3] - (tof/dt) * (newpos[3] - old_z); 
        x_at_wall   = newpos[1] - (tof/dt) * (newpos[1] - old_x); 
        // Position update
        newpos[2] = WALL.TW   + newvel[2] * tof;
        newpos[3] = z_at_wall + newvel[3] * tof;
        newpos[1] = x_at_wall + newvel[1] * tof;

    }
    if (aft) {

        updateBoundary = true;
        // PERIODIC WALL
        newpos[3] = fmod(newpos[3], Lz);

    }
    if (front) {

        updateBoundary = true;
        // PERIODIC WALL
        newpos[3] = fmod(newpos[3], Lz);

    }

    return updateBoundary;
}
__device__ void write_dataXYZ( float *x_in, float *y_in, float *z_in, 
                               float *x,    float *y,    float *z, 
                               int *cid, int *c_key, int *c_idx, int ppc, int adrs, int part ) {  

    // copy all particles in this cell to global memory
    while (part < ppc) {

        // update global position and velocity
        x_in[c_key[adrs + part]] = x[part];
        y_in[c_key[adrs + part]] = y[part];
        z_in[c_key[adrs + part]] = z[part];

        // update global cell id
        c_idx[adrs + part] = cid[part];

        // increment particle in case some were missed
        part += blockDim.x;
    }

}
__device__ void write_dataUVW( float *u_in, float *v_in, float *w_in, 
                               float *u,    float *v,    float *w, 
                               int *c_key, int ppc, int adrs, int part ) { 

    // copy all particles in this cell to global memory
    while (part < ppc) {

        // update global position and velocity
        u_in[c_key[adrs + part]] = u[part];
        v_in[c_key[adrs + part]] = v[part];
        w_in[c_key[adrs + part]] = w[part];

        // increment particle in case some were missed
        part += blockDim.x;
    }

}
__device__ void read_dataXYZ( float *x_in, float *y_in, float *z_in, 
                              float *x,    float *y,    float *z,
                              int *c_key, int ppc, int adrs, int part ) {
    
    // copy all particles in this cell to shared memory
    while (part < ppc) {

        // update local position and velocity
        x[part] = x_in[c_key[adrs + part]];
        y[part] = y_in[c_key[adrs + part]];
        z[part] = z_in[c_key[adrs + part]];

        // increment particle in case some are missed
        part += blockDim.x;
    }

}
__device__ void read_dataUVW( float *u_in, float *v_in, float *w_in, 
                              float *u,    float *v,    float *w,
                              int *c_key, int ppc, int adrs, int part ) {
    
    // copy all particles in this cell to shared memory
    while (part < ppc) {

        // update local position and velocity
        u[part] = u_in[c_key[adrs + part]];
        v[part] = v_in[c_key[adrs + part]];
        w[part] = w_in[c_key[adrs + part]];

        // increment particle in case some are missed
        part += blockDim.x;
    }

}
__device__ void debuggerprinter(float rand1, float rand2, float rand3, int idx, int *c_idx, int *c_key,
                                float *u, float *v, float *w, float *x, float *y, float *z) {
    
    printf("vact: %f\tvmax: %f\n",vACT, vMAX);
    printf("%i\n", c_idx[idx]);
    printf("%f\t%f\t%f\n", rand1, rand2, rand3);
    printf("u: %f, v: %f, w:%f, x: %f, y: %f, z:%f\n", u[idx], v[idx], w[idx], x[idx], y[idx], z[idx]);

}
__device__ void debuggerprinter2(int part, int ppc,
                                float *u, float *v, float *w, float *x, float *y, float *z) {
    if (part < ppc) {
        int c = particleCellIndex(x[part], y[part], z[part]);
        printf("c: %i,\tu: %12.12f,\tv: %12.12f,\tw:%12.12f,\tx: %12.12f,\ty: %12.12f,\tz:%12.12f\n", 
        c, u[part], v[part], w[part], x[part], y[part], z[part]);
    }

}
// GLOBAL
__global__ void initialise( float *x, float *y, float *z, float *u, float *v, float *w, 
                            int *c_idx, float *c_rem, float *c_dVM, int *c_key ) {
    
    // This function initialises the velocity, positions, and particle-cell index
    // of every particle in the simulation.

    // Indexing Variables
    int OGidx= threadIdx.x + blockIdx.x * blockDim.x; // particle index (global)
    int idx  = OGidx;

    // DSMC variables
    float rand1, rand2, rand3;
    curandState state;

    // Initialise particle-cell properties globally
    idx = OGidx;
    while (idx < particles) {

        // initialise particle velocities according to a Maxwell-Boltzman distribution
        curand_init((unsigned long long)clock() + idx, 0, 0, &state);
        rand1 = curand_uniform(&state);
        rand2 = curand_uniform(&state);
        rand3 = curand_uniform(&state);
        u[idx] = (2.0f * rand1 - 3.0f) * vACT / sqrt(3.0f);
        v[idx] = (2.0f * rand2 - 3.0f) * vACT / sqrt(3.0f);
        w[idx] = (2.0f * rand3 - 3.0f) * vACT / sqrt(3.0f);

        // initialise particle positions using a random uniform distribution
        rand1 = curand_uniform(&state);
        rand2 = curand_uniform(&state);
        rand3 = curand_uniform(&state);
        x[idx] = Lx * rand1;
        y[idx] = Ly * rand2;
        z[idx] = Lz * rand3;

        // cell index for this particle
        c_idx[idx] = particleCellIndex(x[idx], y[idx], z[idx]);
        c_key[idx] = idx;

        //debuggerprinter(rand1, rand2, rand3, idx, c_idx, c_key, u, v, w, x, y, z);

        // incrementing in case particles are missed
        idx += blockDim.x * gridDim.x;

    }

    // Initialise Cells
    while (idx < cells) {

        // no particles have collided initially
        c_rem[idx] = 0.0f;

        // guess for maximum relative velocity
        c_dVM[idx] = 2.0f * vMAX / 3.0f / sqrt(3.0f); 

        // increment incase cell missed
        idx += blockDim.x * gridDim.x;
    }
}
__global__ void advect( float *x_in, float *y_in, float *z_in, float *u_in, float *v_in, float *w_in, 
                            int *c_ppc, int *c_idx, int *c_key, int *c_adrs ) {
    
    // This function advects particles through space and accounts for those that escape the domain.

    // Indexing Variables
    int idx  = threadIdx.x + blockIdx.x * blockDim.x; // random number generation variable
    int part = threadIdx.x;  // particle index (within cell)
    int cell = blockIdx.x;   // current cell
    int ppc  = c_ppc[cell];  // particles in this cell
    int adrs = c_adrs[cell]; // address of first particle in this cell (sorted)
 
    // SHARED MEMORY INITIALISATION
    __shared__ float x[threadsPerBlock];
	__shared__ float y[threadsPerBlock];
	__shared__ float z[threadsPerBlock];
	__shared__ float u[threadsPerBlock];
	__shared__ float v[threadsPerBlock];
	__shared__ float w[threadsPerBlock];
    __shared__ int cid[threadsPerBlock];

    // copy data to shared memory from global
    read_dataXYZ(x_in, y_in, z_in, 
              (float *)x, (float *)y, (float *)z,
              c_key, ppc, adrs, part );
    read_dataUVW(u_in, v_in, w_in, 
              (float *)u, (float *)v, (float *)w, 
              c_key, ppc, adrs, part );
    __syncthreads();

    // ______________________________________________________________________________
    // ADVECTION
    // wall initialisation
    wall WALL;
    // particle index
    while (part < ppc)
    {
        // debugger printer 2
        // this debugger with run with particles = 5 and timesteps = 1 for validation
        // debuggerprinter2(part, ppc, (float *)u, (float *)v, (float *)w, (float *)x, (float *)y, (float *)z);
        // printf("dt: %12.12f\n", dt);

        // save old positions
        float old_x, old_y, old_z;
        old_x = x[part];
        old_y = y[part];
        old_z = z[part];

        // update position
        x[part] += u[part] * dt;
        y[part] += v[part] * dt;
        z[part] += w[part] * dt;

        // account for cells that escape through the boundaries
        float *newpos = new float[3];
        float *newvel = new float[3];
        newpos[1] = x[part];
        newpos[2] = y[part];
        newpos[3] = z[part];
        bool updateBoundary = boundaryConditions(newpos, newvel, old_x, old_y, old_z, WALL, idx);
        // only update boundary if the particle left the domain. Most particles won't leave, justifying the check.
        if (updateBoundary) {
            x[part] = newpos[1];
            y[part] = newpos[2];
            z[part] = newpos[3];
            u[part] = newvel[1];
            v[part] = newvel[2];
            w[part] = newvel[3];
        }
        delete [] newpos;
        delete [] newvel;

        // debugger printer 2
        // this debugger with run with particles = 5 and timesteps = 1 for validation
        // debuggerprinter2(part, ppc, (float *)u, (float *)v, (float *)w, (float *)x, (float *)y, (float *)z);

        // reindex particles into cells & recreate key
        cid[part] = particleCellIndex(x[part], y[part], z[part]);

        // increment idx to account for missed particles
        part += blockDim.x;
    }
    __syncthreads();
    part = threadIdx.x;

    // copy data from shared memory to global
    write_dataXYZ(x_in, y_in, z_in,
                  (float *)x, (float *)y, (float *)z, 
                  (int *)cid, c_key, c_idx, ppc, adrs, part);
    write_dataUVW(u_in, v_in, w_in, 
                  (float *)u, (float *)v, (float *)w, 
                  c_key, ppc, adrs, part);
    __syncthreads();
    // *****************************************************************************
}
__global__ void collisions(float *u_in, float *v_in, float *w_in, 
        int *c_ppc, int *c_adrs, int *c_key, int *c_idx, float *c_dVM, float *c_rem) {
    
    // This function loops over all potential collisions in a cell and collides particles statistically.

    // Indexing Variables
    int idx  = threadIdx.x + blockIdx.x * blockDim.x; // random number generation variable
    int part = threadIdx.x;  // particle index (within cell)
    int cell = blockIdx.x;   // current cell
    int ppc  = c_ppc[cell];  // particles in this cell
    int adrs = c_adrs[cell]; // address of first particle in this cell (sorted)
 
    // SHARED MEMORY INITIALISATION
	__shared__ float u[threadsPerBlock];
	__shared__ float v[threadsPerBlock];
	__shared__ float w[threadsPerBlock];

    // copy data to shared memory from global
    read_dataUVW(u_in, v_in, w_in, 
                 (float *)u, (float *)v, (float *)w, 
                 c_key, ppc, adrs, part );
    __syncthreads();

    // _____________________________________________________________________________
    // COLLISIONS
    // cfor all particles in current cell
    while (part < ppc)
    {
        // two particle index initialisation
        int p1;
        int p2;
        float rand1, rand2, rand3;
        float vRel, vRelX, vRelY, vRelZ;

        // collision parameters
        int cols;
        float truecols, cosXi, sinXi, theta;
        float VcomX, VcomY, VcomZ;
        
        // omit cells with single part
        if (ppc > 1) {

            // potential number of collisions
            truecols = coefNTC * ppc * (ppc - 1) * c_dVM[cell] + c_rem[cell];
            cols = (int) truecols;
            c_rem[cell] = truecols - (float) cols;

            // loop over all potential number of collisions in cell
            for (int i = 0; i < cols; i++) {
                
                // pick any two particles at random
                curandState state;
                curand_init((unsigned long long)clock() + idx, 0, 0, &state);
                rand1 = curand_uniform(&state); 
                rand2 = curand_uniform(&state);
                p1 = c_key[(int) (ppc * rand1)];
                p2 = c_key[(int) (ppc * rand2)];

                // calculate relative velocity between the selected particles
                vRel = sqrt(pow(u[p1] - u[p2], 2) + 
                            pow(v[p1] - v[p2], 2) + 
                            pow(w[p1] - w[p2], 2));

                // update current maximum relative velocity
                if (vRel > c_dVM[cell]) c_dVM[cell] = vRel;

                // collide only if the following criteria is true
                rand1 = curand_uniform(&state);
                if (vRel / c_dVM[cell] > rand1) {

                    // center of mass velocity
                    VcomX = (u[p1] + u[p2]) / 2;
                    VcomY = (v[p1] + v[p2]) / 2;
                    VcomZ = (w[p1] + w[p2]) / 2;

                    // center of mass frame of reference (see documentation)
                    rand1 = curand_uniform(&state); 
                    rand2 = curand_uniform(&state); 
                    rand3 = curand_uniform(&state);
                    cosXi = 2 * rand1 - 1;
                    sinXi = sqrt( 1.0f - pow(cosXi, 2) );
                    theta = 2 * PI * rand3;

                    // finding the position collision velocities
                    vRelX = vRel * cosXi;
                    vRelY = vRel * sinXi * cos(theta);
                    vRelZ = vRel * sinXi * sin(theta);
                    // first particle
                    u[p1] = VcomX + vRelX;
                    v[p1] = VcomY + vRelY;
                    w[p1] = VcomZ + vRelZ;
                    // second particle
                    u[p2] = VcomX - vRelX;
                    v[p2] = VcomY - vRelY;
                    w[p2] = VcomZ - vRelZ;
                }
            }
        }

        // increment part to account for missed particles
        part += blockDim.x;
    }
    __syncthreads();
    part = threadIdx.x;

    // copy data from shared memory to global
    write_dataUVW(u_in, v_in, w_in, 
                  (float *)u, (float *)v, (float *)w, 
                  c_key, ppc, adrs, part);
    __syncthreads();
    // *****************************************************************************
}
__global__ void sampling(float *u_in, float *v_in, float *w_in,
        int *c_ppc, int *c_adrs, int *c_key, float *save_u,
        float *save_v, float *save_w, float *save_T, float *save_rho, int step) {
    // _____________________________________________________________________________
    // RETRIEVE DATA
    // Indexing Variables
    int part = threadIdx.x;  // particle index (within cell)
    int cell = blockIdx.x;   // current cell
    int ppc  = c_ppc[cell];  // particles in this cell
    int adrs = c_adrs[cell]; // address of first particle in this cell (sorted)
 
    // SHARED MEMORY INITIALISATION
	__shared__ float u[threadsPerBlock];
	__shared__ float v[threadsPerBlock];
	__shared__ float w[threadsPerBlock];
    __shared__ float u_cache[threadsPerBlock];
    __shared__ float v_cache[threadsPerBlock];
    __shared__ float w_cache[threadsPerBlock];
    __shared__ float v2_cache[threadsPerBlock];

    // copy data to shared memory from global
    read_dataUVW(u_in, v_in, w_in, 
                 (float *)u, (float *)v, (float *)w, 
                 c_key, ppc, adrs, part );
    __syncthreads();

    // temporary sums
    float temp_u    = 0;
    float temp_v    = 0;
    float temp_w    = 0;
    float temp_v2   = 0;
    float temp_T    = 0;
    float temp_rho  = 0;
    
    while (part < ppc) {

        // aquire sums
        temp_u  += u[part] / ppc;
        temp_v  += v[part] / ppc;
        temp_w  += w[part] / ppc;
        temp_v2 += (pow(u[part], 2) + pow(v[part], 2) + pow(w[part], 2)) / ppc;

        // increment in case particles were missed
        part += blockDim.x;  

    }
    part = threadIdx.x;
    u_cache[part] = temp_u;
    v_cache[part] = temp_v;
    w_cache[part] = temp_w;
    v2_cache[part] = temp_v2;
    __syncthreads();

    // reductions on sums
    int i = blockDim.x / 2;
    while (i != 0) {

        if (part < i) {
            u_cache[part] += u_cache[part + i];
            v_cache[part] += v_cache[part + i];
            w_cache[part] += w_cache[part + i];
            v2_cache[part] += v2_cache[part + i];
        }
        __syncthreads();
        i /= 2;
    }

    // the outcome of the reduction will be the sum property(s) of all particles in this cell
    // writtne to the temporary variables
    if (part == 0) {

        temp_u = u_cache[0];
        temp_v = v_cache[0];
        temp_w = w_cache[0];
        temp_v2 = v2_cache[0];

        temp_T   = m / 3 / k * (temp_v2 - (pow(temp_u, 2) + pow(temp_v, 2) + pow(temp_w, 2)));
        temp_rho = m * nPARTeff / volCell * ppc;

        // saving data (FINAL)
        save_u[cell + timesteps * step]  = temp_u;
        save_v[cell + timesteps * step]  = temp_v;
        save_w[cell + timesteps * step]  = temp_w;
        save_T[cell + timesteps * step]  = temp_T;
        save_rho[cell + timesteps * step]= temp_rho;
    }
    __syncthreads();

    if (cell == 555 && (step % (timesteps / frequency) == 0 && part == 0)) {
        printf("Center cell Temperature: %f K\n", temp_T);
    }
    // *****************************************************************************
}
__global__ void determineAddress( int idx, int *SUM_CELL, int *c_idx, int *c_adrs, int *c_key ) {
    
    // This function determines the cell address of particles, and how many particles are in each cell.

    __shared__ int cache[threadsPerBlock];
    int bucketID   = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    int tempcount  = 0;

    // Assume particles are in buckets, where the order of the buckets is linear and never changes. I.e.
    // BUCKET   PARTICLE_IN_BUCKET[bucket]  PARTICLE_IN_CELL[bucket]
    // 0        0                           17
    // 1        1                           4
    // 2        2                           13
    // 3        3                           900
    // ****** perform sorting operation ******
    // BUCKET   PARTICLE_IN_BUCKET[bucket]  PARTICLE_IN_CELL[bucket]
    // 0        1                           4
    // 1        2                           13
    // 2        0                           17
    // 3        3                           900
    // then, we are in the current cell if the bucket contains the cell. I.e., c_idx[i].
    // PARTICLE_IN_BUCKET refers to c_key[i].

    // for all buckets, check if they are in the current cell
    while (bucketID < particles) {

        // if current bucket refers to cell, then a particle must be in this cell
        if (c_idx[bucketID] == idx) {

            // increment particle counter
            tempcount++;

            // the zeroth bucket refers to sorted cells, and hence the particle in the bucket will  
            // by definition be the first particle in that cell. bucketID can only be zero ONCE.
            if (bucketID == 0) {
                c_adrs[idx] = 0;
            }
            // for all other buckets, it must be checked to see if that bucket is the first mention
            // of the current cell. I.e., if previous bucket does NOT contain the current cell ID...
            else if (c_idx[bucketID] != c_idx[bucketID - 1]) {

                // save the key address for this cell
                c_adrs[idx] = bucketID;

            }
        }
        bucketID += blockDim.x * gridDim.x;
    }
    cache[cacheIndex] = tempcount;

    // syncronise threads in this block
    __syncthreads();

    // perform reduction on cache
    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    // the  outcome of the reduction will be the total particles in this cell, written to the first thread
    if (cacheIndex == 0)
        SUM_CELL[blockIdx.x] = cache[0];

}
__global__ void findparticles( int* c_idx, int *c_ppc, int *c_adrs, int *c_key ) {
    // _____________________________________________________________________________
    // SORTING PARTICLE ARRAYS BASED ON c
    // determine addresses and particle counts
    int partial_c = 0;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    while (idx < cells) {
        
        // temporary sum array required for reduction
        int *SUM_CELL = new int[blocksPerGrid];

        // address and partial sums are found with this kernel call
        // kernel called for every cell, so it is parallelised
        determineAddress<<< blocksPerGrid , threadsPerBlock >>>(
            idx, SUM_CELL, c_idx, c_adrs, c_key );
        // ensure all threads in previous kernel call have written to SUM_CELL before continuing
        cudaDeviceSynchronize(); 

        // particle counts - reduction step
        for (int i = 0; i < blocksPerGrid; i++) {
            partial_c += SUM_CELL[i];
        }
        c_ppc[idx] = partial_c;

        delete [] SUM_CELL;

        // increment idx to account for missed cells
        idx += blockDim.x * gridDim.x;

    }
    // *****************************************************************************
}
// DEBUGGING
void debuggercells( data *c ) {

    int saveSimP = particles * sizeof(float);
    int *idx, *key; 
    idx = (int*)calloc( particles, sizeof(int) );
    key = (int*)calloc( particles, sizeof(int) );
    

    HANDLE_ERROR( cudaMemcpy( idx, c->c_idx, saveSimP,
                              cudaMemcpyDeviceToHost ) );
    HANDLE_ERROR( cudaMemcpy( key, c->c_key, saveSimP,
                              cudaMemcpyDeviceToHost ) );

    printf("i\t\tc_idx[i]\t\tc_key[i]\n------------------------\n");
    for (int i = 0; i < particles; i++) {
        printf("%i\t\t%i\t\t\t\t\t%i\n", i, idx[i], key[i]);
    }
    printf("\n\n");

    HANDLE_ERROR( cudaMemcpy( c->c_idx, idx, saveSimP,
                              cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( c->c_key, key, saveSimP,
                              cudaMemcpyHostToDevice ) );
    free( idx );
    free( key );
     
}
void debuggercells2( data *c ) {

    int saveSimP = particles * sizeof(float);
    int saveSimC = h_cells   * sizeof(int); 
    int *idx, *key, *ppc, *adrs; 
    idx = (int*)calloc( particles, sizeof(int) );
    key = (int*)calloc( particles, sizeof(int) );
    ppc = (int*)calloc( h_cells, sizeof(int) );
    adrs= (int*)calloc( h_cells, sizeof(int) );
    

    HANDLE_ERROR( cudaMemcpy( idx, c->c_idx, saveSimP,
                              cudaMemcpyDeviceToHost ) );
    HANDLE_ERROR( cudaMemcpy( key, c->c_key, saveSimP,
                              cudaMemcpyDeviceToHost ) );
    HANDLE_ERROR( cudaMemcpy( ppc, c->c_ppc, saveSimC,
                              cudaMemcpyDeviceToHost ) );                          
    HANDLE_ERROR( cudaMemcpy( adrs, c->c_adrs, saveSimC,
                              cudaMemcpyDeviceToHost ) );    

    // printf("i\t\tc_idx[i]\t\tc_key[i]\t\tc_idx[c_key[i]]\t\tc_ppc[c_idx[c_key[i]]]\t\tc_adrs[c_idx[c_key[i]]]\n");
    // printf("-----------------------------------------------------------------------------------------------\n");
    // for (int i = 0; i < particles; i++) {
    //     printf("%i\t\t%i\t\t\t\t\t%i\t\t\t\t\t\t%i\t\t\t\t\t\t\t\t%i\t\t\t\t\t\t\t\t\t\t\t\t\t%i\n", 
    //             i, idx[i], key[i], idx[key[i]], ppc[idx[key[i]]], adrs[idx[key[i]]]);
    // }
    // printf("\n\n");

    // GENERIC INFORMATION
    printf("i\t\tc_idx[i]\t\tc_key[i]\t\tc_ppc[c_idx[i]]\t\tc_adrs[c_idx[i]]\n");
    printf("--------------------------------------------------------------\n");
    for (int i = 0; i < particles; i++) {
        printf("%i\t\t%i\t\t\t\t\t%i\t\t\t\t\t\t%i\t\t\t\t\t\t\t\t\t%i\n", 
                i, idx[i], key[i], ppc[idx[i]], adrs[idx[i]]);
    }
    printf("\n\n");

    // CHECKING IF ANY CELL HAS MORE THAN 0 PARTICLES
    // for (int i = 0; i < h_cells; i++) {
    //     if (ppc[i] != 0) {
    //         printf("ppc[%i] = %i\n", i, ppc[i]);
    //     }
    // }

    HANDLE_ERROR( cudaMemcpy( c->c_idx, idx, saveSimP,
                              cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( c->c_key, key, saveSimP,
                              cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( c->c_ppc, ppc, saveSimC,
                              cudaMemcpyHostToDevice ) );                          
    HANDLE_ERROR( cudaMemcpy( c->c_adrs, adrs, saveSimC,
                              cudaMemcpyHostToDevice ) );   
    free( idx );
    free( key );
    free( ppc );
    free( adrs);
     
}
void debuggerparticles( data *p ) {

    int saveSimPa = particles * sizeof(float);
    float *x, *y, *z, *u, *v, *w; 
    x = (float*)calloc( particles, sizeof(float) );
    y = (float*)calloc( particles, sizeof(float) );
    z = (float*)calloc( particles, sizeof(float) );
    u = (float*)calloc( particles, sizeof(float) );
    v = (float*)calloc( particles, sizeof(float) );
    w = (float*)calloc( particles, sizeof(float) );  

    HANDLE_ERROR( cudaMemcpy( x, p->dev_x, saveSimPa,
                              cudaMemcpyDeviceToHost ) );
    HANDLE_ERROR( cudaMemcpy( y, p->dev_y, saveSimPa,
                              cudaMemcpyDeviceToHost ) );
    HANDLE_ERROR( cudaMemcpy( z, p->dev_z, saveSimPa,
                              cudaMemcpyDeviceToHost ) );                          
    HANDLE_ERROR( cudaMemcpy( u, p->dev_u, saveSimPa,
                              cudaMemcpyDeviceToHost ) );  
    HANDLE_ERROR( cudaMemcpy( v, p->dev_v, saveSimPa,
                              cudaMemcpyDeviceToHost ) );  
    HANDLE_ERROR( cudaMemcpy( w, p->dev_w, saveSimPa,
                              cudaMemcpyDeviceToHost ) );  

    // do debugging
    int pidx, cid, gx, gy, gz;
    printf("Debug particles\n");
    for (int i = 0; i < particles; i++) {

        // correct particle
        pidx = i;

        // grid locations
        gx = (int) (x[pidx] / h_dLx);
        gy = (int) (y[pidx] / h_dLy);
        gz = (int) (z[pidx] / h_dLz);

        // 1D grid location
        cid = gx + DIMx * gy + DIMx * DIMy * gz;

        // debug
        printf("c: %i,\tu: %12.12f,\tv: %12.12f,\tw:%12.12f,\tx: %12.12f,\ty: %12.12f,\tz:%12.12f\n", 
        cid, u[pidx], v[pidx], w[pidx], x[pidx], y[pidx], z[pidx]);

    }
    printf("\n\n");
    

    // copy back and free
    HANDLE_ERROR( cudaMemcpy( p->dev_x, x, saveSimPa,
                              cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( p->dev_y, y, saveSimPa,
                              cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( p->dev_z, z, saveSimPa,
                              cudaMemcpyHostToDevice ) );                          
    HANDLE_ERROR( cudaMemcpy( p->dev_u, u, saveSimPa,
                              cudaMemcpyHostToDevice ) );  
    HANDLE_ERROR( cudaMemcpy( p->dev_v, v, saveSimPa,
                              cudaMemcpyHostToDevice ) );  
    HANDLE_ERROR( cudaMemcpy( p->dev_w, w, saveSimPa,
                              cudaMemcpyHostToDevice ) );

    free( x );
    free( y );
    free( z );
    free( u );
    free( v );
    free( w );

}
// MAINS
void simulation( data *p, data *c , data *s) {

    // global initialisation
    initialise<<< blocksPerGrid, threadsPerBlock >>>
        ( p->dev_x, p->dev_y, p->dev_z, p->dev_u, p->dev_v, p->dev_w,
          c->c_idx, c->c_rem, c->c_dVM, c->c_key );
    cudaDeviceSynchronize();

    // debuggercells(c);

    // SORTING & KEY
    thrust::device_ptr<int> dev_c_idx( c->c_idx );
    thrust::device_ptr<int> dev_c_key( c->c_key );
    thrust::sort_by_key( dev_c_idx, dev_c_idx + particles, dev_c_key );
    thrust::sort( dev_c_idx, dev_c_idx + particles );

    // debuggercells(c);

    // ADDRESS FINDING
    findparticles<<< blocksPerGrid, threadsPerBlock >>>( c->c_idx, c->c_ppc, c->c_adrs, c->c_key );
    cudaDeviceSynchronize();

    // printf("After initial sorting\n");
    // debuggercells2(c);
    // debuggerparticles(p);

    // SIMULATION LOOP
    for (int step = 0; step < timesteps; step++) {

        // ADVECTION
        advect<<< blocksPerGrid, threadsPerBlock >>>
            ( p->dev_x, p->dev_y, p->dev_z, p->dev_u, p->dev_v, p->dev_w, 
              c->c_ppc, c->c_idx, c->c_key, c->c_adrs );
        cudaDeviceSynchronize();

        // SORTING
        thrust::device_ptr<int> dev_c_idx( c->c_idx );
        thrust::device_ptr<int> dev_c_key( c->c_key );
        thrust::sort_by_key( dev_c_idx, dev_c_idx + particles, dev_c_key );
        thrust::sort( dev_c_idx, dev_c_idx + particles );

        // ADDRESS FINDING
        findparticles<<< blocksPerGrid, threadsPerBlock >>>( c->c_idx, c->c_ppc, c->c_adrs, c->c_key );
        cudaDeviceSynchronize();

        // this debugger was run with particles = 5 and timesteps = 1 for validation
        // printf("After Advection\n");
        // debuggerparticles(p);

        // COLLISIONS
        collisions<<< blocksPerGrid, threadsPerBlock >>>
            ( p->dev_u, p->dev_v, p->dev_w, c->c_ppc, c->c_adrs, c->c_key, c->c_idx, c->c_dVM, c->c_rem );
        cudaDeviceSynchronize();

        // this debugger was run with particles = 5 and timesteps = 1 for validation
        // printf("After Collisions\n");
        // debuggerparticles(p);

        // SAMPLING
        // sampling<<< blocksPerGrid, threadsPerBlock >>>
        //    ( p->dev_u, p->dev_v, p->dev_w, c->c_ppc, c->c_adrs, c->c_key, 
        //      s->dev_u, s->dev_v, s->dev_w, s->dev_T, s->dev_rho, step);
        // cudaDeviceSynchronize();

        // update user of simulation progress
        if (step % ((int)(timesteps / frequency)) == 0) {
            printf("TS: %i\n", step);
        }
    }
    printf("TS: %i\n", timesteps); 
}
int main( void ) {

    // _________________________________________________________________________________________
    // ALLOCATION
    // Initialisation variable
    float h_vMAX = vmax();
    float h_vACT = vactual(h_vMAX);

    // Allocate constant variables to device
    HANDLE_ERROR( cudaMemcpyToSymbol( cells, &h_cells,                       sizeof(int) ));
    HANDLE_ERROR( cudaMemcpyToSymbol( vMAX, &h_vMAX,                         sizeof(float) ));
    HANDLE_ERROR( cudaMemcpyToSymbol( vACT, &h_vACT,                         sizeof(float) ));
    HANDLE_ERROR( cudaMemcpyToSymbol( n, &h_n,                               sizeof(float) ));
    HANDLE_ERROR( cudaMemcpyToSymbol( wallTemp, &h_wallTemp,                 sizeof(float) ));
    HANDLE_ERROR( cudaMemcpyToSymbol( maxTemp, &h_maxTemp,                   sizeof(float) ));
    HANDLE_ERROR( cudaMemcpyToSymbol( minTemp, &h_minTemp,                   sizeof(float) ));
    HANDLE_ERROR( cudaMemcpyToSymbol( MBP, &h_MBP,                           sizeof(float) ));
    HANDLE_ERROR( cudaMemcpyToSymbol( Lx, &h_Lx,                             sizeof(float) ));
    HANDLE_ERROR( cudaMemcpyToSymbol( Ly, &h_Ly,                             sizeof(float) ));
    HANDLE_ERROR( cudaMemcpyToSymbol( Lz, &h_Lz,                             sizeof(float) ));
    HANDLE_ERROR( cudaMemcpyToSymbol( N_dt_tau, &h_N_dt_tau,                 sizeof(float) ));
    HANDLE_ERROR( cudaMemcpyToSymbol( PI, &h_PI,                             sizeof(float) ));
    HANDLE_ERROR( cudaMemcpyToSymbol( k, &h_k,                               sizeof(float) ));
    HANDLE_ERROR( cudaMemcpyToSymbol( m, &h_m,                               sizeof(float) ));
    HANDLE_ERROR( cudaMemcpyToSymbol( d, &h_d,                               sizeof(float) ));
    HANDLE_ERROR( cudaMemcpyToSymbol( dLx, &h_dLx,                           sizeof(float) ));
    HANDLE_ERROR( cudaMemcpyToSymbol( dLy, &h_dLy,                           sizeof(float) ));
    HANDLE_ERROR( cudaMemcpyToSymbol( dLz, &h_dLz,                           sizeof(float) ));
    HANDLE_ERROR( cudaMemcpyToSymbol( volCell, &h_volCELL,                   sizeof(float) ));
    HANDLE_ERROR( cudaMemcpyToSymbol( nPARTeff, &h_nPARTeff,                 sizeof(float) ));
    HANDLE_ERROR( cudaMemcpyToSymbol( vMP, &h_vMP,                           sizeof(float) ));
    HANDLE_ERROR( cudaMemcpyToSymbol( vAVE, &h_vAVE,                         sizeof(float) ));
    HANDLE_ERROR( cudaMemcpyToSymbol( dt, &h_dt,                             sizeof(float) ));
    HANDLE_ERROR( cudaMemcpyToSymbol( coefNTC, &h_coefNTC,                   sizeof(float) ));

    // Initialise data structures
    data save;
    data simP;
    data simC;
    // Data structure sizes: 
    // - save:   floating point simulation, all XYZ cells, all timesteps
    // - simP:   floating point particles, all particles, single timestep
    // - simC:   int cells, all cells, single timestep
    int saveSize = DIMx * DIMy * DIMz * timesteps * sizeof(float);
    int saveSimP = particles * sizeof(float);
    int saveSimPi= particles * sizeof(int);
    int saveSimC = h_cells   * sizeof(int);
    int saveSimCf= h_cells   * sizeof(float);

    // Allocate arrays to device && host
    // save
    HANDLE_ERROR( cudaMalloc( (void**)&save.dev_u, saveSize ) );
    HANDLE_ERROR( cudaMalloc( (void**)&save.dev_v, saveSize ) );
    HANDLE_ERROR( cudaMalloc( (void**)&save.dev_w, saveSize ) );
    HANDLE_ERROR( cudaMalloc( (void**)&save.dev_T, saveSize ) );
    HANDLE_ERROR( cudaMalloc( (void**)&save.dev_rho, saveSize ) );
    // simPart
    HANDLE_ERROR( cudaMalloc( (void**)&simP.dev_u, saveSimP ) );
    HANDLE_ERROR( cudaMalloc( (void**)&simP.dev_v, saveSimP ) );
    HANDLE_ERROR( cudaMalloc( (void**)&simP.dev_w, saveSimP ) );
    HANDLE_ERROR( cudaMalloc( (void**)&simP.dev_x, saveSimP ) );
    HANDLE_ERROR( cudaMalloc( (void**)&simP.dev_y, saveSimP ) );
    HANDLE_ERROR( cudaMalloc( (void**)&simP.dev_z, saveSimP ) );
    // simCell
    HANDLE_ERROR( cudaMalloc( (void**)&simC.c_idx, saveSimPi ) );
    HANDLE_ERROR( cudaMalloc( (void**)&simC.c_key, saveSimPi ) );

    HANDLE_ERROR( cudaMalloc( (void**)&simC.c_ppc, saveSimC ) );
    HANDLE_ERROR( cudaMalloc( (void**)&simC.c_adrs,saveSimC ) );
    HANDLE_ERROR( cudaMalloc( (void**)&simC.c_rem, saveSimCf ) );
    HANDLE_ERROR( cudaMalloc( (void**)&simC.c_dVM, saveSimCf ) );
    // ****************************************************************************************




    // _______________________________________________________________________________
    // SIMULATIONS
    // Timer variables
    cudaEvent_t     start, stop;
    float           totalTime;
    // Start timer (outside of all simulations)
    HANDLE_ERROR( cudaEventCreate( &start ) );
    HANDLE_ERROR( cudaEventCreate( &stop ) );
    HANDLE_ERROR( cudaEventRecord( start, 0 ) );

    // function call that goes to the simulation loop
    simulation( &simP, &simC, &save );

    // end timer (outside of all simulations)
    HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
    HANDLE_ERROR( cudaEventSynchronize( stop ) );
    HANDLE_ERROR( cudaEventElapsedTime( &totalTime, start, stop ) );
    HANDLE_ERROR( cudaEventDestroy( start ) );
    HANDLE_ERROR( cudaEventDestroy( stop ) );
    printf( "Time to complete all simulations:  \t%3.1f ms\n", totalTime );
    // ******************************************************************************




    // ______________________________________________________________________________
    // MEMORY MANAGEMENT
    // free memory on the gpu side
    // save
    HANDLE_ERROR( cudaFree( save.dev_u ) );
    HANDLE_ERROR( cudaFree( save.dev_v ) );
    HANDLE_ERROR( cudaFree( save.dev_w ) );
    HANDLE_ERROR( cudaFree( save.dev_T ) );
    HANDLE_ERROR( cudaFree( save.dev_rho ) );
    // simPart
    HANDLE_ERROR( cudaFree( simP.dev_x ) );
    HANDLE_ERROR( cudaFree( simP.dev_y ) );
    HANDLE_ERROR( cudaFree( simP.dev_z ) );
    HANDLE_ERROR( cudaFree( simP.dev_u ) );
    HANDLE_ERROR( cudaFree( simP.dev_v ) );
    HANDLE_ERROR( cudaFree( simP.dev_w ) );
    // simCell
    HANDLE_ERROR( cudaFree( simC.c_idx ) );
    HANDLE_ERROR( cudaFree( simC.c_key ) );
    HANDLE_ERROR( cudaFree( simC.c_ppc ) );
    HANDLE_ERROR( cudaFree( simC.c_adrs ) );
    HANDLE_ERROR( cudaFree( simC.c_rem ) );
    HANDLE_ERROR( cudaFree( simC.c_dVM ) );
    // ******************************************************************************
}
// *********************************************************************************************