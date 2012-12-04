#ifndef JETSUB5B_H
#define JETSUB5B_H

//EKS common blocks 
struct eks_input
{
  float nflavor, lambda;
  int nfl;
  float rts;
  float r;
  float muuvoverpj, mucooverpj;
  bool stypea,stypeb,stypec,styped;
  int calctype;
  bool sborn, svirtual, sacoll, sasoft, sbcoll, sbsoft, s1coll, s1soft, s2coll, s2soft;
  bool s2to2, safinite, sbfinite, s1finite, s2finite;
  bool ppbar;
  float yscale, pscale;
  float pjmin, pjmax, yjmax;
  float ptresolution, angleresolution;
  //dimensions of phase space
  int ndim;
} ;


//integrate functions
__device__ void integrate2to2(float, float, float, float, float, int*, float* , float* pdf);
__device__ void integratea(float, float, float, float, float, float, float, float, int*, float*);
__device__ void integrateb(float, float, float, float, float, float, float, float, int*, float*);
__device__ void integrate1(float, float, float, float, float, float, float, float, int*, float*);
__device__ void integrate2(float, float, float, float, float, float, float, float, int*, float*);


// parton 
__device__ float effparton(float, float, float, int, float* pdf);
__device__ float alphas(float);
__device__ float parton(int, int, float* pdf);
__device__ float altarelli(int, int, float);
__device__ float altrliprime(int, int, float);
__device__ float find_z(int);

// general functions
__device__ float theta( bool condition );
__device__ float li2(float);
__device__ float psitilde4(int, float[4][4]);
__device__ float psitilde6ns(int, float[4][4]);
__device__ float psictilde(int, int, int, float[4][4]);
__device__ float gammaval(int );
__device__ float gammaprime(int );
__device__ float color(int );
__device__ float convert(float);
__device__ float wnum(float);


// sub routines
__device__ void residueinit();
__device__ void setmu(int, float, float, float, float, float, float, float, float, float, float*, float*);
__device__ void report(int, float, float, float, float, float, float, float, float, float, float, int*, float*);

__global__ void GPU_submit(int*, float*, float* pdf);
__device__ void reno(int*, float* , float* pdf);


// test function
__global__ void test_GPU(int* event_index_on_device, float* warehouse_on_device, float* pdf_on_device );


//special functions
__device__ void initialize(int*, float*);
__device__ void finish();


//declare input arrays
//memory 
__constant__ eks_input eks_setting_on_device;
  // radom variables
__constant__ float input_variables[5000];

/* __shared__ float container_out[200]; */

__constant__ float alphas_input[1000];

__constant__ float  z_input[2000];



#endif
