#include "eks.h"

//========================================================================

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
      std::cout <<  cudaGetErrorString( err ) << " in " << file << " at line " << line  << std::endl;
      exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

//========================================================================
eks::eks()
{
  
}
//========================================================================
eks::~eks()
{

}
//========================================================================

void eks::read_card(card& cd)
{
  /*read parameters from input card and init EKS program;
   */

  //  Switch for p-pbar collisions (if true) or p-p collisions (if false)
  eks_input_variables.ppbar = cd.ppbar_collider;
  
  //  Jet physics is normally in the range of five flavors, so we
  //  set the number of flavors to 5. (This could be changed for special
  //  purposes, but the program works with a fixed NFL. Scheme switching
  //  for as mu becomes greater than a heavy quark mass is not 
  //  implemented.)
  eks_input_variables.nflavor = 5.0;
  eks_input_variables.nfl = 5;

  //  Physics parameters:
  eks_input_variables.rts = cd.square_s;
  
  //  Parton resolution parameters in PT and angle.
  eks_input_variables.ptresolution = 0.001;
  eks_input_variables.angleresolution = 0.001;

  eks_input_variables.lambda = 0.226;

  //  Ratio of renormalization scale MUUV and factorization
  //  scale MUCO to PJ:
  eks_input_variables.muuvoverpj = cd.renomalization_scale;
  eks_input_variables.mucooverpj = cd.factorization_scale;
  
  // Scales to tell RENO where to put rapidities and transverse
  //   momenta for partons 1 and 2.
  eks_input_variables.yscale = 1.2;
  eks_input_variables.pscale = 0.06 * cd.square_s;
  
  // Limits on PT and ABS(Y).
  eks_input_variables.pjmin = 0.003 * cd.square_s;
  eks_input_variables.pjmax = 0.5 *  cd.square_s;
  eks_input_variables.yjmax = 4.0;
  
  //  Switches (should all be .TRUE.)
  eks_input_variables.stypea = true;
  eks_input_variables.stypeb = true;
  eks_input_variables.stypec = true;
  eks_input_variables.styped = true;

  eks_input_variables.sborn = true;
  eks_input_variables.svirtual = true;
  eks_input_variables.sacoll = true;
  eks_input_variables.sasoft = true;
  eks_input_variables.sbcoll = true;
  eks_input_variables.sbsoft = true;
  eks_input_variables.s1coll = true;
  eks_input_variables.s1soft = true;
  eks_input_variables.s2coll = true;
  eks_input_variables.s2soft = true;

  eks_input_variables.s2to2 = true;
  eks_input_variables.safinite = true;
  eks_input_variables.sbfinite = true;
  eks_input_variables.s1finite = true;
  eks_input_variables.s2finite = true;
  
  if( ! cd.next_to_leading_order )
    {
      eks_input_variables.svirtual = false;
      eks_input_variables.sacoll = false;
      eks_input_variables.sasoft = false;
      eks_input_variables.sbcoll = false;
      eks_input_variables.sbsoft = false;
      eks_input_variables.s1coll = false;
      eks_input_variables.s1soft = false;
      eks_input_variables.s2coll = false;
      eks_input_variables.s2soft = false;

      eks_input_variables.safinite = false;
      eks_input_variables.sbfinite = false;
      eks_input_variables.s1finite = false;
      eks_input_variables.s2finite = false;
      std::cout << "BORN Calculation only!"<<std::endl;
    }
  
  // GPU computing
  number_of_blocks = cd.number_of_blocks;
  number_of_threads = cd.number_of_threads;
  
  //
  if(cd.next_to_leading_order)
    {
      eks_input_variables.ndim = 7;
    }
  else
    {
      eks_input_variables.ndim = 4;
    }
  
}

//========================================================================

float eks::integrand(int* event_index, const double* xx,  float* warehouse, float* alphas_in, float* pdf_in,  float* z, int n_events)
{
  
  
  // number of input variables
  int n_variables = n_events * eks_input_variables.ndim;

  // translate double to float
  float input[n_variables];
  for(int i = 0; i < n_variables; i++)
    {
      input[i] = xx[i];
    }
  
  // float* temp = (float*)malloc(17*11*n_events*sizeof(float))
 
  // //declare input arrays
  int* event_index_on_device;
  float* warehouse_on_device;
  float* pdf_on_device;
  

  // allocate the memory on the GPU
  HANDLE_ERROR( cudaMalloc( (void**) &warehouse_on_device, 17 * 11 * n_events * sizeof(float) ) );
  HANDLE_ERROR( cudaMalloc( (void**) &event_index_on_device, n_events * sizeof(int) ) );
  HANDLE_ERROR( cudaMalloc( (void**) &pdf_on_device, 44*n_events * sizeof(float) ) );

  // HANDLE_ERROR( cudaMalloc( (void**) &eks_setting_on_device, sizeof(eks_input) ) );
  // HANDLE_ERROR( cudaMalloc( (void**) &input_variables, n_variables * sizeof(float) ) );
  // HANDLE_ERROR( cudaMalloc( (void**) &pdf_input, 44 * n_events * sizeof(float) ) );
  // HANDLE_ERROR( cudaMalloc( (void**) &alphas_input,  n_events * sizeof(float) ) );
  // HANDLE_ERROR( cudaMalloc( (void**) &z_input, 2 * n_events * sizeof(float)  ) );
  
  //copy  input variables to the GPU
  // HANDLE_ERROR( cudaMemcpy( warehouse_on_device, warehouse, 17 * 11 * n_events * sizeof(float), cudaMemcpyHostToDevice ) );
  HANDLE_ERROR( cudaMemcpy( pdf_on_device, pdf_in,  44 * n_events * sizeof(float), cudaMemcpyHostToDevice ) );

  //---------------
  HANDLE_ERROR( cudaMemcpyToSymbol( eks_setting_on_device, &eks_input_variables,  sizeof(eks_input), 0, cudaMemcpyHostToDevice ) );
  HANDLE_ERROR( cudaMemcpyToSymbol( input_variables, input,  n_variables * sizeof(float), 0, cudaMemcpyHostToDevice ) );
  HANDLE_ERROR( cudaMemcpyToSymbol( alphas_input, alphas_in,  n_events * sizeof(float), 0, cudaMemcpyHostToDevice ) );
  HANDLE_ERROR( cudaMemcpyToSymbol( z_input, z, 2 * n_events * sizeof(float) , 0, cudaMemcpyHostToDevice ) );
 
  //copy input variables to the GPU
  // HANDLE_ERROR( cudaMemcpyToSymbol( eks_setting_on_device, &eks_input_variables, sizeof(eks_input_variables) ) );
  // HANDLE_ERROR( cudaMemcpyToSymbol( "input_variables", input, n_variables * sizeof(float), 0, cudaMemcpyHostToDevice ) );
  // HANDLE_ERROR( cudaMemcpyToSymbol( "pdf_input", pdf_in, 44 * n_events * sizeof(float), 0, cudaMemcpyHostToDevice) );
  // HANDLE_ERROR( cudaMemcpyToSymbol( "alphas_input", alphas_in, n_events * sizeof(float), 0, cudaMemcpyHostToDevice ) );
  // HANDLE_ERROR( cudaMemcpyToSymbol( "z_input", z, 2 * n_events * sizeof(float), 0, cudaMemcpyHostToDevice ) );
  

  // // capture the start time
  cudaEvent_t start, stop;
  cudaEventCreate( &start);
  cudaEventCreate( &stop );
  cudaEventRecord( start, 0 );


  //for debug, copy the device memory back to host to see if copy is correct
  //pdf and alphas and z for effparton
  float* pdf_temp = new float[44 * n_events]; //for leading order it is 44 pdf per event.
  float* alphas_temp = new float[n_events];
  float* z_temp = new float[n_events * 2];
  HANDLE_ERROR( cudaMemcpy(pdf_temp, pdf_on_device, 44 * n_events * sizeof(float),  cudaMemcpyDeviceToHost) );
  HANDLE_ERROR( cudaMemcpy(alphas_temp, alphas_input, n_events * sizeof(float),  cudaMemcpyDeviceToHost) );
  HANDLE_ERROR( cudaMemcpy(z_temp, z_input,  n_events * sizeof(float),  cudaMemcpyDeviceToHost) );


  // run!
  GPU_submit<<<number_of_blocks,number_of_threads>>>(event_index_on_device, warehouse_on_device, pdf_on_device);
  
  //get stop time, and display the timing results
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );
  float elapsedTime;
  cudaEventElapsedTime( &elapsedTime, start, stop );
  //std::cout<<"GPU calculation completed================" <<std::endl;
  
  // copy output variables from device to host
  HANDLE_ERROR( cudaMemcpy(warehouse, warehouse_on_device, 17 * 11 * n_events * sizeof(float), cudaMemcpyDeviceToHost) );
  HANDLE_ERROR( cudaMemcpy(event_index, event_index_on_device, n_events * sizeof(float), cudaMemcpyDeviceToHost) );
  
  // HANDLE_ERROR( cudaMemcpyFromSymbol( &eks_input_variables, eks_setting_on_device,  sizeof(eks_input), 0, cudaMemcpyDeviceToHost ) );
  

  // release locked memory 
  cudaFree( event_index_on_device );
  cudaFree( warehouse_on_device );
  cudaFree( &eks_setting_on_device);
  cudaFree( input_variables );
  cudaFree( pdf_on_device );
  cudaFree( alphas_input );
  cudaFree( z_input );

  return elapsedTime;
  //return 0.0;
}


//========================================================================


//========================================================================

eks& eks::operator=(eks& rhs)
{
  return rhs;
}

//========================================================================
