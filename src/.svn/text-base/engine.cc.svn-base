#include <iostream>
#include <stdlib.h>
//--------------------
#include "engine.h"

/* this is the engine for EKS that switches from different monte carlo methods.*/

//========================================================================


void engine::init()
{
  /* 
     this function initial all the parameter needed to set up the monte carlo engine.
   */

  if(input_card.next_to_leading_order) 
    {
      ndim = 7;
    }
  else
    {
      ndim = 4;
    }
  
  //common parameters
  ncomp = 1;
  userdata = NULL;
  epsrel = 1E-5;
  epsabs = 1E-12;
  flags = 2;
  seed = input_card.mc_seed;
  mineval = 0;
  maxeval = input_card.data_point_numbers;
  nstart = maxeval/10.0;
  // nincrease = input_card.data_point_numbers/10;
  nincrease = 0;

  nbatch = input_card.batch; // how many events to submit to the gpu

  gridno = 0;
  statefile = NULL;
  
  //suave parameter
  nnew = 10000;
  flatness = 25.;
  
  //Divonne parameters
  key1 = 47;
  key2 = 1;
  key3 = 1;
  maxpass = 5;
  border = 0.;
  maxchisq = 10.;
  mindeviation = 0.25;
  ngiven = 0;
  ldxgiven = ndim;
  nextra = 0;
  
  //cuhre parameter
  key = 0;


  //other parameter.
  neval = new int(0);
  fail = new int(0);
  nregions = new int(0);

  // GPU computing parameter
  number_of_threads = input_card.number_of_threads;
  number_of_blocks = input_card.number_of_blocks;
  
  //parameter for uniform random number engine;
  
}

//========================================================================

void engine::read_card(card& card_in)
{
  input_card = card_in;
}

//========================================================================

void engine::boot(Controlbox* const box, integrand_t integrand) 
{
  switch(input_card.mc_engine_type)
    {
    case 0:
      std::cout<<"engine is not set up correctly with engine type : "<< input_card.mc_engine_type <<std::endl;
      break;
    case 1:
      Vegas(ndim, ncomp, integrand, (void*)box, epsrel, epsabs, flags, seed, mineval, maxeval, nstart, nincrease, nbatch, gridno, statefile, neval, fail, integral, error, prob);
      break;
    case 2:
      Suave(ndim, ncomp, integrand, (void*)box, epsrel, epsabs, flags, seed, mineval, maxeval, nnew, flatness, nregions, neval, fail, integral, error, prob);
      break;
    case 3:
      Divonne(ndim, ncomp, integrand, (void*)box, epsrel, epsabs, flags, seed, mineval, maxeval, key1,  key2,  key3,  maxpass,  border, maxchisq, mindeviation,  ngiven,  ldxgiven,  xgiven,  nextra, peakfinder, nregions, neval, fail, integral, error, prob);
      break;
    case 4:
      Cuhre(ndim, ncomp, integrand, (void*)box, epsrel, epsabs, flags, mineval, maxeval, key, nregions, neval, fail, integral, error, prob);
      break;
    case 5:
      uniform_engine(box,integrand);
      break;
    default:
      std::cout<<"engine is not set up correctly with engine type : "<< input_card.mc_engine_type <<std::endl;
      break;
    }
}

//========================================================================

engine& engine::operator=(engine& rhs)
{
  return rhs;
}
void engine::uniform_engine(Controlbox* const box, integrand_t integrand)
{
  //the random number array
  double* xx = new double[number_of_blocks*number_of_threads*ndim];
  double* ff = new double[number_of_blocks*number_of_threads*ncomp]; 
  
  /* initialize random seed: */
  srand (seed);
  
  //set the weight for uniform distribution
  double weight = (double)1.0/(double)maxeval;
  
  //always the firt iteration, vegas may have several iterations.
  int iteration = 1;

  for(int i = 0; i < maxeval/number_of_threads; ++i)
    {
      //generate random number
      for( int j = 0; j < number_of_threads*ndim; ++j)
	{
	  xx[j] = (double)rand()/(double)RAND_MAX;
	}
      //calculate the integrand;
      integrand(&ndim, xx, &ncomp, ff, box, &weight, &iteration, number_of_threads);
      if(i % (maxeval/number_of_threads/10) == 0 ) std::cout<<"evaluated "<< i*number_of_threads <<" events so far"<<std::endl;
    }
}
