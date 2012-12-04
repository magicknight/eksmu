#ifndef ENGINE_H
#define ENGINE_H

#include "cuba.h"
#include "card.h"

class Controlbox;

class engine
{
 public:
  void read_card(card&);
  void init();
  void boot(Controlbox* const , integrand_t);
  
  void uniform_engine(Controlbox* const,integrand_t);

  engine& operator=(engine&);

 public:
  card input_card;

  //vegas parameter
  int ndim;
  int ncomp;
  //integrand_t integrand; 
  void *userdata;
  double epsrel; 
  double epsabs;
  int flags; 
  int seed;
  int mineval; 
  int maxeval;
  int nstart; 
  int nincrease; 
  int nbatch;
  int gridno; 
  int *neval;
  int *fail;
  double integral[]; 
  double error[]; 
  double prob[];
  char *statefile;

  // suave parameter
  int nnew;
  double flatness;
  int *nregions;
  
  //Divonne parameters;
  int key1; 
  int key2; 
  int key3; 
  int maxpass; 
  double border; 
  double maxchisq; 
  double mindeviation; 
  int ngiven; 
  int ldxgiven; 
  double xgiven[]; 
  int nextra; 
  peakfinder_t peakfinder;

  //Cuhre parameter
  int key;

  // GPU computing parameter
  int number_of_threads;
  int number_of_blocks;
};

#endif
