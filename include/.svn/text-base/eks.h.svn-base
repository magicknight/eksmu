#ifndef EKS_H
#define EKS_H

#include "card.h"
#include "container.h"
#include "jetsub5b.h"



// common block to retrive events from eks.
typedef struct{
  double p1[17];
  double p2[17];
  double p3[17];

  double y1[17];
  double y2[17];
  double y3[17];

  double phi1[17];
  double phi2[17];
  double phi3[17];

  double weight[17]; //weights of each event.
  int nparton[17];
  int nevent; //number of events;
} WAREHOUSE;
extern WAREHOUSE warehouse_;

class eks{
 public:
  eks();
  virtual ~eks();
  eks& operator=(eks&);
  
  
 public:
  void read_card(card&);
  float integrand(int*, const double*, float*, float* alphas, float* pdf, float* z,  int);
  eks_input eks_input_variables;

 private:
  // GPU computing parameters
  int number_of_blocks;
  int number_of_threads;
  
};
#endif

