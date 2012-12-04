/*
  this is pdf, to get the pdfs from lhapdf.
  
 */

#ifndef PDF_H
#define PDF_H

#include <string>
#include <iostream>

//--------------------
//----------------------
#include "card.h"

class pdf {
  
 
 public:
  pdf();   
  virtual ~pdf();

 public:  
  //inittial funcitons
  void read_card(card&);

  void find_pdf(const double* x, float* alphas, float* pdfs, float* z,  int n_events);
  
  // private:
  //-------------------- input card
  card input_card;

  /* 
     Pointers to the ROOT application for writing the ntuple, 
     file with the ntuple, and the ntuple itself
  */
};

#endif

