#ifndef RECORDER_H
#define RECORDER_H

#include "TApplication.h"
#include "TFile.h"
#include "TTree.h"
//--------------------------------------------
#include <iostream>
#include <fstream>
//--------------------------------------------
#include "card.h"
#include "container.h"

class recorder
{
 public:
  recorder();
  virtual ~recorder();
  recorder& operator=(recorder& );

 public:
  void record(int*, float*, double*, int*);
  void read_card(card&);
  void init();
  void init_root();
  void init_log();
  void fill_header();

  void printnt();

 public:
  card input_card;

  fstream fs;
  TFile *outfile;   
  TTree *nt, *header;
  
  Float_t p1;
  Float_t p2;
  Float_t p3;

  Float_t y1;
  Float_t y2;
  Float_t y3;

  Float_t phi1;
  Float_t phi2;
  Float_t phi3;

  Float_t weight;

  
  Int_t nparton;

  Int_t iteration;

  // number of blocks times the number of  events is the nubmer of phase space points
  Int_t number_of_points;
  
};
#endif
