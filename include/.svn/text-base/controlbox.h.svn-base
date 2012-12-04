/*
  this is control box, I got random numbers from monte carlo integrator, feed them to reno, read out the events from reno, then write them to a root file, final, I feed the event weights to vegas.
  
 */

#ifndef CONTROLBOX_H
#define CONTROLBOX_H

#include <string>
#include <iostream>
#include <fstream>

//--------------------
//----------------------
#include "card.h"
#include "recorder.h"
#include "engine.h"
#include "eks.h"
#include "container.h"
#include "pdf.h"

class Controlbox {
  
  friend int bearing();

 public:
  Controlbox();   
  virtual ~Controlbox();

 public:  
  //inittial funcitons
  void insert_card(card&);
  void set_recorder();
  void set_engine();
  void set_eks();
  void set_pdf();
  void read_bins();

  void printnt();
  void print_xsec();
  void run();  
  void find_total_Xsec(const int* ncomp, double ff[], float* warehouse, int* event_index, double* weight, int n_events);
  void integral(const int* ndim, const double xx[], const int *ncomp, double ff[], double* weight, int* iter);
  // private:
  //-------------------- input card, event recorder, and mc engine
  card input_card;
  recorder event_recorder;
  engine mc_engine;
  eks eks5b;
  container ct;
  pdf lhapdf;

  // the integrand time
  float compute_time;
  /* 
     Pointers to the ROOT application for writing the ntuple, 
     file with the ntuple, and the ntuple itself
  */

  // read in the bins settting
  
  fstream fs;
  std::string aline;

  int n_ybin;
  float* ybins;
  float* n_ptbin;
  float** ptbins;

  float** errorList;
  float** xsecList;

};

#endif

