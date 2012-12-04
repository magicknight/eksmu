#ifndef CARD_H
#define CARD_H

#include <string>
#include <iostream>
#include <fstream>


/*
  this is the card for input or output
 */

class card {
  friend std::ostream& operator<<(std::ostream&, const card&);

 public:
  
  card& operator=(card &); 

  void  make_card(const char*);
  void skip_comment(std::fstream&);
  
  //varables 
  std::string card_name;
  double square_s;
  bool ppbar_collider;
  bool next_to_leading_order;
  int pdf_set_number;
  double renomalization_scale, factorization_scale;
  int mc_engine_type;
  int mc_seed;
  int data_point_numbers;
  bool write_to_log_txt;
  std::string log_file_name;
  bool write_to_root_file;
  std::string root_file_name;
  double pt_min, pt_max, y_min, y_max;
  int number_of_blocks, number_of_threads;
  int batch; //nubmer of events go to the GPU

 card(const std::string &name = "",
       double RootS = 0.,
       bool ppbar = false,
       bool NLO = false,
       int pdf = 0,
       double rscale = 0.,
       double fscale = 0.,
       int mctype = 0,
       int seed = 0,
       int events = 0,
       bool log = false,
      std::string logfile = "default.log",
       bool root = false,
      std::string rootfile = "default.root",
       double ptmin = 0.,
       double ptmax = 0.,
       double ymin = 0.,
      double ymax = 0.,
      int n_of_threads = 1000,
      int n_of_blocks = 1,
      int n_batch = 1000
      ):card_name(name),
    square_s(RootS),
    ppbar_collider(ppbar),
    next_to_leading_order(NLO),
    pdf_set_number(pdf),
    renomalization_scale(rscale),
    factorization_scale(fscale),
    mc_engine_type(mctype),
    mc_seed(seed),
    data_point_numbers(events),
    write_to_log_txt(log),
    log_file_name(logfile),
    write_to_root_file(root),
    root_file_name(rootfile),
    pt_min(ptmin),
    pt_max(ptmax),
    y_min(ymin),
    y_max(ymax),
    number_of_blocks(n_of_blocks),
    number_of_threads(n_of_threads),
    batch(n_batch)
    {};
  
  
  virtual ~card();
  

};

#endif
