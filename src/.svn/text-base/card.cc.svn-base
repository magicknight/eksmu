#include "card.h"
#include <iomanip>
#include <iostream>

using namespace std;

//========================================================================
//========================================================================
card::~card()
{
}
//========================================================================



card& card::operator=(card& rhs)
{
  if(this != &rhs){ //make sure not self copy
    /*copy a card to another card*/
    card_name = rhs.card_name;
    square_s = rhs.square_s;
    ppbar_collider = rhs.ppbar_collider ;
    next_to_leading_order = rhs. next_to_leading_order;
    pdf_set_number = rhs.pdf_set_number ;
    renomalization_scale = rhs.renomalization_scale;
    factorization_scale = rhs.factorization_scale;
    mc_engine_type = rhs.mc_engine_type ;
    mc_seed  = rhs.mc_seed ;
    data_point_numbers  = rhs.data_point_numbers ;
    write_to_log_txt  = rhs.write_to_log_txt ;
    log_file_name = rhs.log_file_name;
    write_to_root_file  = rhs.write_to_root_file ;
    root_file_name = rhs.root_file_name;
    pt_min  = rhs.pt_min ;
    pt_max  = rhs.pt_max ;
    y_min  = rhs.y_min ;
    y_max  = rhs.y_max ;
    number_of_blocks = rhs.number_of_blocks;
    number_of_threads = rhs.number_of_threads;
    batch = rhs.batch;
  }
  return *this;
}// end of operator =

//========================================================================

void card::make_card(const char* filename)
{
  /*
    read variables from filename.
   */
  
  fstream fs;
  fs.open(filename);

  skip_comment(fs);
  fs >> square_s >> ppbar_collider;
  
  skip_comment(fs);
  fs >> next_to_leading_order;

  skip_comment(fs);
  fs >> pdf_set_number;
  
  skip_comment(fs);
  fs >> renomalization_scale >> factorization_scale;

  skip_comment(fs);
  fs >> mc_engine_type;
  
  skip_comment(fs);
  fs >> mc_seed;

  skip_comment(fs);
  fs >> data_point_numbers;

  skip_comment(fs);
  fs >> write_to_log_txt >> log_file_name;

  skip_comment(fs);
  fs >> write_to_root_file >> root_file_name;

  skip_comment(fs);
  fs >> y_min >> y_max >> pt_min >> pt_max;

  skip_comment(fs);
  fs >> number_of_blocks >> number_of_threads;

  skip_comment(fs);
  fs >> batch;

  fs.close();

}

//========================================================================


void card::skip_comment(fstream& fs)
{
  /* 
     this function skips the comments line in input file.
   */
  //declare variable
  char head;
  string aline;
  
  fs >> head;

  //skip header
  while(head == '#')
    {
      getline(fs, aline);
      fs >> head;      
    }
  
  //not comment, set the position to the head of line.
  fs.seekg( -1, ios::cur);
  
  return;
}

//========================================================================

ostream& operator<<(ostream& os, const card& cd)
{
  /*
    overloaded operator to print the card
   */

  os<<"===================================================================================================="<<endl;
  os<< cd.card_name <<endl;
  os<<"===================================================================================================="<<endl;
  
  

  // adjust output to the left
  os << left;
 
  // print table header
  os << setw(5) << "#" << setw(25) << "Name" << setw(30) << "Value" << setw(100) << "Comment" << endl << endl;
  // print data
  os << setw(5) << 1 << setw(25) << "square_s"   << setw(30) <<  cd.square_s      << setw(100) <<   "square s:"     << endl;
  os << setw(5) << 2 << setw(25) << "ppbar_collider"   << setw(30) <<  cd.ppbar_collider      << setw(100) <<   "PPBAR true/false (0:p-pbar collider ; 1:pp collider)"     << endl;
  os << setw(5) << 3 << setw(25) << "next_to_leading_order"   << setw(30) <<  cd.next_to_leading_order      << setw(100) <<   "Order of the calculation: 0 for born, 1 for NLO"     << endl;
  os << setw(5) << 4 << setw(25) << "pdf_set_number"   << setw(30) <<  cd.pdf_set_number      << setw(100) <<   "Iset of PDF "     << endl;
  os << setw(5) << 5 << setw(25) << "renomalization_scale"   << setw(30) <<  cd.renomalization_scale      << setw(100) <<   "scale choices: mu_uv / PJ and mu_co / PJ"     << endl;
  os << setw(5) << 6 << setw(25) << "factorization_scale"   << setw(30) <<  cd.factorization_scale      << setw(100) <<   "...."     << endl;
  os << setw(5) << 7 << setw(25) << "mc_engine_type"   << setw(30) <<  cd.mc_engine_type      << setw(100) <<   "MC generator := foam = 0, vegas = 1,  suave = 2"     << endl;
  os << setw(5) << 8 << setw(25) << "mc_seed"   << setw(30) <<  cd.mc_seed      << setw(100) <<   "MC SEED"     << endl;
  os << setw(5) << 9 << setw(25) << "data_point_numbers"   << setw(30) <<  cd.data_point_numbers      << setw(100) <<   "number of RENO points"     << endl;
  os << setw(5) << 10 << setw(25) << "write_to_log_txt"   << setw(30) <<  cd.write_to_log_txt      << setw(100) <<   "Write to a log file   (1:yes, 0: no)"     << endl;
  os << setw(5) << 11 << setw(25) << "log_file_name"   << setw(30) <<  cd.log_file_name      << setw(100) <<   "the log file name"     << endl;
  os << setw(5) << 12 << setw(25) << "write_to_root_file"   << setw(30) <<  cd.write_to_root_file      << setw(100) <<   "Write to a ROOT file (1:yes, 0: no)"     << endl;
  os << setw(5) << 13 << setw(25) << "root_file_name"   << setw(30) <<  cd.root_file_name      << setw(100) <<   "the ROOT file name"     << endl;
  os << setw(5) << 14 << setw(25) << "pt_min"   << setw(30) <<  cd.pt_min      << setw(100) <<   "Cuts on each jet: ymin, ymax, ptmin, ptmax"     << endl;
  os << setw(5) << 15 << setw(25) << "pt_max"   << setw(30) <<  cd.pt_max      << setw(100) <<   "...."     << endl;
  os << setw(5) << 16 << setw(25) << "y_min"   << setw(30) <<  cd.y_min      << setw(100) <<   "...."     << endl;
  os << setw(5) << 17 << setw(25) << "y_max"   << setw(30) <<  cd.y_max      << setw(100) <<   "...."     << endl;

 
  return os;
}

//========================================================================
