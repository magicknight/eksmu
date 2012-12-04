/*
  the first line of eksmu

 */
#include <iostream>
#include <time.h>
#include <iomanip>

//--------------------------------------------
#include "card.h"
#include "eks.h"
#include "engine.h"
#include "container.h"
#include "recorder.h"
#include "controlbox.h"


using namespace std;

//============================================
int main(){
    
  card input_card("EKS input card");
  input_card.make_card("eks.in");
  

  Controlbox box;
  box.insert_card(input_card);
  box.set_engine();
  box.set_eks();
  box.set_recorder();
  box.set_pdf();
  box.read_bins();

  cout<<input_card;

  
  //run the calculation
  box.run();
  
  //print the cross section
  box.print_xsec();
  
  //print the cpu time
  cout<<fixed<<setprecision(2)<< box.compute_time/1000.0 << " second seconds spent on GPU calculation." << endl;

  return 0;
}
