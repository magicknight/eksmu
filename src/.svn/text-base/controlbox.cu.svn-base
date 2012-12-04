#include "controlbox.h"
#include <math.h>
#include <iomanip>
  
using namespace std;
/*
  this is the bearing function called by vegas
 */

//========================================================================

Controlbox::Controlbox()
{
  // read in the bins settting
  const int max_ptbin = 100;
  const int max_ybin = 10;
  
  ybins = new float[max_ybin];
  n_ptbin = new float[max_ybin];
  
  ptbins = new float*[max_ybin];
  errorList= new float*[max_ybin];
  xsecList = new float*[max_ybin];

  for(int i = 0; i < max_ybin; ++i)
    {
      ptbins[i] = new float[max_ptbin];
      errorList[i] = new float[max_ptbin];
      xsecList[i] = new float[max_ptbin];
    }
  
}

//========================================================================

Controlbox::~Controlbox()
{
  const int max_ybin = 10;
  
  delete[] ybins;
  delete[] n_ptbin;
  
  for(int i = 0; i < max_ybin; ++i)
    {
      delete[] ptbins[i];
      delete[] errorList[i];
      delete[] xsecList[i];
    }
  delete[] ptbins;
  delete[] errorList;
  delete[] xsecList;
}

//========================================================================




void Controlbox::insert_card(card& card_in)
{
  input_card = card_in;
  
}

//========================================================================

void Controlbox::set_recorder()
{
  event_recorder.read_card(input_card);
  event_recorder.init();
}

//========================================================================

void Controlbox::set_engine()
{
  mc_engine.read_card(input_card);
  mc_engine.init();;

}
//========================================================================

void Controlbox::set_eks()
{
  eks5b.read_card(input_card);
}

//========================================================================

void Controlbox::set_pdf()
{
  lhapdf.read_card(input_card);
}

//========================================================================

static int bearing (const int* ndim, const double xx[], const int *ncomp, double ff[], void *box, double* weight, int* iter, int n_events) 
{  
  /*
    this function get random number from monte carlo engine then run other parts of the controlbox.
    and it is for Cuba library
  */
  // set up the container for output variables
  // the first element is the number of total events, 
  int number_of_variables =  n_events * (17*11);  

  float* warehouse_of_output_variables = new float[number_of_variables];  // total number of varialbes should be (number of threads)*( 17(number of events per phase space) * (variable per event = 11 ))

  int* event_index = new int[ n_events ];
  
  //pdf and alphas and z for effparton
  float* pdf = new float[44 * n_events]; //for leading order it is 44 pdf per event.
  float* alphas = new float[n_events];
  float* z = new float[n_events * 2];

  //get the pdfs.
  ((Controlbox*)box)->lhapdf.find_pdf(xx,  alphas, pdf, z, n_events);
 
  // eks program
  ((Controlbox*)box)->compute_time += ((Controlbox*)box)->eks5b.integrand(event_index,xx,warehouse_of_output_variables, alphas, pdf, z, n_events);
  
  // record the data of ct to root or log (if write to root is set to true)
  //zliang 2012: we dont record for now
  //((Controlbox*)box)->event_recorder.record(event_index, warehouse_of_output_variables, weight, iter);


 //return a number to monte carlo engine.
  ((Controlbox*)box)->find_total_Xsec(ncomp, ff, warehouse_of_output_variables, event_index, weight, n_events);
  
  delete[] warehouse_of_output_variables;
  delete[] event_index;
  delete[] pdf;
  delete[] alphas;
  
  return 0;
}
//========================================================================

void Controlbox::read_bins()
{

  fs.open("bins.in");
  
  //read header
  getline(fs,aline);
  getline(fs,aline);
  getline(fs,aline);

  // read ybins

  fs >> n_ybin;
 
  for(int i = 0; i < n_ybin+1; ++i){
    fs>> ybins[i];
  }
  
  
  //read pt bins
  for(int i = 0; i < n_ybin; ++i){
    fs>>n_ptbin[i];
    for(int j = 0; j < n_ptbin[i]+1; ++j){
      fs>> ptbins[i][j];
    }
  }

  //init xsect and error
  for(int i = 0; i < n_ybin; ++i){
    for(int j = 0; j < n_ptbin[i]; ++j){
      xsecList[i][j] = 0.0;
      errorList[i][j] = 0.0;
    }
  }//i

  fs.close();
  
}
//========================================================================

void Controlbox::find_total_Xsec(const int* ncomp, double ff[], float* warehouse, int* event_index, double* weight, int n_events)
{
  /*
    if the cross section is non zero, we return 1 to monte carlo engine, else we return zero;
  */
  //  cout << " printing out cross sections ......................" << endl;




  for(int i = 0; i < n_events; ++i)
    {
      for(int j = 0; j < event_index[i]; ++j)
	{
	  int event_position = i*17*11 + j*11;
	  int npartons = warehouse[event_position];
	  float p1 = warehouse[event_position + 1];
	  float y1 = warehouse[event_position + 2];
	  float phi1 = warehouse[event_position + 3];
	  float  p2 = warehouse[event_position + 4];
	  float  y2 =  warehouse[event_position + 5];
	  float  phi2 =  warehouse[event_position + 6];
	  float p3 =   warehouse[event_position + 7];
	  float y3 =  warehouse[event_position + 8];
	  float phi3 =  warehouse[event_position + 9];
	  float integral =  warehouse[event_position + 10] * *weight;

	  ff[i] = integral;
	  for(int k = 0; k < n_ybin; k++)
	    {
	      for(int l = 0; l < n_ptbin[k];  l++)
		{
		  // if in the bin?
		  if( y1 > ybins[k] && y1 < ybins[k+1] )
		    {
		      if( p1 > ptbins[k][l] && p1 < ptbins[k][l+1] )
			{
			  xsecList[k][l] += integral;
			} // pt bin
		    } // ybin

		  if( y2 > ybins[k] && y2 < ybins[k+1] )
		    {
		      if( p2 > ptbins[k][l] && p2 < ptbins[k][l+1] )
			{
			  xsecList[k][l] += integral;
			} // pt bin
		    } // ybin
		} // for go through pt bins
	    } // go through y bins;
	} // go through channels

    } // for events
  
}

//========================================================================

void Controlbox::print_xsec()
{
  //print output
  cout<<endl<<endl;
  cout<<"pt min"<<"\t\t"<<"pt max"<<"\t\t"<< "Xsec/dp/dy"<<"\t\t"<<"Error"<<endl;

  for(int k = 0; k < n_ybin; ++k){
    cout.unsetf ( ios_base::floatfield  ); 
    cout<<"--------------------------------------------"<<endl;

    cout<<ybins[k]<<" < y < " <<ybins[k+1]<<endl;

    cout<<"--------------------------------------------"<<endl;

    for(int l = 0; l < n_ptbin[k]; ++l){
      cout<<fixed<<setprecision(2)<<ptbins[k][l]<<"\t\t"<<ptbins[k][l+1]<<"\t\t"<<scientific<<setprecision(6)<< xsecList[k][l]<<"\t\t"<< errorList[k][l]<<endl;
    }
  }
  cout<<endl<<endl;
}
//========================================================================

void Controlbox::run()
{
  mc_engine.boot( this, bearing);
  
  //fill the root file header.
  if(input_card.write_to_root_file) 
    {
      event_recorder.fill_header();
    }
}

//========================================================================
