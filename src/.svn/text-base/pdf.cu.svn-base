#include "pdf.h"
#include <math.h>
#include "LHAPDF/LHAPDF.h"

using namespace std;

//========================================================================

pdf::pdf()
{

}

//========================================================================

void pdf::read_card(card& card_in)
{
  input_card = card_in;
}

//========================================================================

pdf::~pdf()
{

}

//========================================================================
/* 

   find the pdfs for the input variables
   this is only for leading order.
   
*/
void pdf::find_pdf(const double* x, float* alphas, float* pdfs, float* z, int n_events)
{
  double  x1, x2, x3, x4, x5, x6, x7;
  float y1, p2, y2, phi2, p1;
  float xa, xb;
  float ey1,ey2;
  float rts = input_card.square_s;
  int nfl = 5;
  float yscale = 1.2;
  float pscale = 0.06 * input_card.square_s;
  float tiny = 1.0e-10;
  float pi = 3.141592654;
  float muuv, muco;

  //init pdf set
  LHAPDF::initPDFSet(input_card.pdf_set_number, 1);
  
  //init pdf subset
  //LHAPDF::initPDF(0);
  
   if( !input_card.next_to_leading_order )
     { //if leading order

       
       //calculation of the kinematics
       for(int i = 0; i < n_events; ++i)
	 {
	   x1 = x[i];
	   x2 = x[i+1];
	   x3 = x[i+2];
	   x4 = x[i+3];
      
	   // common kinematic variables
	   y1 = yscale * log( (x1 + tiny) / (1.0 + tiny - x1) );
      
	   p2 = pscale * x2 / (1.0 - x2 + 2 * pscale/rts);
      
	   y2 = yscale * log( (x3 + tiny) / (1.0 + tiny - x3) );
  
	   phi2 = 2.0*pi*x4 - pi;

	   // leading order
	   p1 = p2;
      
	   ey1 = exp(y1);
	   ey2 = exp(y2);
      
	   xa =  (ey1+ey2) * p2/rts;
	   xb = (1.0/ey1 + 1.0/ey2) * p2/rts;

	   muuv = input_card.renomalization_scale * p2;
	   muco = input_card.factorization_scale * p2;
	   
	   //find z for effparton
	   float za = xa + (1.0 - xa) * rand() / double(RAND_MAX);
	   float zb = xb + (1.0 - xb) * rand() / double(RAND_MAX);
	   
	   z[2*i] = za;
	   z[2*i + 1] = zb;
	   
	   //get the pdfs
	   vector<double> result = LHAPDF::xfx(xa, muco);
	   for(int j = 1; j < 12; ++j){
	     pdfs[i*44 + j-1]  = result.at(j)/xa;
	   }
	   pdfs[i*44 + 3] = result.at(5);
	   pdfs[i*44 + 4] = result.at(4);	   
	   pdfs[i*44 + 6] = result.at(8);
	   pdfs[i*44 + 7] = result.at(7);
	   

	   result = LHAPDF::xfx(xb, muco);
	   for(int j = 1; j < 12; ++j){
	     pdfs[i*44 + 11 + j]  = result.at(j)/xb;
	   }
	   pdfs[i*44 + 11 + 3] = result.at(5);
	   pdfs[i*44 + 11 + 4] = result.at(4);	   
	   pdfs[i*44 + 11 + 6] = result.at(8);
	   pdfs[i*44 + 11 + 7] = result.at(7);

	   result = LHAPDF::xfx(xa/za, muco);
	   for(int j = 1; j < 12; ++j){
	     pdfs[i*44 + 22 + j]  = result.at(j)/xa*za;
	   }
	   pdfs[i*44 + 22 + 3] = result.at(5);
	   pdfs[i*44 + 22 + 4] = result.at(4);	   
	   pdfs[i*44 + 22 + 6] = result.at(8);
	   pdfs[i*44 + 22 + 7] = result.at(7);
	   
	   result = LHAPDF::xfx(xb/zb, muco);
	   for(int j = 1; j < 12; ++j){
	     pdfs[i*44 + 33 + j]  = result.at(j)/xb*zb;
	   }
	   pdfs[i*44 + 33 + 3] = result.at(5);
	   pdfs[i*44 + 33 + 4] = result.at(4);	   
	   pdfs[i*44 + 33 + 6] = result.at(8);
	   pdfs[i*44 + 33 + 7] = result.at(7);	   
	   
	   
	   // for( int aa = -nfl; aa < nfl; ++aa)
	   //   {
	   //     pdfs[i*44 + aa + 5] = LHAPDF::xfx(xa, muco)/xa;
	   //     pdfs[i*44 + aa + 5 + 11] = LHAPDF::xfx(xb, muco)/xb;
	   //     pdfs[i*44 + aa + 5 + 22] = LHAPDF::xfx(xa/za, muco)/xa*za;
	   //     pdfs[i*44 + aa + 5 + 33] = LHAPDF::xfx(xb/zb, muco)/xb*zb;

	   //   }// for aa
	   
	   // switch the up and down quark since cteq use different index from lhapdf
	   // float temp = pdfs[i*44 + 8];
	   // pdfs[i*44  + 8] = pdfs[i*44  + 9];
	   // pdfs[i*44  + 9] = temp;
	   
	   // temp = pdfs[i*44  + 11];
	   // pdfs[i*44  + 12] = pdfs[i*44  + 11];
	   // pdfs[i*44  + 11] = temp;
	   
	   // // switch for xb
	   // temp = pdfs[i*44  + 19];
	   // pdfs[i*44  + 19] = pdfs[i*44  + 20];
	   // pdfs[i*44  + 19] = temp;
	   
	   // temp = pdfs[i*44  + 22];
	   // pdfs[i*44  + 23] = pdfs[i*44  + 22];
	   // pdfs[i*44  + 22] = temp;

	   // // switch for xa/za
	   // temp = pdfs[i*44  + 30];
	   // pdfs[i*44  + 30] = pdfs[i*44  + 31];
	   // pdfs[i*44  + 30] = temp;
	   
	   // temp = pdfs[i*44  + 33];
	   // pdfs[i*44  + 34] = pdfs[i*44  + 33];
	   // pdfs[i*44  + 33] = temp;


	   // // switch for xb/zb
	   // temp = pdfs[i*44  + 41];
	   // pdfs[i*44  + 41] = pdfs[i*44  + 42];
	   // pdfs[i*44  + 41] = temp;
	   
	   // temp = pdfs[i*44  + 44];
	   // pdfs[i*44  + 45] = pdfs[i*44  + 44];
	   // pdfs[i*44  + 44] = temp;


	   //fill the alphas
	   alphas[i] = LHAPDF::alphasPDF(muuv); 
	      
	 } // for i < neverts

     } // if leading order
      

}

//========================================================================
