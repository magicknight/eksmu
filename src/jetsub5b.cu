#include "../include/jetsub5b.h"
#include <math.h>
/*
  a C version of the jet cross section calculation program EKS.

  C version is written by Zhihua Liang @ SMU,
  beginning at Feb 8th 2012
 */

//below is the header of the fortran version.

/************************************************************************************
C23456789012345678901234567890123456789012345678901234567890123456789012
C***********************************************************************
C***********************************************************************
C                         30 June 2010
C***********************************************************************
C
C                            JETSUBS
C                  Version 3.0,  23 February 1991
C                  Version 3.01, 14 March 1991:
C                     - misc small changes to enhance portability
C                  Version 3.1,  26 February 1992:
C                     - changes in soft integrals to match paper
C                     - a few other changes to match paper and simplify
C                     - common block /ARRAYS/ enlarged
C                     - switch for proton-proton collisions
C                  Version 3.4, 13 June 1996
C                     - only the name is changed, to indicate by
C                       the name that this is the version to use
C                       with jet3_4.f
C                     - name change is incorportated in PRINTVERSION;
C                       version date left unchanged.
C                   Version 3.4.01, 2 September 1997
C                     - check for divide by zero incorporated in
C                       variable SINGULARITY.
C                   Version 3.41, 1 March 2008
C                     - some minor changes in syntax to keep
C                       compilers happy
C                     - array sizes fixed by parameter statements and
C                       minor bug in array initialization fixed
C                   Version 3.41, 1 March 2008
C                     - some minor changes in syntax to keep
C                       the gfortran compiler happy
C                     - array sizes fixed by parameter statements and
C                       minor bug in array initialization fixed
C                   Version 3.42, September 24, 2009
C                     - gamma changed to gammaval; 
C                     - two continue statements added.
C                   Version 5x, March 2010
C                     - new version to act as a simple event generator
C
C                   version 5b, modified @ Dec 2011 by Zhihua Liang
C
C***********************************************************************
C
C   Subroutines for JET.  Those that one is likely to want to modify,
C   which are consequently kept with the main program, are designated
C   with a (*).
C
C     SUBROUTINE RENO(NRENO)
C
C      SUBROUTINE INTEGRATE2TO2(Y1,Y2,PHI2,P2,JACOBIAN)
C        Virtual:
C         FUNCTION PSITILDE4(PROCESS,K)
C         FUNCTION PSITILDE6NS(PROCESS,K)
C        Collinear:
C         FUNCTION EFFPARTON(A,X,SCALE)
C            FUNCTION ALTARELLI(AOUT,AIN,Z)
C            FUNCTION ALTRLIPRIME(AOUT,AIN,Z)
C         FUNCTION GAMMAVAL(NPARTON)
C         FUNCTION GAMMAPRIME(NPARTON)
C        Soft:
C         FUNCTION PSICTILDE(PROCESS,I1,I2,K)
C         FUNCTION LI2(X)
C
C      SUBROUTINE INTEGRATEA(Y1,P2,Y2,PHI2,XI,W,PHI3,COEF)
C         FUNCTION FA(Y1,P2,Y2,PHI2,XI,W,PHI3,MUUV,MUCO)
C      SUBROUTINE INTEGRATEB(Y1,P2,Y2,PHI2,XI,W,PHI3,COEF)
C         FUNCTION FB(Y1,P2,Y2,PHI2,XI,W,PHI3,MUUV,MUCO)
C      SUBROUTINE INTEGRATE1(Y2,P1,Y1,PHI1,P3,Y3,PHI3,COEF)
C         FUNCTION F1(Y2,P1,Y1,PHI1,P3,Y3,PHI3,MUUV,MUCO)
C      SUBROUTINE INTEGRATE2(Y1,P2,Y2,PHI2,P3,Y3,PHI3,COEF)
C         FUNCTION F2(Y1,P2,Y2,PHI2,P3,Y3,PHI3,MUUV,MUCO)
C
C Used in INTEGRATE2TO2,INTEGRATEA,INTEGRATEB,INTEGRATE1,INTEGRATE2.
C           SUBROUTINE JETDEFBORN(Y1,P2,Y2,PHI2,
C     >                       OK,PJ,YJ,UNUSED,SVALUE) (*)
C           SUBROUTINE JETDEF(CASE,Y1,P2,Y2,PHI2,P3,Y3,PHI3,
C     >                       OK,PJ,YJ,UNUSED,SVALUE)  (*)
C           SUBROUTINE SETMU(PJ,YJ,UNUSED,MUUV,MUCO) (*)
C           SUBROUTINE BINIT(PJ,YJ,UNUSED,INTEGRAND) (*)
C              FUNCTION TRIAL(PJ,YJ) (*)
C   The JETDEF subroutines calculate the 'measured' variables, like
C   PJ,YJ, given the parton momenta.  Then SETMU uses the measured
C   variables to set the mu values.  At this point the physics functions
C   can be calculated.  Finally, the results are smeared a bit over 
C   the physics variables and stored by BINIT.
C
C Used in FA,FB,F1,F2:
C      FUNCTION CROSSINGSIGN(PROCESS,PERMINV)
C      FUNCTION CONVERT(PHI)
C      FUNCTION RESIDUE(PROCESS,J,N,M,KIN)
C        SUBROUTINE RESIDUEINIT
C           LOGICAL FUNCTION MATCH(J,N,M,JJ,NN,MM)
C        SUBROUTINE PERMUTE
C      SUBROUTINE PARTONSIN(LMNSTY,XA,XB,MUCO)
C      FUNCTION LUMINOSITY(LMNSTY,PROCESS,PERMINV)
C
C Auxiliary functions:
C      FUNCTION PARTON(NPARTON,X,SCALE)  Parton distributions (*)
C      FUNCTION ALPHAS(Q)  Alpha-s
C      FUNCTION WNUM(NPARTON)   Spin and color weight
C      FUNCTION COLOR(NPARTON)  Color charge
C      FUNCTION THETA(BOOL)  Theta function
C      FUNCTION FEXP(Y)  Exponential that doesn't crash for large y
C      SUBROUTINE PRINTVERSION
C
C Random numbers:
C      FUNCTION RANDOM(DUMMY)
C        SUBROUTINE RANDOMINIT(IRAN)
C        SUBROUTINE NEWRAN
C
C***********************************************************************
C***********************************************************************
C***********************************************************************
C***********************************************************************
C
C
************************************************************************************/
/* a test program to test the gpu performance */
__global__ void test_GPU(int* event_index_on_device, float* warehouse_on_device, float* pdf_on_device )
{


}
/******************************************************************************/


/* 
   main program for running reno on graphic device.
*/
__global__ void GPU_submit(int* event_index_on_device, float* warehouse_on_device, float* pdf_on_device )
{
  // initialize the event id 
  initialize(event_index_on_device, warehouse_on_device);
  
  // run calculation function
  reno(event_index_on_device, warehouse_on_device,  pdf_on_device);

  // finish
  finish();
}

/******************************************************************************/
/******************************************************************************/

__device__ void reno( int* event_index_on_device, float* warehouse_on_device, float* pdf_on_device )
{
  /* the main sub program to calculate jet cross section */
  //declare variables.
  //   (Azimuthal angles are defined to lie between - pi and pi)
  int ndim;
  float x1, x2, x3, x4, x5, x6, x7;  
  float rts;
  float yscale, pscale;
  float pjmin, pjmax, yjmax;
  bool s2to2;
  bool safinite, sbfinite, s1finite, s2finite;
  float tiny;
  float y1,y2,y3,phi1,phi2,phi3,p1,p2,p3,w,xi;
  float p1x,p1y,p2x,p2y;
  float cos0,phi0,sin0,x0,y0,z0;
  float dy1dx1,dp2dx2,dy2dx3,dphi2dx4,dp3dx5;
  float dy2dx1,dp1dx2,dy1dx3,dphi1dx4;
  float dxidx5,dwdx6,dphi3dx7;
  float domegadx67,dy3phi3domega;

  float jacobian, singularity, coef;
  
  int process, nperm, i, j;
  float pi = 3.141592654f;
  
  // Initialization from input variables.
  ndim = eks_setting_on_device.ndim;
  rts = eks_setting_on_device.rts;
  yscale = eks_setting_on_device.yscale;
  pscale = eks_setting_on_device.pscale;
  pjmin = eks_setting_on_device.pjmin;
  pjmax = eks_setting_on_device.pjmax;
  yjmax = eks_setting_on_device.yjmax;
  s2to2 = eks_setting_on_device.s2to2;
  safinite = eks_setting_on_device.safinite;
  sbfinite = eks_setting_on_device.sbfinite;
  s1finite = eks_setting_on_device.s1finite;
  s2finite = eks_setting_on_device.s2finite;
  
  
  //   Permutations for each process.  The separate definitions for
  //   each process, with an equivalence statement, are made in order
  //   to keep under 20 continuation lines, for reasons of program
  //   probability.

  // -- SET UP SUM OVER PERMUTATIONS FOR EACH PROCESS IN ORDER:
  //
  int perms[4][5][30] = 
    {
      // -- FIRST PROCESS A:  USE (1,3) <-> (2,4) AND
      // -- ALSO USE C INVARIANCE (1,2) <-> (3,4)
      // -- TO MAKE PI(1)<PI(2),PI(1)<PI(3),PI(1)<PI(4)
      1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,
      2,2,2,2,2,2,3,3,4,4,5,5,3,3,4,4,5,5,3,3,4,4,5,5,3,3,4,4,5,5,
      3,3,4,4,5,5,2,2,2,2,2,2,4,5,3,5,3,4,4,5,3,5,3,4,4,5,3,5,3,4,
      4,5,3,5,3,4,4,5,3,5,3,4,2,2,2,2,2,2,5,4,5,3,4,3,5,4,5,3,4,3,
      5,4,5,3,4,3,5,4,5,3,4,3,5,4,5,3,4,3,2,2,2,2,2,2,1,1,1,1,1,1,

      // -- FOR PROCESS B LET PI(3) < PI(4) ALSO USING EXTRA SYMMETRY
      1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      2,2,2,3,3,4,4,5,5,3,4,5,3,4,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      3,3,4,2,2,2,2,2,2,4,3,3,4,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      4,5,5,4,5,3,5,3,4,5,5,4,5,5,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      5,4,3,5,4,5,3,4,3,2,2,2,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,

      // -- FOR PROCESS C USE 345 SYMMETRY TO MAKE PI(3) < PI(4) < PI(5)
      //    AND C INVARIANCE TO MAKE PI(1)<PI(2)
      1,1,1,1,2,2,2,3,3,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      2,3,4,5,3,4,5,4,5,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      3,2,2,2,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      4,4,3,3,4,3,3,2,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      5,5,5,4,5,5,4,5,4,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,

      //-- FOR PROCESS D SYMMETRIC IN EVERYTHING
      1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
      };
  
  //  Initializations.
  int permarr[5][30][4],perminvarr[5][30][4];
  int nperms[4] = {30,15,10,1};
  
  // Initialize logic arrays for RESIDUE
  residueinit();

  // Initialize permutation arrays
  for( process = 0; process < 4; ++process )
    {
      for(  nperm = 0;  nperm <   nperms[process]; ++nperm)
	{
	  for( i = 0; i < 5; ++i)
	    {
	      j = perms[process][i][nperm];
	      perminvarr[process][nperm][j] = i;
	      permarr[process][nperm][i] = j;
	    }
	}
    }

  //  Set parameters for choosing kinematic variables
  tiny = 1.0e-10f;
 
  //   Calculation starts here.
  // get the random numbers.
  int input_position = blockIdx.x*blockDim.x+threadIdx.x*ndim;
  x1 = input_variables[input_position];
  x2 = input_variables[input_position+1];
  x3 = input_variables[input_position+2];
  x4 = input_variables[input_position+3];
  // if it is next to leading order
  if(ndim > 6 )
    {
      x5 = input_variables[input_position+4];
      x6 = input_variables[input_position+5];
      x7 = input_variables[input_position+6];
    }

  // common kinematic variables
  y1 = yscale * log( (x1 + tiny) / (1.0f + tiny - x1) );
  dy1dx1 = yscale * ( 1.0f + 2.0f*tiny ) / (x1 + tiny) * (1.0f + tiny -x1);
      
  p2 = pscale * x2 / (1.0f- x2 + 2 * pscale/rts);
  dp2dx2 = pscale * (1.0f + 2.0f * pscale/rts ) / pow( (1.0f - x2 + 2.0f*pscale/rts), 2);

  y2 = yscale * log( (x3 + tiny) / (1.0f + tiny - x3) );
  dy2dx3 = yscale * ( 1.0f + 2.0f*tiny ) / (x3 + tiny) * (1.0f + tiny -x3);
      
  phi2 = 2.0f*pi*x4 - pi;
  dphi2dx4 = 2.0f * pi;

  
  // ----------------------- Do 2-to-2 terms ----------------------------
  if( s2to2 )
    {
      
      jacobian = dy1dx1 * dp2dx2 * dy2dx3 * dphi2dx4;
      
      if( (fabs(y1) < yjmax) && (fabs(y2) < yjmax) && ( p2 > pjmin  ) && ( p2 < pjmax  ) )
	{
	  integrate2to2(y1,y2,phi2,p2,jacobian, event_index_on_device, warehouse_on_device, pdf_on_device);
	} // pjmax, pjmin, yjmax cut

    } // if s2to2
      
  // -------------------------- Do Term A ----------------------------
  if(safinite)
    {
      xi = pow(x5,2);
      dxidx5 = 2.0f*x5;
      
      w = rts * pow(x6,2) /(1.0f - x6);
      dwdx6 = rts * x6 * (2.0f - x6) /pow( (1.0f - x6), 2);
      
      phi3 = 2.0f * pi * x7 - pi;
      dphi3dx7 = 2.0f * pi;
      
      jacobian = dy1dx1 * dp2dx2 * dy2dx3 * dphi2dx4 * dxidx5 * dwdx6 * dphi3dx7;
      singularity = xi * w;
      
      // Proceed as long as SINGULARITY is not too small:
      if(singularity > 1.0e-6f )
	{
	  coef = jacobian / singularity;

	  // Check if we are in bounds.
	  if( (fabs(y1) < yjmax) && (fabs(y2) < yjmax) && ( p2 > pjmin  ) && ( p2 < pjmax  ) )
	    {
	      p3 = xi * w;
	      p1x = - p2 * cos(phi2) - p3*cos(phi3);
	      p1y = - p2 * sin(phi2) - p3*sin(phi3);
	      p1 = sqrt( pow(p1x,2) + pow(p1y,2) );
	      if( (p1 > pjmin) && (p1 < pjmax) )
		{
		  // OK, we are in bounds. Proceed.
		  integratea(y1, p2, y2, phi2, xi, w, phi3, coef, event_index_on_device, warehouse_on_device);
		  
		  if(w < rts )
		    {
		      integratea( y1, p2, y2, phi2, xi, 0.0f, phi3, -coef, event_index_on_device, warehouse_on_device);		    
		    }// w < rts
		  
		  if(xi < (1.0f - ( exp(y1) + exp(y2) )*p2/rts) )
		    {
		      integratea( y1, p2, y2, phi2, 0.0f, w, phi3, -coef, event_index_on_device, warehouse_on_device);
		      if(w < rts ) 
			{
			  integratea( y1, p2, y2, phi2, 0.0f, 0.0f, phi3, coef, event_index_on_device, warehouse_on_device);
			} // w < rts
		    } // xi < 
		} // p1 in bound
	    } // y1, y2 in bound
	} // singularity      
    } // safinite

  // -------------------------- Do Term B ----------------------------
  if(sbfinite)
    {
      xi = pow(x5,2);
      dxidx5 = 2.0f*x5;
      
      w = rts * pow(x6,2) /(1.0f - x6);
      dwdx6 = rts * x6 * (2.0f - x6) /pow( (1.0f - x6), 2);
      
      phi3 = 2.0f * pi * x7 - pi;
      dphi3dx7 = 2.0f * pi;
      
      jacobian = dy1dx1 * dp2dx2 * dy2dx3 * dphi2dx4 * dxidx5 * dwdx6 * dphi3dx7;
      singularity = xi * w;
      
      // Proceed as long as SINGULARITY is not too small:
      if(singularity > 1.0e-6f )
	{
	  coef = jacobian / singularity;

	  // Check if we are in bounds.
	  if( (fabs(y1) < yjmax) && (fabs(y2) < yjmax) && ( p2 > pjmin  ) && ( p2 < pjmax  ) )
	    {
	      p3 = xi * w;
	      p1x = - p2 * cos(phi2) - p3*cos(phi3);
	      p1y = - p2 * sin(phi2) - p3*sin(phi3);
	      p1 = sqrt( pow(p1x,2) + pow(p1y,2) );
	      if( (p1 > pjmin) && (p1 < pjmax) )
		{
		  // OK, we are in bounds. Proceed.
		  integrateb(y1, p2, y2, phi2, xi, w, phi3, coef, event_index_on_device, warehouse_on_device);
		  
		  if(w < rts )
		    {
		      integrateb( y1, p2, y2, phi2, xi, 0.0f, phi3, -coef, event_index_on_device, warehouse_on_device);		    
		    }// w < rts
		  
		  if(xi < (1.0f - ( exp(-y1) + exp(-y2) )*p2/rts) )
		    {
		      integrateb( y1, p2, y2, phi2, 0.0f, w, phi3, -coef, event_index_on_device, warehouse_on_device);
		      if(w < rts ) 
			{
			  integrateb( y1, p2, y2, phi2, 0.0f, 0.0f, phi3, coef, event_index_on_device, warehouse_on_device);
			} // w < rts
		    } // xi < 
		} // p1 in bound
	    } // y1, y2 in bound
	} // singularity      
    }// sbfinite
 
  // -------------------------- Do Term 1 ----------------------------
  if(s1finite) 
    {
      y2 = yscale * log( (x1 + tiny)/(1.0f + tiny - x1) );
      dy2dx1 = yscale * (1.0f + 2.0f * tiny) / ( x1 + tiny) * (1.0f + tiny -x1);
      
      p1 = pscale * x2 / ( 1.0f - x2 + 2.0f * pscale/rts) ;
      dp1dx2 = pscale * (1.0f + 2.0f * pscale/rts )/ pow( (1.0f - x2 + 2.0f * pscale/rts),2);
      
      y1 = yscale * log( (x3 + tiny)/ (1.0f + tiny -x3) );
      dy1dx3 = yscale * (1.0f + 2.0f * tiny ) / ( (x3+tiny) * (1.0f + tiny -x3 ) );
      
      phi1 = 2.0f * pi * x4 - pi;
      dphi1dx4 = 2.0f * pi;

      p3 = p1 * pow(x5, 2);
      dp3dx5 = 2.0f * p1 * x5;
      
      cos0 = 1.0f - 2.0f* pow(x6,2);
      phi0 = 2.0f * pi * x7 - pi;
      domegadx67 = 8.0f * pi * x6;

      sin0 = sqrt( 1.0f - pow(cos0,2) );
      x0 = cos0;
      y0 = sin0 * cos(phi0);
      z0 = sin0 * sin(phi0);

      y3 = y1 + 2.0f * log( 1.0f + z0)/ (1.0f - z0);
      phi3 = convert(phi1 + atan2(y0, x0) );

      dy3phi3domega = 4.0f/ (1.0f - pow(z0, 2) );
      jacobian = dy2dx1* dp1dx2 * dy1dx3 * dphi1dx4 * dp3dx5 * domegadx67 * dy3phi3domega;
      
      singularity = p3 * ( cosh(y1-y3) - cos( phi1 - phi3) );
      
      // Proceed as long as SINGULARITY is not too small:
      if(singularity > 1.0e-6f )
	{
	  
	  coef = jacobian / singularity;

	  // Check if we are in bounds.
	  if( (fabs(y1) < yjmax) && (fabs(y2) < yjmax) && ( p1 > pjmin  ) && ( p1 < pjmax  ) )
	    {
	      p2x = - p1 * cos(phi1) - p3*cos(phi3);
	      p2y = - p1 * sin(phi1) - p3*sin(phi3);
	      p2 = sqrt( pow(p2x,2) + pow(p2y,2) );
	      if( (p2 > pjmin) && (p2 < pjmax) )
		{
		  // OK, we are in bounds. Proceed.
		  integrate1(y2, p1, y1, phi1, p3, y3, phi3, coef, event_index_on_device, warehouse_on_device);
		  integrate1(y2, p1, y1, phi1, p3, y1, phi1, -coef, event_index_on_device, warehouse_on_device);
		  
		  if(p3 < 0.5f*p1 )
		    {
		      integrate1(y2, p1, y1, phi1, 0.0f, y3, phi3, -coef, event_index_on_device, warehouse_on_device);
		      integrate1(y2, p1, y1, phi1, 0.0f, y1, phi1, coef, event_index_on_device, warehouse_on_device);
		    }// p3 < 0.5*p1

		} // p2 in bound
	    } // y1, y2 in bound
	}// singularity
    }// s1finite
  
  // -------------------------- Do Term 2 ----------------------------
  if(s2finite)
    {
      
      y1 = yscale * log( (x1 + tiny)/(1.0f + tiny - x1) );
      dy1dx1 = yscale * (1.0f + 2.0f * tiny) / ( x1 + tiny) * (1.0f + tiny -x1);
      
      p2 = pscale * x2 / ( 1.0f - x2 + 2.0f * pscale/rts) ;
      dp2dx2 = pscale * (1.0f + 2.0f * pscale/rts )/ pow( (1.0f - x2 + 2.0f * pscale/rts),2);
      
      y2 = yscale * log( (x3 + tiny)/ (1.0f + tiny -x3) );
      dy2dx3 = yscale * (1.0f + 2.0f * tiny ) / ( (x3+tiny) * (1.0f + tiny -x3 ) );
      
      phi2 = 2.0f * pi * x4 - pi;
      dphi2dx4 = 2.0f * pi;

      p3 = p2 * pow(x5, 2);
      dp3dx5 = 2.0f * p2 * x5;
      
      cos0 = 1.0f - 2.0f* pow(x6,2);
      phi0 = 2.0f * pi * x7 - pi;
      domegadx67 = 8.0f * pi * x6;

      sin0 = sqrt( 1.0f - pow(cos0,2) );
      x0 = cos0;
      y0 = sin0 * cos(phi0);
      z0 = sin0 * sin(phi0);

      y3 = y2 + 2.0f * log( 1.0f + z0)/ (1.0f - z0);
      phi3 = convert(phi2 + atan2(y0, x0) );

      dy3phi3domega = 4.0f/ (1.0f - pow(z0, 2) );
      jacobian = dy1dx1* dp2dx2 * dy2dx3 * dphi2dx4 * dp3dx5 * domegadx67 * dy3phi3domega;
      
      singularity = p3 * ( cosh(y2-y3) - cos( phi2 - phi3) );
      
      // Proceed as long as SINGULARITY is not too small:
      if(singularity > 1.0e-6f )
	{
	  
	  coef = jacobian / singularity;

	  // Check if we are in bounds.
	  if( (fabs(y1) < yjmax) && (fabs(y2) < yjmax) && ( p2 > pjmin  ) && ( p2 < pjmax  ) )
	    {
	      p1x = - p2 * cos(phi2) - p3*cos(phi3);
	      p1y = - p2 * sin(phi2) - p3*sin(phi3);
	      p1 = sqrt( pow(p1x,2) + pow(p1y,2) );
	      if( (p1 > pjmin) && (p1 < pjmax) )
		{
		  // OK, we are in bounds. Proceed.
		  integrate2(y1, p2, y2, phi2, p3, y3, phi3, coef, event_index_on_device, warehouse_on_device);
		  integrate2(y1, p2, y2, phi2, p3, y2, phi2, -coef, event_index_on_device, warehouse_on_device);
		  
		  if(p3 < 0.5f*p2 )
		    {
		      integrate2(y1,p2,y2,phi2,0.0f, y3,phi3,-coef, event_index_on_device, warehouse_on_device);
		      integrate2(y1,p2,y2,phi2,0.0f, y2,phi2,coef, event_index_on_device, warehouse_on_device);
		    }// p3 < 0.5*p2
		  
		} // p1 in bound
	    } // y1, y2 in bound
	}// singularity
    } // s2finite

  // ---------------- end of integrations ----------------
  
  report(0, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, event_index_on_device, warehouse_on_device);
  
}


/******************************************************************************/
/******************************************************************************/


/***********************************************************************
C
C***********************************************************************
C
C                   Routines for 2-to-2 integrals
C
C***********************************************************************
C
************************************************************************/

__device__ void integrate2to2(float y1, float y2, float phi2, float p2, float jacobian, int* event_index_on_device, float* warehouse_on_device, float* pdf_on_device)
{
  /*
    This function calculates the integrand multiplying 
    DY1 DP2 DY2 DPHI2
    in the calculation of the 2 -> 2 cross section
  */
  
  float integrand2to2;
  int nfl;
  nfl = eks_setting_on_device.nfl;
  
  float rts;
  rts = eks_setting_on_device.rts;

  // Switches
  bool sborn, svirtual;
  bool sacoll, sasoft;
  bool sbcoll, sbsoft;
  bool s1coll, s1soft;
  bool s2coll, s2soft;  
  sborn = eks_setting_on_device.sborn;
  svirtual = eks_setting_on_device.svirtual;
  sacoll = eks_setting_on_device.sacoll;
  sasoft = eks_setting_on_device.sasoft;
  sbcoll = eks_setting_on_device.sbcoll;
  sbsoft = eks_setting_on_device.sbsoft;
  s1coll = eks_setting_on_device.s1coll;
  s1soft = eks_setting_on_device.s1soft;
  s2coll = eks_setting_on_device.s2coll;
  s2soft = eks_setting_on_device.s2soft;
  
  bool stype[4];
  
  // Switch for p-pbar collisions (if true) or p-p collisions (if false)
  bool ppbar = eks_setting_on_device.ppbar;
 
  // Common block for mu values for 2-to-2 subroutines
  float muuv = eks_setting_on_device.muuvoverpj;
  float muco = eks_setting_on_device.mucooverpj;
  
  //  Loop controls,permutations,parton labels
  int perminv[4], perm[4];
  int process, nperm, flavors1, flavors2, a1, a2;
  int n1, n2, i, j;
  
  //  Parton distributions
  float prtna[13], prtnb[13];
  float effprtna[13], effprtnb[13];
  
  //   Useful parameters
  float pi, log2;
  pi = 3.141592654f;
  log2 = 0.693147181f;

  //   Parton flavors
  int aa, ab;
  
  //QCD functions
  float d, d0;
  float jab, ja1, ja2, jba, jb1, jb2, j1a, j1b, j12, j2a, j2b, j21;
  
  //Kinematical variables
  float shat, uhat, that;
  float xa, xb;
  float pin[4][4], k[4][4];
  float ey1, ey2;
  float p1, phi1;

  // Ellis and Sexton scale parameter
  float qes;
  
  //QCD functions
  float psi4, psi6ns, psicab, psica1, psica2, psicb1, psicb2, psic12;
  float l, leffa, leffb, sign;
  
  //Precalculated logs
  float logs, logt, logu, logmusq, log16p2sq;
  float logxa, logxb;
  
  //   Each column is one of the permutations we need.
  int  perms[4][4][12] =      
    {
      1,1,1,1,1,1,2,2,2,2,3,3,
      2,2,3,3,4,4,3,3,4,4,4,4,
      3,4,2,4,2,3,1,4,1,3,1,2,
      4,3,4,2,3,2,4,1,3,1,2,1,
      
      1,1,1,2,2,3,0,0,0,0,0,0,
      2,3,4,3,4,4,0,0,0,0,0,0,
      3,2,2,1,1,1,0,0,0,0,0,0,
      4,4,3,4,3,2,0,0,0,0,0,0,
      
      1,1,1,2,2,2,3,3,3,4,4,4,
      2,3,4,1,3,4,1,2,4,1,2,3,
      3,2,2,3,1,1,2,1,1,2,1,1,
      4,4,3,4,4,3,4,4,2,3,3,2,
      
      1,0,0,0,0,0,0,0,0,0,0,0,
      2,0,0,0,0,0,0,0,0,0,0,0,
      3,0,0,0,0,0,0,0,0,0,0,0,
      4,0,0,0,0,0,0,0,0,0,0,0
    };
  int nperms[4] = {12,6,12,1};
  
  // -- Initialize computation ------------
  p1 = p2;
  phi1 = convert(phi2 + pi);
  setmu(2, p1, y1, phi1, p2, y2, phi2, 0.0f, 0.0f, 0.0f, &muuv, &muco);
  
  ey1 = exp(y1);
  ey2 = exp(y2);
  xa = (ey1 + ey2)*p2/rts;
  xb = (1.0f/ey1 + 1.0f/ey2)* p2/rts;
  shat = pow(p2,2) * (2.0f + ey2/ey1 + ey1/ey2 );
  that = -pow(p2,2) * (1.0f + ey2/ey1);
  uhat = -pow(p2,2) * (1.0f + ey1/ey2);
  
  if( (xa > 1.0f) || (xb > 1.0f) ) return;

  qes = rts/10.0f;
  
  //  Momentum matrix (incoming)
  pin[0][0] = 0.0f;
  pin[0][1] = shat/2.0f;
  pin[0][2] = that/2.0f;
  pin[0][3] = uhat/2.0f;
  pin[1][0] = shat/2.0f;
  pin[1][1] = 0.0f;
  pin[1][2] = uhat/2.0f;
  pin[1][3] = that/2.0f;
  pin[2][0] = that/2.0f;
  pin[2][1] = uhat/2.0f;
  pin[2][2] = 0.0f;
  pin[2][3] = shat/2.0f;
  pin[3][0] = uhat/2.0f;
  pin[3][1] = that/2.0f;
  pin[3][2] = shat/2.0f;
  pin[3][3] = 0.0f;
  
  d0 = pow(alphas(muuv),2) * p2/2.0f/pow(rts,4);
  d = d0 * alphas(muuv)/2.0f/pi;
  
  logs = log(shat/pow(qes,2) );
  logt = log(-that/pow(qes,2) );
  logu = log(-uhat /pow(qes,2) );
  logmusq = log( pow(muco,2)/pow(qes,2) );
  logxa = log( pow( xa/(1.0f-xa), 2  )  );
  logxb = log( pow( xb/(1.0f-xb), 2  )  );
  log16p2sq = log( 16.0f* pow(p2,2)/pow(qes,2) );
  
  // The tilde J_MN soft integrals
  j21 = 0.25f * pow( (2.0f * log2 - logs),2) - 0.25f * pow( (2.0f*logs - logt - logu), 2) + (logs - logt)*(logs - logu);
  j2a = 0.25f * pow( (2.0f * log2 - logu),2) - 0.25f * pow( (2.0f*logs - logt), 2) - log2 * log(2.0f + uhat/shat);
  j2b = 0.25f * pow( (2.0f * log2 - logt),2) - 0.25f * pow( (logs - logu),2) - log2 * log(2.0f + that/shat);
  
  j12 = j21;
  j1a = j2b;
  j1b = j2a;
  
  jab = 0.25f * pow( (logxa - logs),2) + pow(pi,2) / 12.0f;
  ja1 = 0.25f * pow( (logxa - logt),2) - 0.25f * pow( log(0.5f + 0.25f * that/shat),2) + pow(log2,2) + pow(pi,2)/12.0f - 0.5f * li2( ( shat + that )/(2.0f*shat + that) ); 
  ja2 = 0.25f * pow( logxa-logt,2) - 0.25f * pow( log(0.5f + 0.25f * uhat/shat),2 ) + pow(log2,2) + pow(pi,2)/12.0f - 0.5f * li2( (shat + uhat)/(2.0f*shat + uhat) );
  
  jba = 0.25f * pow( logxb - logs, 2) + pow(pi,2)/12.0f;
  jb1 = ja2 - 0.25f * pow( logxa - logu,2) + 0.25f * pow( logxb - logu,2 );
  jb2 = ja1 - 0.25f * pow( logxa - logt,2) + 0.25f * pow(logxb - logt,2);

  // ---- Calculate parton distributions. For p-pbar collisions, exchange
  // partons -> antipartons for hadron B.
  
  for(aa = -nfl; aa < nfl; ++aa)
    {
      prtna[aa] = parton(aa, 0, pdf_on_device)/(xa*wnum(aa));
      effprtna[aa] = effparton(aa,xa,muco, 0 , pdf_on_device)/(xa*wnum(aa));
    } // aa
  if(ppbar)
    {
      for(ab = -nfl; ab < nfl; ++ab)
	{
	  prtnb[ab] = parton(-ab, 1, pdf_on_device)/(xb*wnum(ab));
	  effprtna[ab] = effparton(-ab,xb,muco, 1, pdf_on_device)/(xb*wnum(ab));
	}      
    } //if ppbar
  else
    {
      for(ab = -nfl; ab < nfl; ++ab)
	{
	  prtnb[ab] = parton(ab,1, pdf_on_device)/(xb*wnum(ab));
	  effprtna[ab] = effparton(ab,xb,muco, 1, pdf_on_device)/(xb*wnum(ab));
	} // for ab 
    } //else
  
  // ----- Initialize sum:
  integrand2to2 = 0.0f;
  
  // ----- Sum over processes A,B,C,D (PROCESS = 1,2,3,4) -------
  for(process = 0; process < 4; ++process)
    {
      if(stype[process])
	{
	  // ----- Sum over independent permutations the partons
	  for(nperm = 0; nperm < nperms[process]; ++nperm)
	    {
	      //  Calculate inverse permutation
	      for(i = 0; i < 4; ++i)
		{
		  j = perms[process][i][nperm];
		  perminv[j] = i;
		  perm[i] = j;
		} // i 
	      
	      //      Set K matrix as permutation of P matrix:
	      //      K(I) = P(J) with J = PERM(I).
	      for(i = 0; i < 4; ++i)
		{
		  for( j = 0; j < 4; ++j)
		    {
		      k[j][i] = pin[ perms[process][i][nperm] ] [ perms[process][j][nperm] ];
		    }//j 
		} // i
	      
	      //  Here we calculate functions that don't depend on the choice
	      //  of quark flavors. Note that these are the 'tilde' functions,
	      //  so we need a crossing sign when we use them.
	      psi4 = psitilde4(process, k);
	      psi6ns = psitilde6ns(process,k);
	      psicab = psictilde(process, perminv[0],perminv[1],k);
	      psica1 = psictilde(process, perminv[0],perminv[2],k);
	      psica2 = psictilde(process, perminv[0],perminv[3],k);
	      psicb1 = psictilde(process, perminv[1],perminv[2],k);
	      psicb2 = psictilde(process, perminv[1],perminv[3],k);
	      psic12 = psictilde(process, perminv[2],perminv[3],k);

	      // Find how many flavors for incoming partons to sum over.
	      if (process == 0)
		{ 
		  flavors1 = nfl;
		  flavors2 = nfl;
		}
	      else if (process == 1) {
		flavors1 = nfl;
		flavors2 = 1;
	      }
	      else if (process == 2) 
		{
		  flavors1 = nfl;
		  flavors2 = 1;
		}
	      else if (process == 3) 
		{
		  flavors1 = 1;
		  flavors2 = 1;
		} // if process, else if
	      
	      //----- Sum over particular parton flavors to calculate luminosities
	      l = 0.0f;
	      leffa = 0.0f;
	      leffb = 0.0f;
	      
	      for(n1 = 0; n1 < flavors1; ++n1)
		{
		  for(n2 = 0; n2 < flavors2; ++n2)
		    {
		      if ( !((process == 0) && (n1 == n2)) )
			{
			  //  Find flavors for incoming partons: a(J) = a_0(I) with J = PERM(I)
			  //  and set outgoing flavors to 0 (gluon) or 101 (generic
			  //  quark or antiquark).
			  if (process == 0) 
			    {
			      if (perm[0] == 0)
				aa =   n1;
			      else if (perm[1] == 0) 
				aa =   n2;
			      else if (perm[2] == 0) 
				aa = - n1;
			      else if (perm[3] == 0) 
				aa = - n2;
			      if (perm[0] == 1) 
				ab =   n1;
			      else if (perm[1] == 1) 
				ab =   n2;
			      else if (perm[2] == 1) 
				ab = - n1;
			      else if (perm[3] == 1) 
				ab = - n2;
			      a1 = 101;
			      a2 = 101;
			    } // process ==0
			  else if (process == 1) 
			    {
			      if (perm[0] == 0) 
				aa =   n1;
			      else if (perm[1] == 0) 
				aa =   n1;
			      else if (perm[2] == 0) 
				aa = - n1;
			      else if (perm[3] == 0) 
				aa = - n1;
			      if (perm[0] == 1) 
				ab =   n1;
			      else if (perm[1] == 1) 
				ab =   n1;
			      else if (perm[2] == 1) 
				ab = - n1;
			      else if (perm[3] == 1) 
				ab = - n1;
			      a1 = 101;
			      a2 = 101;
			    } // process == 1
			  else if (process == 2) 
			    {
			      if (perm[0] == 0) 
				aa =   n1;
			      else if (perm[1] == 0) 
				aa = - n1;
			      else if (perm[2] == 0) 
				aa =   0;
			      else if (perm[3] == 0) 
				aa =   0;
			      if (perm[0] == 1) 
				ab =   n1;
			      else if (perm[1] == 1) 
				ab = - n1;
			      else if (perm[2] == 1) 
				ab =   0;
			      else if (perm[3] == 1) 
				ab =   0;
			      if (perm[0] == 2) 
				a1 =  101;
			      else if (perm[1] == 2) 
				a1 =  101;
			      else if (perm[2] == 2) 
				a1 =   0;
			      else if (perm[3] == 2) 
				a1 =   0;
			      if (perm[0] == 3) 
				a2 =  101;
			      else if (perm[1] == 3) 
				a2 =  101;
			      else if (perm[2] == 3) 
				a2 =   0;
			      else if (perm[3] == 3) 
				a2 =   0;
			    }// process == 2
			  else if (process == 3) 
			    {
			      aa = 0;
			      ab = 0;
			      a1 = 0;
			      a2 = 0;
			    } // process == 3
			  
			  // Calculate luminosity factors
			  l = l + prtna[aa]*prtnb[ab];
			  leffa = leffa + effprtna[aa]*prtnb[ab];
			  leffb = leffb + prtna[aa]*effprtnb[ab];
			  

			  //  Here we close the loops for flavor sums, having calculated the
			  //  luminosity factors.  In the following parts of the calculation,
			  //  the indices AA,AB,A1,A2 are used, but the calculation needs only
			  //  to remember if Ax.EQ.0 for gluon or Ax.NE.0 for quark or antiquark.
			  //  As we exit these loops, the Ax are correctly set for this purpose.
			  
			  //  ----- Close IF that omits the case N1=N2 in process A.
			}// if !((process == 0) && (n1 == n2))  

		      // ----- Close loop  for sum over flavors N2.
		    } // for(n2 = 0; n2 < flavors2; ++n2)
		  
		  // ----- Close loop for sum over flavors N1.
		} // for n1
	      
	      // Get crossing sign
	      if ( ((aa == 0) && (ab != 0)) || ((ab == 0) && (aa != 0))     ) 
		sign = - 1.0f;
	      else
		sign =   1.0f;
	      
	      // Calculate contribution from Born graph
	      if (sborn) 
		integrand2to2 = integrand2to2 + d0 * l * sign * psi4;
	      
	      // Calculate contribution from virtual graphs
	      if (svirtual) 
		integrand2to2 = integrand2to2 + d * l  * sign * psi6ns;
	      
	      //  calculate contribution from term a
	      if(sacoll) 
		integrand2to2 = integrand2to2
		  + d * sign * psi4
		  *( - l * logmusq
		     *( gammaval(aa) -  color(aa) *logxa  )
		     + leffa );

	      if(sasoft) 
		integrand2to2 = integrand2to2
		  + d * l * sign
		  *( psicab * jab + psica1 * ja1 + psica2 * ja2);
	      
	      //  calculate contribution from term b
	      if(sbcoll) 
		integrand2to2 = integrand2to2
		  + d * sign * psi4
		  *( - l * logmusq
		     *( gammaval(ab) -  color(ab) * logxb  )
		     + leffb );
	      
	      if(sbsoft) 
		integrand2to2 = integrand2to2
		  + d * l * sign
		  *( psicab * jba + psicb1 * jb1 + psicb2 * jb2);
	      
	      //  calculate contribution from term 1
	      if(s1coll) 
		integrand2to2 = integrand2to2
		  + d * l * sign * psi4
		  *( - log16p2sq
		     *( gammaval(a1) - 2.0f * color(a1) * log2  )
		     + gammaprime(a1) - 2.0f * color(a1) * pow(log2,2) );

	      if(s1soft) 
		integrand2to2 = integrand2to2
		  + d * l * sign
		  *( psica1 * j1a + psicb1 * j1b + psic12 * j12);

	      //  calculate contribution from term 2
	      if(s2coll) 
		integrand2to2 = integrand2to2
		  + d * l * sign * psi4
		  *( - log16p2sq
		     *( gammaval(a2) - 2.0f * color(a2) * log2  )
		     + gammaprime(a2) - 2.0f * color(a2) * pow(log2,2) );

	      if(s2soft) 
		integrand2to2 = integrand2to2
		  + d * l * sign
		  *( psica2 * j2a + psicb2 * j2b + psic12 * j21);
	      // ---- Done!
	      
	    } // for nperm = 0  ----- Close DO for sum over permutations.
	} // if stype[process]  ----- Close IF that omits process if switch is off.
    } //  ----- Close DO for sum over processes A,B,C,D.
  
  integrand2to2 = integrand2to2 * jacobian;
  
  report(2, p1, y1, phi1, p2, y2, phi2, 0.0f, 0.0f, 0.0f, integrand2to2, event_index_on_device, warehouse_on_device);
}

/******************************************************************************/
/******************************************************************************/

__device__ void integratea(float, float, float, float, float, float, float, float , int* event_index_on_device, float* warehouse_on_device)
{

}

/******************************************************************************/
/******************************************************************************/

__device__ void integrateb(float, float, float, float, float, float, float, float, int* event_index_on_device, float* warehouse_on_device)
{

}

/******************************************************************************/
/******************************************************************************/

__device__ void integrate1(float, float, float, float, float, float, float, float, int* event_index_on_device, float* warehouse_on_device)
{

}

/******************************************************************************/
/******************************************************************************/

__device__ void integrate2(float, float, float, float, float, float, float, float, int* event_index_on_device, float* warehouse_on_device)
{
  
}

/******************************************************************************/
/******************************************************************************/

__device__ void setmu(int npartons, float p1, float y1, float phi1, float p2, float y2, float phi2, float p3, float y3, float phi3, float* muuv, float* muco)
{
  *muuv = eks_setting_on_device.muuvoverpj * p1;
  *muco = eks_setting_on_device.mucooverpj * p1;
}

/******************************************************************************/
/******************************************************************************/

__device__ float convert(float)
{

}

/******************************************************************************/
/******************************************************************************/

__device__ float effparton(float a, float x, float scale, int a_or_b, float* pdf_on_device)
{
  //  Integrates FCOLL(z) from X to 1, Monte Carlo style,
  //  if called many times.
  float z, dzdx1, random;
  float factor, subtraction;
  float fcoll;
  float rts = eks_setting_on_device.rts;
  float nfl = eks_setting_on_device.nfl;

  // z = x + (1.0f - x) * rand() / float(RAND_MAX);
  z = find_z(a_or_b);
  dzdx1 = 1.0f - x;
  //------
  //  calculate fcoll(z), the integrand for the 2 -> 2 coll pieces.
  
  factor = 2.0f * log(rts*x*(1.0f - z)/scale/z);
  subtraction = factor * 2.0f * color(a) * parton(a,a_or_b, pdf_on_device) /z/(1.0f - z);
  
  fcoll = - subtraction;

  if (a != 0) 
    {
      fcoll = fcoll + (factor * altarelli(a,a,z) - altrliprime(a,a,z)) * parton(a,2+a_or_b, pdf_on_device) /z + (factor * altarelli(a,0,z)- altrliprime(a,0,z)) * parton(0,2+a_or_b, pdf_on_device) /z;
    }
  else
    {
      for( int aprime = - nfl; aprime <= nfl; ++aprime )
	{
	  fcoll = fcoll + (factor * altarelli(a,aprime,z)- altrliprime(a,aprime,z)) * parton(aprime,2+a_or_b, pdf_on_device) /z;
	} // for aprime
    } // if a!= 0

  //------
  return fcoll * dzdx1;

}

/******************************************************************************/
/******************************************************************************/

  __device__ float altarelli(int aout,int ain, float z)
  {
    /*   The Altarelli-Parisi kernel P_{AOUT/AIN}(z) */

    float return_number = 0;

    float v = 8.0f;
    float n = 3.0f;
    float twon = 6.0f;
    float cf = 4.0f/3.0f;
    
    if ( (z > 1.0f) || (z < 0.0f) )
      {
        return 0.0f;
      }
    
    if ( (aout == 0) && (ain == 0) )
      {
        return_number = twon *( (1.0f - z) /z + z /(1.0f - z) + z*(1.0f - z) );
      }
    else if( (aout != 0) && (ain == aout) )
      {
	return_number  = cf *(1.0f + pow(z,(float)2.0f) ) /(1.0f - z);
      }
    else if ( (aout != 0) && (ain == 0) )
      {
	return_number  = 0.5f * ( pow(z,(float)2.0f) + pow( ( 1.0f - z), 2.0f) );
      }
    else if ( (aout == 0) && (ain != 0) )
      {
        return_number = cf *( 1.0f + pow( (1.0f - z),2.0f) ) /z;
      }
    else
      {
	return_number = 0.0f;
      }
    return return_number;

  }

/******************************************************************************/
/******************************************************************************/

  __device__ float altrliprime(int aout,int ain, float z)
  {
    
    /*
      The derivative of the Altarelli-Parisi kernel
       P_{AOUT/AIN}(z,\epsilon) w.r.t. \epsilon (\epsilon = 0)
    */
    float return_value = 0.0f;

    float v = 8.0f;
    float n = 3.0f;
    float twon = 6.0f;
    float cf = 4.0f/3.0f;

    if ( (z > 1.0f) || (z < 0.0f) ) 
      {
	return_value = 0.0f;
        return return_value;
      }
    
    if ( (aout == 0.0f) && (ain == 0) )
      {
        return_value = 0.0f;
      }
    else if ( (aout != 0.0f) && (ain == aout) )
      {
        return_value = - cf * (1.0f - z);
      }
    else if ( (aout != 0.0f) && (ain == 0) )
      {
        return_value = 0.5f * ( pow(z,(float)2.0f) + pow( (1.0f - z), 2.0f) - 1.0f );
      }
    else if ( (aout == 0) && (ain != 0) )
	{
	  return_value = - cf * z;
	}
    else
      {
        return_value = 0.0f;
      }
    
    return return_value;
    

  }
/******************************************************************************/
/******************************************************************************/

__device__ float alphas(float)
{
  return alphas_input[blockIdx.x*blockDim.x+threadIdx.x];
}

/******************************************************************************/
/******************************************************************************/

__device__ float parton(int flavor, int a_or_b, float* pdf_on_device)
{
  return pdf_on_device[blockIdx.x*blockDim.x+threadIdx.x*44 + flavor + 5 + 11*a_or_b];

}


/******************************************************************************/
/******************************************************************************/

__device__ float find_z(int a_or_b)
{
  return z_input[blockIdx.x*blockDim.x+threadIdx.x*2 + a_or_b];

}

/******************************************************************************/
/******************************************************************************/


__device__ float theta( bool condition )
{
  float theta;
  // theta function
  if(condition) theta = 1.0f;
  else theta = 0.0f;
  return theta;
}

/******************************************************************************/
/******************************************************************************/

__device__ float li2(float x)
{
  // Let ERR be the target fractional error
  float xn;
  float n;
  float err = 1.0e-6f;
  float li2 = 0.0f;
  
  xn = 1.0f;
  for(n = 1; n < 100.0f; ++n)
    {
      xn = xn*x;
      if(xn < err) return li2;
      li2 = li2 + xn/pow(n,2);
    }
  return li2;
}

/******************************************************************************/
/******************************************************************************/

__device__ float wnum(float nparton)
{
  //   This function gives the spin and color weighting for
  //   parton NPARTON: 6 for quarks and antiquarks, 16 for gluons
  float wnum = 0.0f;
  
  if(nparton == 0) wnum = 16.0f;
  else wnum = 6.0f;
  
  return wnum;
}
/*****************************************************************************
C***********************************************************************
C  --------------------------------------------
C  Functions for Born and Virtual Contributions
C  --------------------------------------------
C***********************************************************************
******************************************************************************/


__device__ float psitilde4(int process, float k[4][4])
{
  // This function gives the psitilde^(4) functions from E.S.
  float psitilde4;
    
  float s = k[1][0];
  float t = k[2][0];
  float u = k[3][0];
  float n = 8.0f;
  float v = 3.0f;
  
  if(process == 0)
    {
      psitilde4 = 2.0f * v * ( pow(s,2) + pow(u,2) ) / pow(t,2);
    }
  else if(process == 1)
    {
      psitilde4 = 2.0f * v * ( pow(s,2) + pow(u,2) ) / pow(t,2) + 2.0f * v * ( pow(s,2) + pow(t,2) ) / pow(u,2) - 4.0f * v / n * pow(s,2) / u / t;
    }
  else if(process == 2)
    {
      psitilde4 = 2.0f * v / n * ( v / u / t - 2.0f * pow(n,2)/pow(s,2) ) * (pow(t,2) + pow(u,2) );
    }
  else if(process == 3)
    {
      psitilde4 = 4.0f * v * pow(n,2) * (pow(s,2)/pow(t,2)/pow(u,2) ) /pow(s,2)/pow(t,2)/pow(u,2) * (pow(s,4) + pow(t,4) + pow(u,4) );
    }
  else psitilde4 = 0.0f;
  
  return psitilde4;
}

/******************************************************************************/
/******************************************************************************/

__device__ float psitilde6ns(int process, float k[4][4])
{

  //      This function gives the psitilde^(6)_NS functions calculated
  //      by subtracting the singular pieces from from E.S.
  float psitilde6ns;
  float nflavor;
  float muuv;
  float qes;
  float pi, n, v;
  float s, t, u;
  float ls, lt, lu, lmu, l2s, l2t, l2u;
  float rts = eks_setting_on_device.rts;
  
  nflavor = eks_setting_on_device.nflavor;
  muuv = eks_setting_on_device.muuvoverpj;

  qes = rts/10.0f;
  
  pi = 3.141592654f;
  v = 8.0f;
  n = 3.0f;
  
  s = 2.0f * k[1][0];
  t = 2.0f*  k[2][0];
  u = 2.0f * k[3][0];
  
  // Here are the Ellis and Sexton log functions
  ls = log( abs( s/pow(qes,2))  );
  lt = log( abs( t/pow(qes,2))  );
  lu = log( abs( u/pow(qes,2))  );
  lmu = log( abs( pow(muuv,2)/pow(qes,2) ));
  l2s = pow(ls,2) - pow(pi,2) * theta(s > 0.0f);
  l2t = pow(lt,2) - pow(pi,2) * theta(t > 0.0f);
  l2u = pow(lu,2) - pow(pi,2) * theta(u > 0.0f);
  
  // Find what type, then calculate
  if(process == 0)
    {
      psitilde6ns = 
	v/( 9.0f*n*pow(t,2) ) * (- 9*pow(pi,2)* (pow(n,2)-4 ) * (pow(s,2) - pow(u,2) ) 
	+  2*(72 + 13*pow(n,2) - 10*n*nflavor + 9*pow(n,2)*pow(pi,2))
	*(pow(s,2) + pow(u,2))
	+  6*n*(11*n - 2*nflavor)*( pow(s,2) + pow(u,2))*lmu
	- 36*t*u*ls
	+  3*(- 6*pow(s,2) - 30*pow(u,2) + 4*n*nflavor*(pow(s,2) + pow(u,2))
	      - pow(n,2)*(7*pow(s,2) + 3*pow(t,2) + pow(u,2)))*lt
	- 18*s*t*(pow(n,2) -2)*lu
	- 36*(3*pow(s,2) + pow(u,2))*ls*lt
	- 18*(pow(n,2) -2)*(pow(s,2) + 3*pow(u,2))*lt*lu
	+ 18*(pow(s,2) - pow(u,2))*l2s
	+  9*(pow(n,2)*(pow(s,2) + 3*pow(u,2)) + 2*(3*pow(s,2) - pow(u,2)))*l2t
	-  9*(pow(n,2) - 2)*(pow(s,2) - pow(u,2) )*l2u);      
    }
  else if(process == 1)
    {
      psitilde6ns =
	40*n*nflavor*pow(s,2)*t*u - 4*pow(n,2)*pow(s,2)*t*u*(13 + 9*pow(pi,2))
        - 9*t*u*(32*pow(s,2) + pow(pi,2)*(pow(s,2) + 2*t*u))
        + 36*n*(pow(s,2)*(4 + pow(pi,2)) * (pow(t,2) + pow(u,2))
		+ (4 - pow(pi,2))  *  (pow(t,4) + pow(u,4)))
        + pow(n,3)*(pow(s,2)*(26 + 9*pow(pi,2))*(pow(t,2) + pow(u,2))
		    + (26 + 27*pow(pi,2))*(pow(t,4) + pow(u,4)))
	- 20*pow(n,2)*nflavor*(pow(t,4) + pow(u,4) + pow(s,2)*(pow(t,2) + pow(u,2)));

      psitilde6ns =  psitilde6ns
	+ 6*n*(11*n - 2*nflavor)
	*(-2*pow(s,2)*t*u + n*(pow(t,4) + pow(u,4) + pow(s,2)*(pow(t,2) + pow(u,2)))) *lmu
	- 36*n*t*u*(pow(t,2) + pow(u,2))*ls
	- 6*u*(- 2*pow(n,2)*pow(s,2)*t + 2*n*nflavor*pow(s,2)*t
	       - 2*pow(n,2)*nflavor*u*(pow(s,2) + pow(u,2))
	       + 6*s*t*(t + 2*u) + pow(n,3)*(-3*pow(t,3) + 2*pow(t,2)*u
					     + 7*t*pow(u,2) + 4*pow(u,3))
	       + 3*n*(2*pow(t,3) + 3*pow(t,2)*u + 2*t*pow(u,2) + 6*pow(u,3))) *lt
	- 6*t*(- 2*pow(n,2)*pow(s,2)*u + 2*n*nflavor*pow(s,2)*u
	       - 2*pow(n,2)*nflavor*t*(pow(s,2) + pow(t,2))
	       + 6*s*u*(2*t + u)
	       + pow(n,3)*(4*pow(t,3) + 7*pow(t,2)*u + 2*t*pow(u,2) - 3*pow(u,3))
	       + 3*n*(6*pow(t,3) + 2*pow(t,2)*u + 3*t*pow(u,2) + 2*pow(u,3))) * lu;

      psitilde6ns =  psitilde6ns
	- 36*u*(- pow(s,2)*t - pow(n,2)*pow(s,2)*t + n*u*(3*pow(s,2) + pow(u,2))) *ls*lt
	- 36*t*(- pow(s,2)*u - pow(n,2)*pow(s,2)*u + n*t*(3*pow(s,2) + pow(t,2))) *ls*lu
	- 18*(  2*n*(-2 + pow(n,2))*(pow(t,2) - t*u + pow(u,2))
		*(2*pow(t,2) + 3*t*u + 2*pow(u,2))
		+ t*u*(3*pow(t,2) + 4*t*u + 3*pow(u,2))) * lt*lu
	+ 18*n*t*u*(pow(s,2) + pow(t,2) + pow(u,2)) * l2s
	+ 9*u*(- 2*pow(n,2)*pow(s,2)*t - 4*n*(pow(s,3) - s*pow(t,2) - pow(t,3))
	       - t*(3*pow(t,2) + 8*t*u + 3*pow(u,2))
	       - 2*pow(n,3)*(pow(t,3) - t*pow(u,2) - 2*pow(u,3))) * l2t
	+ 9*t*(- 2*pow(n,2)*pow(s,2)*u - 4*n*(pow(s,3) - s*pow(u,2) - pow(u,3))
	       - u*(3*pow(t,2) + 8*t*u + 3*pow(u,2))
	       + 2*pow(n,3)*(2*pow(t,3) + pow(t,2)*u - pow(u,3))) * l2u;

      psitilde6ns = v/(9*pow(n,2)*pow(t,2)*pow(u,2)) * psitilde6ns;
    }
  else if(process == 2)
    {
      psitilde6ns =
        3*pow(s,2)*( -7*(pow(t,2) + pow(u,2) ) + pow(pi,2)*(3*pow(s,2) -2*t*u) )
        + 3*pow(n,2)*( 14*(pow(t,4) + pow(u,4)) + t*u*(13*pow(s,2) + 4*t*u )
			+ pow(pi,2)*(pow(t,3)*u + 2*pow(t,2)*pow(u,2) + t*pow(u,3))    )
        - 3*pow(n,4)*( 7*(pow(t,4) + pow(u,4))+ t*u*(pow(s,2) + 10*t*u)
			-  pow(pi,2)*pow(t-u,2)*(pow(t,2) + pow(u,2))  )
        + 2*n*(11*n - 2*nflavor)*(-pow(s,2) + pow(n,2)*(pow(t,2) + pow(u,2)))
	*(pow(t,2) + pow(u,2)) * lmu
        - 3*t*u*(  4*pow(t,2) + 8*t*u + 4*pow(u,2)
		   + pow(n,2)*(-1 + pow(n,2))*(pow(t,2) - 10*t*u + pow(u,2)) ) * ls
        + 3*s*u*(s + pow(n,2)*u)*(2*t + 3*u + pow(n,2)*(2*t - 3*u)) * lt
	+ 3*s*t*(s + pow(n,2)*t)*(3*t + 2*u + pow(n,2)*(-3*t + 2*u)) * lu;

      psitilde6ns = psitilde6ns 
        - 6*(  pow(n,4)*u*(u - t)*(2*pow(t,2) + t*u + pow(u,2))
	       + s*(s - pow(n,2)*t)*(2*pow(t,2) + 2*t*u + pow(u,2))  ) * ls*lt
        - 6*(  pow(n,4)*t*(t - u)*(pow(t,2) + t*u + 2*pow(u,2))
	       + s*(s - pow(n,2)*u)*(pow(t,2) + 2*t*u + 2*pow(u,2))  ) * ls*lu
        + 12*pow(n,2)*pow(s,2)*(pow(t,2) + pow(u,2)) * lt*lu
        + 3*( 2*pow(s,4) - 2*pow(n,4)*t*u*(pow(t,2) + pow(u,2))
	      + pow(n,2)*(pow(t,2) + t*u + 2*pow(u,2))*(2*pow(t,2) + t*u + pow(u,2)) ) *l2s
        - 3*s*(- pow(n,4)*u*(2*pow(t,2) - t*u + pow(u,2))
	       + (pow(n,2)*t - s)*(2*pow(t,2) + 2*t*u + pow(u,2)) ) *l2t
        - 3*s*(- pow(n,4)*t*(2*pow(u,2) - t*u + pow(t,2) )
	       + (pow(n,2)*u - s)*( 2*pow(u,2) + 2*t*u + pow(t,2))) *l2u;

      psitilde6ns = v/(3.0f*pow(n,2)*pow(s,2)*t*u) * psitilde6ns;
    }
  else if(process == 3)
    {
      psitilde6ns =
	2*pow(n,2)*nflavor*( - (66 + 27*pow(pi,2))*pow(s,2)*pow(t,2)*pow(u,2)
			     + 40*(pow(s,6) + pow(t,6) + pow(u,6)))
        + 2*pow(n,3)*(  6*(125 - 27*pow(pi,2))*pow(s,2)*pow(t,2)*pow(u,2)
			- 4*(67 - 9*pow(pi,2))*(pow(s,6) + pow(t,6) + pow(u,6)))
        + 6*pow(n,2)*(11*n - 2*nflavor)*pow( (pow(s,2) +  pow(t,2) + pow(u,2)),3) *lmu
        + 6*pow(n,2)*t*u*(pow(s,2) + pow(t,2) + pow(u,2))
	*(  nflavor *(5* pow(t,2) + 2*t*u + 5* pow(u,2))
	    - 2*n*(7* pow(t,2) - 8*t*u + 7* pow(u,2)) ) *ls;
      
      psitilde6ns = psitilde6ns 
        + 6*pow(n,2)*s*u*(pow(s,2) +  pow(t,2) +  pow(u,2))
	*(  nflavor *(5* pow(s,2) + 2*s*u + 5* pow(u,2))
	    - 2*n*(7* pow(s,2) - 8*s*u + 7*pow(u,2)) ) *lt
        + 6* pow(n,2)*s*t*( pow(s,2) + pow(t,2) + pow(u,2))
	*(  nflavor* (5* pow(s,2) + 2*s*t + 5*pow(t,2))
	    - 2*n*(7*pow(s,2) - 8*s*t + 7* pow(t,2)) ) *lu
        - 36* pow(n,2)* pow(u,2)*( nflavor*s*t*( pow(s,2) +  pow(t,2))
				   + 2*n*(2* pow(s,4) + 2* pow(s,3)*t + 3* pow(s,2)*pow(t,2)+ 2*s* pow(t,3)
					  + 2*pow(t,4) ) )*ls*lt;

      psitilde6ns = psitilde6ns 
        - 36* pow(n,2)* pow(s,2)*(  nflavor*t*u*( pow(t,2) +  pow(u,2))
				    + 2*n*(2* pow(t,4) + 2* pow(t,3)*u + 3* pow(t,2)* pow(u,2) + 2*t* pow(u,3)
					   + 2* pow(u,4)))*lt*lu
        - 36* pow(n,2)* pow(t,2)*(  nflavor*s*u*( pow(s,2) +  pow(u,2))
				    + 2*n*(2* pow(s,4) + 2* pow(s,3)*u + 3*pow(s,2)* pow(u,2) + 2*s*pow(u,3)
					   + 2*pow(u,4)))*ls*lu
        + 18*pow(n,2)* pow(s,2)*t*u*(4*n*(pow(t,2) + pow(u,2))
				     - nflavor*( pow(t,2) + 3*t*u +  pow(u,2)))*l2s
        + 18*pow(n,2)*s*pow(t,2)*u*(4*n*( pow(s,2) +  pow(u,2))
				    - nflavor*( pow(s,2) + 3*s*u +  pow(u,2)))*l2t
        + 18*pow(n,2)*s*t*pow(u,2)*(4*n*(pow(s,2) + pow(t,2))
				      - nflavor*( pow(s,2) + 3*s*t +  pow(t,2)))*l2u;
      
      psitilde6ns = v/(9.0f*pow(s,2)*pow(t,2)*pow(u,2)) * psitilde6ns;
    }
  else psitilde6ns = 0.0f;
  
  
  return psitilde6ns;
}

/*****************************************************************************
C
C***********************************************************************
C  --------------------------------
C  Functions for Soft Contributions
C  --------------------------------
C***********************************************************************
C
******************************************************************************/

__device__ float psictilde(int process, int i1, int i2, float k[4][4])
{
  float psictilde;
  
  //      This function gives the lambdatilde(a1,a2,s,t,u) functions
  
  int ii1, ii2;
  float s = 2.0f * k[1][0];
  float t = 2.0f * k[2][0];
  float u = 2.0f * k[3][0];
  
  float hat, hau, hb, hct, hcu, hcs, hds, hdt, hdu;
  float n = 3.0f;
  float v = 8.0f;
  float cf = 4.0f/3.0f;
  
  //Make it a symmetric function of I1,I2
  if(i1 < i2)
    {
      ii1 = i1;
      ii2 = i2;
    }
  else
    {
      ii1 = i2;
      ii2 = i1;
    }
  
  if(process == 0)
    {
      hat = 2.0f * (pow(s,2) + pow(u,2) ) /pow(t,2);
      if( (ii1 == 0) && ( ii2 == 1) ) psictilde = 4.0f * cf * hat;
      else if ( (ii1 == 0) && ( ii2 == 1) ) psictilde = -2.0f * cf * hat;
      else if ( (ii1 == 0) && ( ii2 == 2) ) psictilde = 2.0f * cf * ( pow(n,2) -2.0f) * hat  ;
      else if ( (ii1 == 2) && ( ii2 == 3) ) psictilde = 4.0f * cf * hat;
      else if ( (ii1 == 1) && ( ii2 == 3) ) psictilde = -2.0f * cf * hat;
      else if ( (ii1 == 1) && ( ii2 == 2) ) psictilde = 2.0f * cf  * ( pow(n,2) -2.0f) * hat;
    } // if porcess == 0
  else if( process == 1)
    {
      hat = 2.0f * (pow(s,2) + pow(u,2)) /pow(t,2);
      hau = 2.0f * (pow(s,2) + pow(t,2)) /pow(u,2);
      hb  = 2.0f * pow(s,2) /t /u;
      if ( (ii1 == 0) && (ii2 == 1) )
	psictilde = 4.0f * cf /n
	  *( - hb + n*(hat + hau) - pow(n,2) * hb );
      else if ( (ii1 == 0) && (ii2 == 2) )
	psictilde = 4.0f * cf /n
	  *( hb - n*(hau + 0.5f*hat) + 0.5f * pow(n,3) * hau );
      else if ( (ii1 == 0) && (ii2 == 3) )
	psictilde = 4.0f * cf /n
	*( hb - n*(hat + 0.5f*hau) + 0.5f * pow(n,3) * hat );
      else if ( (ii1 == 2) && (ii2 == 3) ) 
	psictilde = 4.0f * cf /n
	*( - hb + n*(hat + hau) - pow(n,2) * hb );
      else if ( (ii1 == 1) && (ii2 == 3) ) 
	psictilde = 4.0f * cf /n
	*( hb - n*(hau + 0.5f*hat) + 0.5f * pow(n,3) * hau );
      else if ( (ii1 == 1) && (ii2 == 2) ) 
	psictilde = 4.0f * cf /n
	*( hb - n*(hat + 0.5f*hau) + 0.5f * pow(n,3) * hat );
    } //process == 1
  else if( process == 2)
    {
      hct = 2.0f * (pow(t,2) + pow(u,2)) /pow(s,2) * t/u;
      hcu = 2.0f * (pow(t,2) + pow(u,2)) /pow(s,2) * u/t;
      hcs = 2.0f * (pow(t,2) + pow(u,2)) /t /u;
      if     ( (ii1 == 0) && (ii2 == 1) ) 
	psictilde = v * ( - hct - hcu + hcs + hcs /pow(n,2) );
      else if ( (ii1 == 0) && (ii2 == 2) ) 
	psictilde = v * ( pow(n,2) * hcu - hcs );
      else if ( (ii1 == 0) && (ii2 == 3) ) 
	psictilde = v * ( pow(n,2) * hct - hcs );
      else if ( (ii1 == 2) && (ii2 == 3) ) 
	psictilde =  v * pow(n,2) * ( hct + hcu );
      else if ( (ii1 == 1) && (ii2 == 3) ) 
	psictilde = v * ( pow(n,2) * hcu - hcs );
      else if ( (ii1 == 1) && (ii2 == 2) ) 
	psictilde = v * ( pow(n,2) * hct - hcs );
    } // process == 2
  else if( process == 3)
    {
      hds = 2.0f * (pow(t,2) + pow(u,2)) /pow(s,2) /pow(t,2) /pow(u,2)
	* (pow(s,4) + pow(t,4) + pow(u,4));
      hdt = 2.0f * (pow(u,2) + pow(s,2)) /pow(s,2) /pow(t,2) /pow(u,2)
	* (pow(s,4) + pow(t,4) + pow(u,4));
      hdu = 2.0f * (pow(s,2) + pow(t,2)) /pow(s,2) /pow(t,2) /pow(u,2)
	* (pow(s,4) + pow(t,4) + pow(u,4));
      if     ( (ii1 == 0) && (ii2 == 1) ) 
	psictilde = 2.0f * v * pow(n,3) * hds;
      else if ( (ii1 == 0) && (ii2 == 2) ) 
	psictilde = 2.0f * v * pow(n,3) * hdt;
      else if ( (ii1 == 0) && (ii2 == 3) ) 
	psictilde = 2.0f * v * pow(n,3) * hdu;
      else if ( (ii1 == 2) && (ii2 == 3) ) 
	psictilde = 2.0f * v * pow(n,3) * hds;
      else if ( (ii1 == 1) && (ii2 == 3) ) 
	psictilde = 2.0f * v * pow(n,3) * hdt;
      else if ( (ii1 == 1) && (ii2 == 2) ) 
	psictilde = 2.0f * v * pow(n,3) * hdu;
    } // process == 3
  else psictilde = 0.0f;
  
  return psictilde;
}

/******************************************************************************/
/******************************************************************************/


__device__ void residueinit()
{
  

}


/******************************************************************************/
/******************************************************************************/
__device__ float gammaval(int nparton)
{
  /*
    C   This function gives the A-P delta function coefficient for
    C   parton NPARTON: 2 for quarks and antiquarks,
    C   11/2 - Nfl/3 for gluons
  */
  float gammaval;

  float nflavor;
  nflavor = eks_setting_on_device.nflavor;
  
  if(nparton == 0)
    gammaval = 11.0f/2.0f - nflavor/3.0f;
  else 
    gammaval = 2.0f;

  return gammaval;
  
}
/******************************************************************************/
/******************************************************************************/
__device__  float gammaprime(int nparton)
{
  /*
    C   This function gives the remainder from the collinear integration
    C   for the coll-1 and coll-2 contributions for parton NPARTON
  */
  float gammaprime;

  float nflavor = eks_setting_on_device.nflavor;
  float pi = 3.141592654f;
  float v = 8.0f;
  float n = 3.0f;
  
  if(nparton == 0)
    gammaprime = 67.0f * n/9.0f - 23.0f/18.0f * nflavor - 2.0f * pow(pi,2) * color(nparton) /3.0f;
  else 
    gammaprime = 13.0f * v /(4.0f*n) - 2.0f * pow(pi,2) * color(nparton) /3.0f;
  
  return gammaprime;
  
}
/******************************************************************************/
/******************************************************************************/

__device__ float color(int nparton)
{
  /*
    C   This function gives the color charge for
    C   parton NPARTON: 4/3 for quarks and antiquarks, 3 for gluons
   */
  float color;
  
  float ca = 3.0f;
  float cf = 4.0f/3.0f;
  
  if(nparton == 0) 
    color = ca;
  else 
    color = cf;
  
  return color;

}


/******************************************************************************/
/******************************************************************************/
__device__ void initialize( int* event_index_on_device, float* warehouse_on_device)
{
  // initialze the variables;
  int thread_index = blockIdx.x*blockDim.x + threadIdx.x;
  event_index_on_device[thread_index] = 0;
}

/******************************************************************************/
/******************************************************************************/
__device__ void finish()
{
  // destroy allocated memory and put
  
}

/******************************************************************************/
/******************************************************************************/


__device__ void report(int npartons, float p1, float y1, float phi1, float p2, float y2, float phi2, float p3, float y3, float phi3, float integrand, int* event_index_on_device, float* warehouse_on_device)
{
  /* 
     throw the integration output to a memory region that can be read from the host side
   */
  // RETURN if NPARTONS is zero.
  if(npartons == 0) return;
  
  // Do nothing if INTEGRAND is tiny.
  if( abs(integrand) < 1.0e-24f ) return;

  // in what position this event in the thread?
  int thread_id = blockDim.x*blockIdx.x + threadIdx.x;
  // how many events already recorded in this thread?
  int event_id = event_index_on_device[thread_id];
  // from what position we record in warehouse_on_device?
  int event_offset = thread_id*17*11 + event_id*11;
  
  //out put postion 
  //  int output_position = event_offset + event_id*11;
  int output_position = event_offset;
  
  warehouse_on_device[output_position] = npartons;
  warehouse_on_device[output_position + 1] = p1;
  warehouse_on_device[output_position + 2] = y1;
  warehouse_on_device[output_position + 3] = phi1;
  warehouse_on_device[output_position + 4] = p2;
  warehouse_on_device[output_position + 5] = y2;
  warehouse_on_device[output_position + 6] = phi2;
  warehouse_on_device[output_position + 7] = p3;
  warehouse_on_device[output_position + 8] = y3;
  warehouse_on_device[output_position + 9] = phi3;
  warehouse_on_device[output_position + 10] = integrand;
    
  event_index_on_device[thread_id] = event_index_on_device[thread_id]  + 1; 
}


/******************************************************************************/
/******************************************************************************/
