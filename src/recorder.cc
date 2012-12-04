#include "recorder.h"

using namespace std;

recorder::recorder()
{
  p1 = 0.;
  p2 = 0.;
  p3 = 0.;

  y1 = 0.;
  y2 = 0.;
  y3 = 0.;

  phi1 = 0.;
  phi2 = 0.;
  phi3 = 0.;

  weight = 0.;
  nparton = 0.;
}

//========================================================================

recorder::~recorder()
{
  if(input_card.write_to_root_file) 
    {
      printnt();
      nt->Write();
      outfile->Write();
      outfile->Close();
    }
  if(input_card.write_to_log_txt) fs.close();
  // outfile->Close();
}

//========================================================================

void recorder::read_card(card& cd){
  input_card = cd;
}

//========================================================================
void recorder::init()
{
  //  if(input_card.write_to_root_file) std::cout<<input_card<<std::endl;
  if(input_card.write_to_log_txt) init_log();
  if(input_card.write_to_root_file) init_root();

  number_of_points = input_card.number_of_blocks * input_card.number_of_threads;
}
//========================================================================

//========================================================================

void recorder::record(int* event_index, float* warehouse, double* mc_weight, int* iter)
{
  int total_events = 0;
  for(int i = 0; i < number_of_points; i++)
    {
      total_events += event_index[i]; // WRONG CODE !!!!!
    }
  // j is the label for phase space points, and i is the label of events in each phace space points.
  if( input_card.write_to_root_file )
    {
      for(int i = 0; i < total_events; ++i)
	{
	  nparton = warehouse[ i*17 ];

	  p1 = warehouse[i*17 + 1];
	  p2 = warehouse[i*17 + 2];
	  p3 = warehouse[i*17 + 3];

	  y1 = warehouse[i*17 + 4];
	  y2 = warehouse[i*17 + 5];
	  y3 = warehouse[i*17 + 6];

	  phi1 = warehouse[i*17 + 7];
	  phi2 = warehouse[i*17 + 8];
	  phi3 = warehouse[i*17 + 9];
	  
	  weight = warehouse[i*17 + 10]* (*mc_weight); //multiple by the weight from monte carlo generator
	  // weight = (*mc_weight); //multiple by the weight from monte carlo generator

	  // if(weight > 10e5 ) cout<<" weight is " << weight <<endl;
	  // if( *mc_weight > 10e5 ) cout<<" mc_weight is " << weight <<endl;

	  nt->Fill();
	  
	  //interation number
	  iteration = *iter;
	}// int i
    } // if write to root
}

//========================================================================

void recorder::init_root( )
{
  /*
    initiate root file 
  */
  
  /*
    Initialize the ROOT file for a given access mode 
    Inputs: title -- the name of the file (e.g., 'resbos.root')
    access -- a string specifying the type of the access
    (equivalent to the string 'option' in the TFile constructor)
    If access = 'NEW' or 'CREATE'   create a new file and open it 
    for writing. If the file already exists 
    the file is not opened.
    = 'RECREATE'      create a new file. If the file already exists, it will be overwritten.
    = 'UPDATE'        open an existing file for writing. If no file exists, it is created.
    = 'READ'          open an existing file for reading (default).
  */
  
  string filename ="output/"+input_card.root_file_name;
  cout << "opening root file: " << filename << endl;
  outfile = new TFile(filename.c_str(),"RECREATE");
  //cout<<"is zombie? " << outfile->IsZombie() << ", is open? " << outfile->IsOpen() <<endl;
  outfile->cd();


  outfile->GetObject("h10",nt);
  if (nt == 0x0 ) {
    nt=new TTree("h10","h10");
  }

  //outfile->cd();

  header = (TTree*)outfile->Get("header");
  if (header == 0x0 ) 
    header=new TTree("header","header");
 
  header->SetAutoSave(0);

  header->Branch("rts",&(input_card.square_s), "rts/D",10);
  header->Branch("ppbar",&(input_card.ppbar_collider), "ppbar/O",10);
  header->Branch("iset",&(input_card.pdf_set_number), "iset/I",10);
  header->Branch("muuvoverpj",&(input_card.renomalization_scale), "muuvoverpj/D",10);
  header->Branch("mucooverpj",&(input_card.factorization_scale), "mucooverpj/D",10);
  header->Branch("mc_engine",&(input_card.mc_engine_type), "mc_engine/D",10);
  header->Branch("seed",&(input_card.mc_seed), "seed/I",10);
  header->Branch("nreno",&(input_card.data_point_numbers), "nreno/I",10);
  header->Branch("pjmin",&(input_card.pt_min), "pjmin/D",10);
  header->Branch("pjmax",&(input_card.pt_max), "pjmax/D",10);
  header->Branch("yjmin",&(input_card.y_min), "yjmin/D",10);
  header->Branch("yjmax",&(input_card.y_max), "yjmax/D",10);
  header->Branch("iteration",&iteration, "iteration/I",10);

  nt->Branch("NPARTON",nparton,"NPARTON/I");

  nt->Branch("P1",&p1,"P1/F");
  nt->Branch("P2",&p2,"P2/F");
  nt->Branch("P3",&p3,"P3/F");
  
  nt->Branch("Y1",&y1,"Y1/F");
  nt->Branch("Y2",&y2,"Y2/F");
  nt->Branch("Y3",&y3,"Y3/F");
  
  nt->Branch("PHI1",&phi1,"PHI1/F");
  nt->Branch("PHI2",&phi2,"PHI2/F");
  nt->Branch("PHI3",&phi3,"PHI3/F");
  
  nt->Branch("WEIGHT",&weight,"WEIGHT/F");

   // for(int i = 0;  i < 10000; ++i)
   //   {

   //     p1 = i;
   //     p2 = i;
   //     p3 = i;
  
   //     y1 = i;
   //     y2 = i;
   //     y3 = i;
  
   //     phi1 = i;
   //     phi2 = i;
   //     phi3 = i;
  
   //     weight = i;
   //     nparton = i;
   //     nt->Fill();
   //   }

}
//========================================================================

void recorder::init_log()
{
  /*
    initiate log file
  */
  fs.open(input_card.log_file_name.c_str());
}

//========================================================================


/*
  Print the list of branches in the ntuple
 */
void recorder::printnt()
{
  if(input_card.write_to_root_file){
    std::cout << std::endl << "Structure of the ROOT header:" << std::endl;
    header->Print("toponly");
    std::cout << std::endl << "Structure of the ROOT ntuple:" << std::endl;
    nt->Print();
  }
}//printnt

//========================================================================

recorder& recorder::operator=(recorder& rhs)
{
  return rhs;
}

//========================================================================

void recorder::fill_header()
{
  header->Fill();
}

//========================================================================
