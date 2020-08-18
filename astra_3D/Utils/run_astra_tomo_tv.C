#include "astra_ctvlib.hpp"
#include <iostream>
using namespace std; 

int main(int argc, char **argv) {
  // Number of iteration
  int niter = 10;
  // iteration in the TV loop
  int ng = 10;
  // parameter in ART reconstruction
  float beta = 0.25;
  // ART reduction 
  float beta_red = 0.985;
  // Data Tolerrance Parameter
  float eps = 0.019;
  //Reduction Criteria
  float r_max = 0.95;
  float alpha_red = 0.95; 
  float alpha = 0.2;

  int SNR = 100;
  // Outcomes
  bool noise = true; 
  bool save_recon = true; 
  float dPOCS, dp;
  int i=0;
  char fname[255] = "Tilt_Series/256_au_sto.h5";
  char output[255] = "output/recon_256_au_sto.h5";

  astra_ctvlib tomo_obj(256, 256, 256);
  while(i<argc) {
    if (strcmp(argv[i], "-d")==0) {
      strcpy(fname, argv[i+1]);
      i = i+2;
    } else if (strcmp(argv[i], "-o")==0) {
      strcpy(output, argv[i+1]);
      i=i+2;
    } else if (strcmp(argv[i], "-n")==0) {
      niter = int(atof(argv[i+1]));
      i=i+2;
    } else {
      i=i+1;
    }
  }
  int verbose = tomo_obj.get_verbose();
  if (verbose==1) {
    cout << "# ===== Command Line Inputs" << endl; 
    cout << "* Data: " << fname << endl;
    cout << "* Output: " << output << endl;
    cout << "* Niter: " << niter << endl; 
  }

}
