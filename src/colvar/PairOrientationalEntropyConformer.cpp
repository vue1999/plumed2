/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2011-2016 The plumed team
   (see the PEOPLE file at the root of the distribution for a list of names)

   See http://www.plumed.org for more information.

   This file is part of plumed, version 2.

   plumed is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   plumed is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with plumed.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
#include "Colvar.h"
#include "ActionRegister.h"
#include "tools/NeighborListParallel.h"
#include "tools/Communicator.h"
#include "tools/Tools.h"
#include "tools/IFile.h"

#include <string>
#include <math.h>

using namespace std;

namespace PLMD{
namespace colvar{

class PairOrientationalEntropyConformer : public Colvar {
  bool pbc, serial, invalidateList, firsttime, doneigh;
  bool do_pairs;
  NeighborListParallel *nl;
  vector<AtomNumber> center_lista,start_lista,end_lista;
  vector<AtomNumber> vector1start_lista,vector1end_lista;
  vector<AtomNumber> vector2start_lista,vector2end_lista;
  vector<AtomNumber> vector3start_lista,vector3end_lista;
  std::vector<PLMD::AtomNumber> atomsToRequest;
  double maxr;
  vector<unsigned> nhist_;
  unsigned nhist1_nhist2_;
  vector<double> sigma_;
  double rcut2;
  double invTwoPiSigma1Sigma2, sigma1Sqr, sigma2Sqr, twoSigma1Sqr,twoSigma2Sqr;
  double deltar, deltaAngle, deltaCosAngle;
  unsigned deltaBin, deltaBinAngle;
  // Integration routines
  double integrate(Matrix<double> integrand, vector<double> delta)const;
  Vector integrate(Matrix<Vector> integrand, vector<double> delta)const;
  Tensor integrate(Matrix<Tensor> integrand, vector<double> delta)const;
  vector<double> x1, x2, x1sqr, x2sqr;
  // Kernel to calculate g(r)
  double kernel(vector<double> distance, double invNormKernel, vector<double>&der)const;
  // Output gofr and integrand
  void outputGofr(Matrix<double> gofrAA, Matrix<double> gofrBB, Matrix<double> gofrAB);
  int outputStride;
  bool doOutputGofr, doOutputIntegrand;
  mutable PLMD::OFile gofrOfile;
  // Reference g(r)
  bool doReferenceGofr;
  Matrix<double> referenceGofr;
  double epsilon;
  // Average gofr
  Matrix<double> avgGofrAA;
  Matrix<double> avgGofrBB;
  Matrix<double> avgGofrAB;
  unsigned iteration;
  bool doAverageGofr;
  unsigned averageGofrTau;
  // Up-down symmetry
  bool doUpDownSymmetry;
  double startCosAngle;
  // Output spins
  bool doOutputXYZ;
  mutable PLMD::OFile outputFileXYZ;
  void outputXYZ(std::vector<double>);
  // One body
  bool one_body;
  double deBroglie3;
  // Periodic images
  int periodic_images;
public:
  explicit PairOrientationalEntropyConformer(const ActionOptions&);
  ~PairOrientationalEntropyConformer();
  virtual void calculate();
  virtual void prepare();
  static void registerKeywords( Keywords& keys );
};

PLUMED_REGISTER_ACTION(PairOrientationalEntropyConformer,"PAIR_ORIENTATIONAL_ENTROPY_CONFORMER")

void PairOrientationalEntropyConformer::registerKeywords( Keywords& keys ){
  Colvar::registerKeywords(keys);
  keys.addFlag("SERIAL",false,"Perform the calculation in serial - for debug purpose");
  keys.addFlag("NLIST",false,"Use a neighbour list to speed up the calculation");
  keys.addFlag("OUTPUT_GOFR",false,"Output g(r)");
  keys.addFlag("AVERAGE_GOFR",false,"Average g(r) over time");
  keys.addFlag("INDIVIDUAL_PAIRS",false,"Obtain pair entropy of AA, AB, and BB pairs");
  keys.add("optional","AVERAGE_GOFR_TAU","Characteristic length of a window in which to average the g(r). It is in units of iterations and should be an integer. Zero corresponds to an normal average (infinite window).");
  keys.addFlag("UP_DOWN_SYMMETRY",false,"The symmetry is such that parallel and antiparallel vectors are not distinguished. The angle goes from 0 to pi/2 instead of from 0 to pi.");
  keys.add("optional","OUTPUT_STRIDE","The frequency with which the output is written to files");
  keys.addFlag("OUTPUT_INTEGRAND",false,"Output integrand");
  keys.add("optional","NL_CUTOFF","The cutoff for the neighbour list");
  keys.add("optional","NL_STRIDE","The frequency with which we are updating the atoms in the neighbour list");
  keys.add("atoms","ORIGIN","Define an atom that represents the origin from which to calculate the g(r,theta)");
  keys.add("atoms","CENTER","Center atoms");
  keys.add("atoms","START","Start point of vector defining orientation");
  keys.add("atoms","END","End point of vector defining orientation");
  keys.add("compulsory","MAXR","1","Maximum distance for the radial distribution function ");
  keys.add("optional","NHIST","Number of bins in the rdf ");
  keys.add("compulsory","SIGMA","0.1","Width of gaussians ");
  keys.add("optional","REFERENCE_GOFR_FNAME","the name of the file with the reference g(r)");
  keys.addFlag("OUTPUT_XYZ",false,"Output xyz file with the spins");
  keys.add("atoms","VECTOR1START","Start point of first vector");
  keys.add("atoms","VECTOR1END"  ,"End point of first vector");
  keys.add("atoms","VECTOR2START","Start point of second vector");
  keys.add("atoms","VECTOR2END"  ,"End point of second vector");
  keys.add("atoms","VECTOR3START","Start point of third vector");
  keys.add("atoms","VECTOR3END"  ,"End point of third vector");
  keys.add("optional","TEMPERATURE","Temperature in Kelvin. It is compulsory when keyword ONE_BODY is used");
  keys.add("optional","PERIODIC_IMAGES","Number of periodic images to consider in the calculation of g(r). This could be done automatically in the future.");
  keys.add("optional","MASS","Mass in g/mol. It is compulsory when keyword ONE_BODY is used");
  keys.addFlag("ONE_BODY",false,"Add the one body term (S = 5/2 - ln(dens*deBroglie^3) ) to the entropy");
  keys.addOutputComponent("pairAA","INDIVIDUAL_PAIRS","Pair AA contribution to the multicomponent pair entropy");
  keys.addOutputComponent("pairAB","INDIVIDUAL_PAIRS","Pair AB contribution to the multicomponent pair entropy");
  keys.addOutputComponent("pairBB","INDIVIDUAL_PAIRS","Pair BB contribution to the multicomponent pair entropy");
  keys.addOutputComponent("full","INDIVIDUAL_PAIRS","Total multicomponent pair entropy");
}

PairOrientationalEntropyConformer::PairOrientationalEntropyConformer(const ActionOptions&ao):
PLUMED_COLVAR_INIT(ao),
pbc(true),
serial(false),
do_pairs(false),
invalidateList(true),
firsttime(true)
{

  parseFlag("SERIAL",serial);

  parseAtomList("CENTER",center_lista);
  parseAtomList("START",start_lista);
  parseAtomList("END",end_lista);
  if(center_lista.size()!=start_lista.size()) error("Number of atoms in START must be equal to the number of atoms in CENTER");
  if(center_lista.size()!=end_lista.size()) error("Number of atoms in START must be equal to the number of atoms in CENTER");

  parseAtomList("VECTOR1START",vector1start_lista);
  parseAtomList("VECTOR1END",vector1end_lista);
  parseAtomList("VECTOR2START",vector2start_lista);
  parseAtomList("VECTOR2END",vector2end_lista);
  parseAtomList("VECTOR3START",vector3start_lista);
  parseAtomList("VECTOR3END",vector3end_lista);

  bool nopbc=!pbc;
  pbc=!nopbc;

// neighbor list stuff
  doneigh=false;
  bool nl_full_list=true;
  double nl_cut=0.0;
  double nl_skin;
  int nl_st=-1;
  parseFlag("NLIST",doneigh);
  if(doneigh){
   parse("NL_CUTOFF",nl_cut);
   if(nl_cut<=0.0) error("NL_CUTOFF should be explicitly specified and positive");
   parse("NL_STRIDE",nl_st);
   //if(nl_st<=0) error("NL_STRIDE should be explicitly specified and positive");
  }

  if(pbc) log.printf("  using periodic boundary conditions\n");
  else    log.printf("  without periodic boundary conditions\n");

  parse("MAXR",maxr);
  log.printf("  Integration in the interval from 0. to %f \n", maxr );

  parseVector("SIGMA",sigma_);
  if(sigma_.size() != 2) error("SIGMA keyword takes two input values");
  log.printf("  The pair distribution function is calculated with a Gaussian kernel with deviations %f and %f \n", sigma_[0], sigma_[1]);
  double rcut = maxr + 2*sigma_[0];  // 3*sigma is hard coded
  rcut2 = rcut*rcut;
  if(doneigh){
    if(nl_cut<rcut) error("NL_CUTOFF should be larger than MAXR + 2*SIGMA");
    nl_skin=nl_cut-rcut;
  }

  doUpDownSymmetry=false;
  parseFlag("UP_DOWN_SYMMETRY",doUpDownSymmetry);
  if (doUpDownSymmetry) log.printf("  The angle can take values between 0 and pi/2 due to the up down symmetry. \n");

  parseVector("NHIST",nhist_);
  if (nhist_.size()<1) {
     nhist_.resize(2);
     // Default values
     nhist_[0]=ceil(maxr/sigma_[0]) + 1; 
     if (doUpDownSymmetry) nhist_[1]=ceil(1./sigma_[1]) + 1;
     else nhist_[1]=ceil(2./sigma_[1]) + 1;
  }
  if(nhist_.size() != 2) error("NHIST keyword takes two input values");
  nhist1_nhist2_=nhist_[0]*nhist_[1];
  log.printf("  The r-theta space is discretized using a grid of size %u times %u. \n", nhist_[0], nhist_[1] );
  log.printf("  The integration is performed with the trapezoid rule. \n");

  doOutputGofr=false;
  parseFlag("OUTPUT_GOFR",doOutputGofr);
  if (doOutputGofr) { 
     log.printf("  The g(r) will be written to a file \n");
     gofrOfile.link(*this);
     gofrOfile.open("gofr.txt");
  }
  doOutputIntegrand=false;
  parseFlag("OUTPUT_INTEGRAND",doOutputIntegrand);
  if (doOutputIntegrand) {
     log.printf("  The integrand will be written to a file \n");
  }
  outputStride=1;
  parse("OUTPUT_STRIDE",outputStride);
  if (outputStride!=1 && !doOutputGofr && !doOutputIntegrand) error("Cannot specify OUTPUT_STRIDE if OUTPUT_GOFR or OUTPUT_INTEGRAND not used");
  if (outputStride<1) error("The output stride specified with OUTPUT_STRIDE must be greater than or equal to one.");
  if (outputStride>1) log.printf("  The output stride to write g(r) or the integrand is %d \n", outputStride);

  doReferenceGofr=false;
  std::string referenceGofrFileName;
  parse("REFERENCE_GOFR_FNAME",referenceGofrFileName); 
  if (!referenceGofrFileName.empty() ) {
    epsilon=1.e-8;
    log.printf("  Reading a reference g(r) from the file %s . \n", referenceGofrFileName.c_str() );
    doReferenceGofr=true;
    IFile ifile; 
    ifile.link(*this);
    ifile.open(referenceGofrFileName);
    referenceGofr.resize(nhist_[0],nhist_[1]);
    for(unsigned int i=0;i<nhist_[0];i++) {
       for(unsigned int j=0;j<nhist_[1];j++) {
       double tmp_r, tmp_theta;
       ifile.scanField("r",tmp_r).scanField("theta",tmp_theta).scanField("gofr",referenceGofr[i][j]).scanField();
       }
    }
  }

  doAverageGofr=false;
  parseFlag("AVERAGE_GOFR",doAverageGofr);
  if (doAverageGofr) {
     iteration = 1;
     avgGofrAA.resize(nhist_[0],nhist_[1]);
     avgGofrBB.resize(nhist_[0],nhist_[1]);
     avgGofrAB.resize(nhist_[0],nhist_[1]);
  }
  averageGofrTau=0;
  parse("AVERAGE_GOFR_TAU",averageGofrTau);
  if (averageGofrTau!=0 && !doAverageGofr) error("AVERAGE_GOFR_TAU specified but AVERAGE_GOFR not given. Specify AVERAGE_GOFR or remove AVERAGE_GOFR_TAU");
  if (doAverageGofr && averageGofrTau==0) log.printf("The g(r) will be averaged over all frames \n");
  if (doAverageGofr && averageGofrTau!=0) log.printf("The g(r) will be averaged with a window of %d steps \n", averageGofrTau);

  parseFlag("INDIVIDUAL_PAIRS",do_pairs);
  if (do_pairs) log.printf("  The AA, AB, and BB contributions will be computed separately \n");

  doOutputXYZ=false;
  parseFlag("OUTPUT_XYZ",doOutputXYZ);
  if (doOutputXYZ) { 
     log.printf("  An xyz file with the spin will be written \n");
     outputFileXYZ.link(*this);
     outputFileXYZ.open("spin.xyz");
  }

  parseFlag("ONE_BODY",one_body);
  double temperature = -1.;
  double mass = -1.;
  parse("TEMPERATURE",temperature);
  parse("MASS",mass);
  if (one_body) {
     if (temperature>0 && mass>0) log.printf("The one-body entropy will be added to the pair entropy. \n");
     if (temperature<0) error("ONE_BODY keyword used but TEMPERATURE not given. Specify a temperature greater than 0 in Kelvin using the TEMPERATURE keyword. ");
     if (mass<0) error("ONE_BODY keyword used but MASS not given. Specify a mass greater than 0 in g/mol using the MASS keyword. ");
     double planck = 6.62607004e-16; // nm2 kg / s 
     double boltzmann = 1.38064852e-5; // nm2 kg s-2 K-1
     double avogadro= 6.0221409e23 ;
     double deBroglie = planck/std::sqrt(2*pi*(mass*1.e-3/avogadro)*boltzmann*temperature);
     deBroglie3 = deBroglie*deBroglie*deBroglie;
     log.printf("The thermal deBroglie wavelength is %f nm. Be sure to use nm as units of distance. \n", deBroglie);
  }

  periodic_images=0;
  parse("PERIODIC_IMAGES",periodic_images);
  if (periodic_images>0) log.printf("%d periodic images are considered \n", periodic_images);

  checkRead();

  // Neighbor lists
  if (doneigh) {
    nl= new NeighborListParallel(center_lista,pbc,getPbc(),comm,log,nl_cut,nl_full_list,nl_st,nl_skin);
    log.printf("  using neighbor lists with\n");
    log.printf("  cutoff %f, and skin %f\n",nl_cut,nl_skin);
    if(nl_st>=0){
      log.printf("  update every %d steps\n",nl_st);
    } else {
      log.printf("  checking every step for dangerous builds and rebuilding as needed\n");
    }
  }
  atomsToRequest.reserve ( 9*center_lista.size() );
  atomsToRequest.insert (atomsToRequest.end(), center_lista.begin(), center_lista.end() );
  atomsToRequest.insert (atomsToRequest.end(), start_lista.begin(), start_lista.end() );
  atomsToRequest.insert (atomsToRequest.end(), end_lista.begin(), end_lista.end() );
  atomsToRequest.insert (atomsToRequest.end(), vector1start_lista.begin(), vector1start_lista.end() );
  atomsToRequest.insert (atomsToRequest.end(), vector1end_lista.begin()  , vector1end_lista.end() );
  atomsToRequest.insert (atomsToRequest.end(), vector2start_lista.begin(), vector2start_lista.end() );
  atomsToRequest.insert (atomsToRequest.end(), vector2end_lista.begin()  , vector2end_lista.end() );
  atomsToRequest.insert (atomsToRequest.end(), vector3start_lista.begin(), vector3start_lista.end() );
  atomsToRequest.insert (atomsToRequest.end(), vector3end_lista.begin()  , vector3end_lista.end() );
  requestAtoms(atomsToRequest);

  // Define heavily used expressions
  invTwoPiSigma1Sigma2 = (1./(2.*pi*sigma_[0]*sigma_[1]));
  sigma1Sqr = sigma_[0]*sigma_[0];
  sigma2Sqr = sigma_[1]*sigma_[1];
  twoSigma1Sqr = 2*sigma_[0]*sigma_[0];
  twoSigma2Sqr = 2*sigma_[1]*sigma_[1];
  deltar=maxr/(nhist_[0]-1);
  if (!doUpDownSymmetry) {
     deltaCosAngle=2./(nhist_[1]-1);
     startCosAngle=-1.;
  }
  else {
     deltaCosAngle=1./(nhist_[1]-1);
     startCosAngle=0.;
  }
  deltaBin = std::floor(2*sigma_[0]/deltar); // 2*sigma is hard coded
  deltaBinAngle = std::floor(2*sigma_[1]/deltaCosAngle); // 2*sigma is hard coded

  x1.resize(nhist_[0]);
  x1sqr.resize(nhist_[0]);
  x2.resize(nhist_[1]);
  x2sqr.resize(nhist_[1]);
  for(unsigned i=0;i<nhist_[0];++i){
     x1[i]=deltar*i;
     x1sqr[i]=x1[i]*x1[i];
  }
  for(unsigned i=0;i<nhist_[1];++i){
     x2[i]=startCosAngle+deltaCosAngle*i;
     x2sqr[i]=x2[i]*x2[i];
  }


  // Define output components
  if (do_pairs) {
    addComponentWithDerivatives("pairAA"); componentIsNotPeriodic("pairAA");
    addComponentWithDerivatives("pairAB"); componentIsNotPeriodic("pairAB");
    addComponentWithDerivatives("pairBB"); componentIsNotPeriodic("pairBB");
    addComponentWithDerivatives("full"); componentIsNotPeriodic("full");
  } else {
    addValueWithDerivatives(); setNotPeriodic();
  }
}

PairOrientationalEntropyConformer::~PairOrientationalEntropyConformer(){
  if (doneigh) {
     nl->printStats();
     delete nl;
  }
  if (doOutputGofr) gofrOfile.close();
}

void PairOrientationalEntropyConformer::prepare(){
  if(doneigh && nl->getStride()>0){
    if(firsttime) {
      invalidateList=true;
      firsttime=false;
    } else if ( (nl->getStride()>=0) &&  (getStep()%nl->getStride()==0) ){
      invalidateList=true;
    } else if ( (nl->getStride()<0) && !(nl->isListStillGood(getPositions())) ){
      invalidateList=true;
    } else {
      invalidateList=false;
    }
  }
}

// calculator
void PairOrientationalEntropyConformer::calculate()
{
  // Calculate "spin" i.e. conformer
  int number_molecules = center_lista.size();
  std::vector<Vector> deriv_spin(6*number_molecules);
  std::vector<double> spin(number_molecules);
  double number_A_molecules = 0.;
  std::vector<Vector> vectors1(number_molecules), vectors2(number_molecules), vectors3(number_molecules);
  for(int i=0;i<number_molecules;i+=1) {
    Vector vector1=pbcDistance(getPosition(i+3*number_molecules),getPosition(i+4*number_molecules));
    Vector vector2=pbcDistance(getPosition(i+5*number_molecules),getPosition(i+6*number_molecules));
    Vector vector3=pbcDistance(getPosition(i+7*number_molecules),getPosition(i+8*number_molecules));
    vectors1[i] = vector1;
    vectors2[i] = vector2;
    vectors3[i] = vector3;
    double norm_v1 = std::sqrt(vector1[0]*vector1[0]+vector1[1]*vector1[1]+vector1[2]*vector1[2]);
    double norm_v2 = std::sqrt(vector2[0]*vector2[0]+vector2[1]*vector2[1]+vector2[2]*vector2[2]);
    double norm_v3 = std::sqrt(vector3[0]*vector3[0]+vector3[1]*vector3[1]+vector3[2]*vector3[2]);
    double inv_norm_v1 = 1. / norm_v1;
    double inv_norm_v2 = 1. / norm_v2;
    double inv_norm_v3 = 1. / norm_v3;
    vector1 *= inv_norm_v1;
    vector2 *= inv_norm_v2;
    vector3 *= inv_norm_v3;
    Vector vector4=crossProduct(vector1,vector2); 
    spin[i] = (dotProduct(vector3,vector4)+1)/2.;
    number_A_molecules += spin[i];
    Vector deriv_vector1_x = inv_norm_v1*Vector(1,0,0) - vector1*vector1[0]*inv_norm_v1;
    Vector deriv_vector1_y = inv_norm_v1*Vector(0,1,0) - vector1*vector1[1]*inv_norm_v1;
    Vector deriv_vector1_z = inv_norm_v1*Vector(0,0,1) - vector1*vector1[2]*inv_norm_v1;
    Vector deriv_vector2_x = inv_norm_v2*Vector(1,0,0) - vector2*vector2[0]*inv_norm_v2;
    Vector deriv_vector2_y = inv_norm_v2*Vector(0,1,0) - vector2*vector2[1]*inv_norm_v2;
    Vector deriv_vector2_z = inv_norm_v2*Vector(0,0,1) - vector2*vector2[2]*inv_norm_v2;
    Vector deriv_vector3_x = inv_norm_v3*Vector(1,0,0) - vector3*vector3[0]*inv_norm_v3;
    Vector deriv_vector3_y = inv_norm_v3*Vector(0,1,0) - vector3*vector3[1]*inv_norm_v3;
    Vector deriv_vector3_z = inv_norm_v3*Vector(0,0,1) - vector3*vector3[2]*inv_norm_v3;
    double deriv_spin_vector1_x = dotProduct(crossProduct(deriv_vector1_x,vector2),vector3)/2.;
    double deriv_spin_vector1_y = dotProduct(crossProduct(deriv_vector1_y,vector2),vector3)/2.;
    double deriv_spin_vector1_z = dotProduct(crossProduct(deriv_vector1_z,vector2),vector3)/2.;
    deriv_spin[i+0*number_molecules][0] = -deriv_spin_vector1_x;
    deriv_spin[i+0*number_molecules][1] = -deriv_spin_vector1_y;
    deriv_spin[i+0*number_molecules][2] = -deriv_spin_vector1_z;
    deriv_spin[i+1*number_molecules][0] =  deriv_spin_vector1_x;
    deriv_spin[i+1*number_molecules][1] =  deriv_spin_vector1_y;
    deriv_spin[i+1*number_molecules][2] =  deriv_spin_vector1_z;
    double deriv_spin_vector2_x = dotProduct(crossProduct(vector1,deriv_vector2_x),vector3)/2.;
    double deriv_spin_vector2_y = dotProduct(crossProduct(vector1,deriv_vector2_y),vector3)/2.;
    double deriv_spin_vector2_z = dotProduct(crossProduct(vector1,deriv_vector2_z),vector3)/2.;
    deriv_spin[i+2*number_molecules][0] = -deriv_spin_vector2_x;
    deriv_spin[i+2*number_molecules][1] = -deriv_spin_vector2_y;
    deriv_spin[i+2*number_molecules][2] = -deriv_spin_vector2_z;
    deriv_spin[i+3*number_molecules][0] =  deriv_spin_vector2_x;
    deriv_spin[i+3*number_molecules][1] =  deriv_spin_vector2_y;
    deriv_spin[i+3*number_molecules][2] =  deriv_spin_vector2_z;
    double deriv_spin_vector3_x = dotProduct(vector4,deriv_vector3_x)/2.;
    double deriv_spin_vector3_y = dotProduct(vector4,deriv_vector3_y)/2.;
    double deriv_spin_vector3_z = dotProduct(vector4,deriv_vector3_z)/2.;
    deriv_spin[i+4*number_molecules][0] = -deriv_spin_vector3_x;
    deriv_spin[i+4*number_molecules][1] = -deriv_spin_vector3_y;
    deriv_spin[i+4*number_molecules][2] = -deriv_spin_vector3_z;
    deriv_spin[i+5*number_molecules][0] =  deriv_spin_vector3_x;
    deriv_spin[i+5*number_molecules][1] =  deriv_spin_vector3_y;
    deriv_spin[i+5*number_molecules][2] =  deriv_spin_vector3_z;
    /*
    log.printf("Deriv1 x %f y %f z %f \n", deriv_spin_vector1_x, deriv_spin_vector2_y, deriv_spin_vector3_z);
    log.printf("Deriv2 x %f y %f z %f \n", deriv_spin_vector1_x, deriv_spin_vector2_y, deriv_spin_vector3_z);
    log.printf("Deriv3 x %f y %f z %f \n", deriv_spin_vector1_x, deriv_spin_vector2_y, deriv_spin_vector3_z);
    */
  }
  if (doOutputXYZ) { 
    outputXYZ(spin);
  }

  Matrix<double> gofrAA(nhist_[0],nhist_[1]);
  Matrix<double> gofrBB(nhist_[0],nhist_[1]);
  Matrix<double> gofrAB(nhist_[0],nhist_[1]);
  vector<Vector> gofrPrimeCenterAA(nhist_[0]*nhist_[1]*center_lista.size());
  vector<Vector> gofrPrimeCenterBB(nhist_[0]*nhist_[1]*center_lista.size());
  vector<Vector> gofrPrimeCenterAB(nhist_[0]*nhist_[1]*center_lista.size());
  vector<Vector> gofrPrimeStartAA(nhist_[0]*nhist_[1]*center_lista.size());
  vector<Vector> gofrPrimeStartBB(nhist_[0]*nhist_[1]*center_lista.size());
  vector<Vector> gofrPrimeStartAB(nhist_[0]*nhist_[1]*center_lista.size());
  vector<Vector> gofrPrimeEndAA(nhist_[0]*nhist_[1]*center_lista.size());
  vector<Vector> gofrPrimeEndBB(nhist_[0]*nhist_[1]*center_lista.size());
  vector<Vector> gofrPrimeEndAB(nhist_[0]*nhist_[1]*center_lista.size());
  vector<Vector> gofrPrimeVector1StartAA(nhist_[0]*nhist_[1]*center_lista.size());
  vector<Vector> gofrPrimeVector2StartAA(nhist_[0]*nhist_[1]*center_lista.size());
  vector<Vector> gofrPrimeVector3StartAA(nhist_[0]*nhist_[1]*center_lista.size());
  vector<Vector> gofrPrimeVector1StartBB(nhist_[0]*nhist_[1]*center_lista.size());
  vector<Vector> gofrPrimeVector2StartBB(nhist_[0]*nhist_[1]*center_lista.size());
  vector<Vector> gofrPrimeVector3StartBB(nhist_[0]*nhist_[1]*center_lista.size());
  vector<Vector> gofrPrimeVector1StartAB(nhist_[0]*nhist_[1]*center_lista.size());
  vector<Vector> gofrPrimeVector2StartAB(nhist_[0]*nhist_[1]*center_lista.size());
  vector<Vector> gofrPrimeVector3StartAB(nhist_[0]*nhist_[1]*center_lista.size());
  Matrix<Tensor> gofrVirialAA(nhist_[0],nhist_[1]);
  Matrix<Tensor> gofrVirialBB(nhist_[0],nhist_[1]);
  Matrix<Tensor> gofrVirialAB(nhist_[0],nhist_[1]);
  // Calculate volume and density
  double number_B_molecules = number_molecules-number_A_molecules;
  double volume=getBox().determinant();
  double density=number_molecules/volume;
  double densityA=number_A_molecules/volume;
  double densityB=number_B_molecules/volume;
  // Normalization of g(r)
  double normConstantBaseAA = 2*pi*number_A_molecules*densityA;
  double normConstantBaseBB = 2*pi*number_B_molecules*densityB;
  double normConstantBaseAB = 2*pi*number_A_molecules*densityB;
  normConstantBaseAA /= invTwoPiSigma1Sigma2;
  normConstantBaseBB /= invTwoPiSigma1Sigma2;
  normConstantBaseAB /= invTwoPiSigma1Sigma2;
  // Take into account "volume" of angles
  double volumeOfAngles;
  if (!doUpDownSymmetry) volumeOfAngles = 2.;
  else volumeOfAngles = 1.;
  normConstantBaseAA /= volumeOfAngles;
  normConstantBaseBB /= volumeOfAngles;
  normConstantBaseAB /= volumeOfAngles;
  double invNormConstantBaseAA = 1./normConstantBaseAA;
  double invNormConstantBaseBB = 1./normConstantBaseBB;
  double invNormConstantBaseAB = 1./normConstantBaseAB;
  // Box vectors
  Vector BoxVector1(getBox()[0][0],getBox()[0][1],getBox()[0][2]);
  Vector BoxVector2(getBox()[1][0],getBox()[1][1],getBox()[1][2]);
  Vector BoxVector3(getBox()[2][0],getBox()[2][1],getBox()[2][2]);
  // Setup parallelization
  unsigned stride=comm.Get_size();
  unsigned rank=comm.Get_rank();
  if(serial){
    stride=1;
    rank=0;
  }else{
    stride=comm.Get_size();
    rank=comm.Get_rank();
  }
  if (doneigh) {
    if(invalidateList){
      vector<Vector> centerPositions(getPositions().begin(),getPositions().begin() + center_lista.size());
      nl->update(centerPositions);
    }
    for(unsigned int i=0;i<nl->getNumberOfLocalAtoms();i+=1) {
       unsigned index=nl->getIndexOfLocalAtom(i);
       unsigned atom1_mol1=index+center_lista.size();
       unsigned atom2_mol1=index+center_lista.size()+start_lista.size();
       std::vector<unsigned> neighbors=nl->getNeighbors(index);
       Vector position_index=getPosition(index);
       Vector mol_vector1=pbcDistance(getPosition(atom1_mol1),getPosition(atom2_mol1)); 
       double norm_v1 = std::sqrt(mol_vector1[0]*mol_vector1[0]+mol_vector1[1]*mol_vector1[1]+mol_vector1[2]*mol_vector1[2]);
       double inv_v1=1./norm_v1;
       double inv_v1_sqr=inv_v1*inv_v1;
       // Loop over neighbors
       for(unsigned int j=0;j<neighbors.size();j+=1) {  
          unsigned neighbor=neighbors[j];
          Vector distanceMinimumImage;
          if(pbc){
           distanceMinimumImage=pbcDistance(position_index,getPosition(neighbor));
          } else {
           distanceMinimumImage=delta(position_index,getPosition(neighbor));
          }
          for (int l=-periodic_images; l<(periodic_images+1); l++) {
            for (int m=-periodic_images; m<(periodic_images+1); m++) {
              for (int n=-periodic_images; n<(periodic_images+1); n++) {
                // if same atom and same image
                if(getAbsoluteIndex(index)==getAbsoluteIndex(neighbor) && l==0 && m==0 && n==0) continue;
                Vector distance=distanceMinimumImage;
                distance += l*BoxVector1;
                distance += m*BoxVector2;
                distance += n*BoxVector3;
                double d2;
                if ( (d2=distance[0]*distance[0])<rcut2 && (d2+=distance[1]*distance[1])<rcut2 && (d2+=distance[2]*distance[2])<rcut2) {
                   double distanceModulo=std::sqrt(d2);
                   Vector distance_versor = distance / distanceModulo;
                   unsigned bin=std::floor(distanceModulo/deltar);
                   unsigned atom1_mol2=neighbor+center_lista.size();
                   unsigned atom2_mol2=neighbor+center_lista.size()+start_lista.size();
                   Vector mol_vector2=pbcDistance(getPosition(atom1_mol2),getPosition(atom2_mol2));
                   double norm_v2 = std::sqrt(mol_vector2[0]*mol_vector2[0]+mol_vector2[1]*mol_vector2[1]+mol_vector2[2]*mol_vector2[2]);
                   double inv_v2=1./norm_v2;
                   double inv_v1_inv_v2=inv_v1*inv_v2;
                   double cosAngle=dotProduct(mol_vector1,mol_vector2)*inv_v1*inv_v2;
                   Vector der_mol1=mol_vector2*inv_v1_inv_v2-cosAngle*mol_vector1*inv_v1_sqr;
                   if (doUpDownSymmetry && cosAngle<0) {
                      der_mol1 *= -1.;
                   }
                   unsigned binAngle;
                   if (doUpDownSymmetry && cosAngle<0) {
                      binAngle=std::floor((-cosAngle-startCosAngle)/deltaCosAngle);
                   } else {
                      binAngle=std::floor((cosAngle-startCosAngle)/deltaCosAngle);
                   }
                   int minBin, maxBin; // These cannot be unsigned
                   // Only consider contributions to g(r) of atoms less than n*sigma bins apart from the actual distance
                   minBin=bin - deltaBin;
                   if (minBin < 0) minBin=0;
                   if (minBin > (nhist_[0]-1)) minBin=nhist_[0]-1;
                   maxBin=bin +  deltaBin;
                   if (maxBin > (nhist_[0]-1)) maxBin=nhist_[0]-1;
                   int minBinAngle, maxBinAngle; // These cannot be unsigned
                   // Only consider contributions to g(r) of atoms less than n*sigma bins apart from the actual angle
                   minBinAngle=binAngle - deltaBinAngle;
                   maxBinAngle=binAngle +  deltaBinAngle;
                   for(int k=minBin;k<maxBin+1;k+=1) {
                     double invNormKernelAA=invNormConstantBaseAA/x1sqr[k];
                     double invNormKernelBB=invNormConstantBaseBB/x1sqr[k];
                     double invNormKernelAB=invNormConstantBaseAB/x1sqr[k];
                     vector<double> pos(2);
                     pos[0]=x1[k]-distanceModulo;
                     for(int l=minBinAngle;l<maxBinAngle+1;l+=1) {
                        double theta=startCosAngle+deltaCosAngle*l;
                        if (doUpDownSymmetry && cosAngle<0) {
                           pos[1]=theta+cosAngle;
                        } else {
                           pos[1]=theta-cosAngle;
                        }
                        // Include periodic effects
                        int h;
                        if (l<0) {
                           h=-l;
                        } else if (l>(nhist_[1]-1)) {
                           h=2*nhist_[1]-l-2;
                        } else {
                           h=l;
                        }
                        vector<double> dfuncAA(2);
                        vector<double> dfuncBB(2);
                        vector<double> dfuncAB(2);
                        double kernelAA, kernelBB, kernelAB;
                        if (l==(nhist_[1]-1) || l==0) {
                           kernelAA = kernel(pos,2.*invNormKernelAA,dfuncAA);
                           kernelBB = kernel(pos,2.*invNormKernelBB,dfuncBB);
                           kernelAB = kernel(pos,2.*invNormKernelAB,dfuncAB);
                        } else {
                           kernelAA = kernel(pos,invNormKernelAA,dfuncAA);
                           kernelBB = kernel(pos,invNormKernelBB,dfuncBB);
                           kernelAB = kernel(pos,invNormKernelAB,dfuncAB);
                        }
                        // These are divided by 2 because they will be summed twice
                        gofrAA[k][h] += kernelAA*spin[index]*spin[neighbor]/2.;
                        gofrBB[k][h] += kernelBB*(1-spin[index])*(1-spin[neighbor])/2.;
                        gofrAB[k][h] += kernelAB*(spin[index]+spin[neighbor]-2*spin[index]*spin[neighbor])/2.;
           
                        Vector value1AA = dfuncAA[0]*distance_versor;
                        Vector value1BB = dfuncBB[0]*distance_versor;
                        Vector value1AB = dfuncAB[0]*distance_versor;
                        Vector value2_mol1AA = dfuncAA[1]*der_mol1;
                        Vector value2_mol1BB = dfuncBB[1]*der_mol1;
                        Vector value2_mol1AB = dfuncAB[1]*der_mol1;
                        gofrPrimeCenterAA[index*nhist1_nhist2_+k*nhist_[1]+h] += value1AA*spin[index]*spin[neighbor];
                        gofrPrimeCenterBB[index*nhist1_nhist2_+k*nhist_[1]+h] += value1BB*(1-spin[index])*(1-spin[neighbor]);
                        gofrPrimeCenterAB[index*nhist1_nhist2_+k*nhist_[1]+h] += value1AB*(spin[index]+spin[neighbor]-2*spin[index]*spin[neighbor]);
                        gofrPrimeStartAA[index*nhist1_nhist2_+k*nhist_[1]+h] +=  value2_mol1AA*spin[index]*spin[neighbor];
                        gofrPrimeStartBB[index*nhist1_nhist2_+k*nhist_[1]+h] +=  value2_mol1BB*(1-spin[index])*(1-spin[neighbor]);
                        gofrPrimeStartAB[index*nhist1_nhist2_+k*nhist_[1]+h] +=  value2_mol1AB*(spin[index]+spin[neighbor]-2*spin[index]*spin[neighbor]);
                        gofrPrimeEndAA[index*nhist1_nhist2_+k*nhist_[1]+h] -=  value2_mol1AA*spin[index]*spin[neighbor];
                        gofrPrimeEndBB[index*nhist1_nhist2_+k*nhist_[1]+h] -=  value2_mol1BB*(1-spin[index])*(1-spin[neighbor]);
                        gofrPrimeEndAB[index*nhist1_nhist2_+k*nhist_[1]+h] -=  value2_mol1AB*(spin[index]+spin[neighbor]-2*spin[index]*spin[neighbor]);
           
                        gofrPrimeVector1StartAA[index*nhist1_nhist2_+k*nhist_[1]+h] +=  kernelAA*spin[neighbor]*deriv_spin[index];
                        gofrPrimeVector2StartAA[index*nhist1_nhist2_+k*nhist_[1]+h] +=  kernelAA*spin[neighbor]*deriv_spin[index+2*number_molecules];
                        gofrPrimeVector3StartAA[index*nhist1_nhist2_+k*nhist_[1]+h] +=  kernelAA*spin[neighbor]*deriv_spin[index+4*number_molecules];
                        gofrPrimeVector1StartBB[index*nhist1_nhist2_+k*nhist_[1]+h] += -kernelBB*(1-spin[neighbor])*deriv_spin[index];
                        gofrPrimeVector2StartBB[index*nhist1_nhist2_+k*nhist_[1]+h] += -kernelBB*(1-spin[neighbor])*deriv_spin[index+2*number_molecules];
                        gofrPrimeVector3StartBB[index*nhist1_nhist2_+k*nhist_[1]+h] += -kernelBB*(1-spin[neighbor])*deriv_spin[index+4*number_molecules];
                        gofrPrimeVector1StartAB[index*nhist1_nhist2_+k*nhist_[1]+h] +=  kernelAB*(1-2*spin[neighbor])*deriv_spin[index];
                        gofrPrimeVector2StartAB[index*nhist1_nhist2_+k*nhist_[1]+h] +=  kernelAB*(1-2*spin[neighbor])*deriv_spin[index+2*number_molecules];
                        gofrPrimeVector3StartAB[index*nhist1_nhist2_+k*nhist_[1]+h] +=  kernelAB*(1-2*spin[neighbor])*deriv_spin[index+4*number_molecules];
           
                        Tensor vv1AA(value1AA, distance);
                        Tensor vv1BB(value1BB, distance);
                        Tensor vv1AB(value1AB, distance);
                        Tensor vv2_mol1AA(value2_mol1AA, mol_vector1);
                        Tensor vv2_mol1BB(value2_mol1BB, mol_vector1);
                        Tensor vv2_mol1AB(value2_mol1AB, mol_vector1);
                        Tensor vv3_vector1_AA(kernelAA*spin[neighbor]*deriv_spin[index],vectors1[index]);
                        Tensor vv3_vector2_AA(kernelAA*spin[neighbor]*deriv_spin[index+2*number_molecules],vectors2[index]);
                        Tensor vv3_vector3_AA(kernelAA*spin[neighbor]*deriv_spin[index+4*number_molecules],vectors3[index]);
                        Tensor vv3_vector1_BB(-kernelBB*(1-spin[neighbor])*deriv_spin[index],vectors1[index]);
                        Tensor vv3_vector2_BB(-kernelBB*(1-spin[neighbor])*deriv_spin[index+2*number_molecules],vectors2[index]);
                        Tensor vv3_vector3_BB(-kernelBB*(1-spin[neighbor])*deriv_spin[index+4*number_molecules],vectors3[index]);
                        Tensor vv3_vector1_AB(kernelAB*(1-2*spin[neighbor])*deriv_spin[index],vectors1[index]);
                        Tensor vv3_vector2_AB(kernelAB*(1-2*spin[neighbor])*deriv_spin[index+2*number_molecules],vectors2[index]);
                        Tensor vv3_vector3_AB(kernelAB*(1-2*spin[neighbor])*deriv_spin[index+4*number_molecules],vectors3[index]);
           
                        gofrVirialAA[k][h] += (vv1AA/2.+vv2_mol1AA)*spin[index]*spin[neighbor] + vv3_vector1_AA + vv3_vector2_AA + vv3_vector3_AA;
                        gofrVirialBB[k][h] += (vv1BB/2.+vv2_mol1BB)*(1-spin[index])*(1-spin[neighbor]) + vv3_vector1_BB + vv3_vector2_BB + vv3_vector3_BB;
                        gofrVirialAB[k][h] += (vv1AB/2.+vv2_mol1AB)*(spin[index]+spin[neighbor]-2*spin[index]*spin[neighbor]) + vv3_vector1_AB + vv3_vector2_AB + vv3_vector3_AB;
                     }
		   }
                 }
	       }
             }
           }
        }
     }
  } else if (!doneigh) {
    for(unsigned int i=rank;i<center_lista.size();i+=stride) {
      unsigned atom1_mol1=i+center_lista.size();
      unsigned atom2_mol1=i+center_lista.size()+start_lista.size();
      Vector mol_vector1=pbcDistance(getPosition(atom1_mol1),getPosition(atom2_mol1)); 
      double norm_v1 = std::sqrt(mol_vector1[0]*mol_vector1[0]+mol_vector1[1]*mol_vector1[1]+mol_vector1[2]*mol_vector1[2]);
      double inv_v1=1./norm_v1;
      double inv_v1_sqr=inv_v1*inv_v1;
      for(unsigned int j=0;j<center_lista.size();j+=1) {
        double d2;
        Vector distanceMinimumImage;
        if(pbc){
         distanceMinimumImage=pbcDistance(getPosition(i),getPosition(j));
        } else {
         distanceMinimumImage=delta(getPosition(i),getPosition(j));
        }
	for (int l=-periodic_images; l<(periodic_images+1); l++) {
	  for (int m=-periodic_images; m<(periodic_images+1); m++) {
	    for (int n=-periodic_images; n<(periodic_images+1); n++) {
	      // if same atom and same image
              if(getAbsoluteIndex(i)==getAbsoluteIndex(j) && l==0 && m==0 && n==0) continue;
	      Vector distance=distanceMinimumImage;
	      distance += l*BoxVector1;
	      distance += m*BoxVector2;
	      distance += n*BoxVector3;
              //if (distance.modulo()<sqrt(rcut2)) log.printf("index %d , neigh %d , l %d , m %d, n %d, distance %f \n", i, j, l, m, n, distance.modulo() );
              if ( (d2=distance[0]*distance[0])<rcut2 && (d2+=distance[1]*distance[1])<rcut2 && (d2+=distance[2]*distance[2])<rcut2) {
                double distanceModulo=std::sqrt(d2);
                Vector distance_versor = distance / distanceModulo;
                unsigned bin=std::floor(distanceModulo/deltar);
                unsigned atom1_mol2=j+center_lista.size();
                unsigned atom2_mol2=j+center_lista.size()+start_lista.size();
                Vector mol_vector2=pbcDistance(getPosition(atom1_mol2),getPosition(atom2_mol2));
                double norm_v2 = std::sqrt(mol_vector2[0]*mol_vector2[0]+mol_vector2[1]*mol_vector2[1]+mol_vector2[2]*mol_vector2[2]);
                double inv_v2=1./norm_v2;
                double inv_v1_inv_v2=inv_v1*inv_v2;
                double cosAngle=dotProduct(mol_vector1,mol_vector2)*inv_v1*inv_v2;
                Vector der_mol1=mol_vector2*inv_v1_inv_v2-cosAngle*mol_vector1*inv_v1_sqr;
                if (doUpDownSymmetry && cosAngle<0) {
                   der_mol1 *= -1.;
                }
                unsigned binAngle;
                if (doUpDownSymmetry && cosAngle<0) {
                   binAngle=std::floor((-cosAngle-startCosAngle)/deltaCosAngle);
                } else {
                   binAngle=std::floor((cosAngle-startCosAngle)/deltaCosAngle);
                }
                int minBin, maxBin; // These cannot be unsigned
                // Only consider contributions to g(r) of atoms less than n*sigma bins apart from the actual distance
                minBin=bin - deltaBin;
                if (minBin < 0) minBin=0;
                if (minBin > (nhist_[0]-1)) minBin=nhist_[0]-1;
                maxBin=bin +  deltaBin;
                if (maxBin > (nhist_[0]-1)) maxBin=nhist_[0]-1;
                int minBinAngle, maxBinAngle; // These cannot be unsigned
                // Only consider contributions to g(r) of atoms less than n*sigma bins apart from the actual angle
                minBinAngle=binAngle - deltaBinAngle;
                maxBinAngle=binAngle +  deltaBinAngle;
                for(int k=minBin;k<maxBin+1;k+=1) {
                  double invNormKernelAA=invNormConstantBaseAA/x1sqr[k];
                  double invNormKernelBB=invNormConstantBaseBB/x1sqr[k];
                  double invNormKernelAB=invNormConstantBaseAB/x1sqr[k];
                  vector<double> pos(2);
                  pos[0]=x1[k]-distanceModulo;
                  for(int l=minBinAngle;l<maxBinAngle+1;l+=1) {
                    double theta=startCosAngle+deltaCosAngle*l;
                    if (doUpDownSymmetry && cosAngle<0) {
                       pos[1]=theta+cosAngle;
                    } else {
                       pos[1]=theta-cosAngle;
                    }
                    // Include periodic effects
                    int h;
                    if (l<0) {
                       h=-l;
                    } else if (l>(nhist_[1]-1)) {
                       h=2*nhist_[1]-l-2;
                    } else {
                       h=l;
                    }
                    vector<double> dfuncAA(2);
                    vector<double> dfuncBB(2);
                    vector<double> dfuncAB(2);
                    double kernelAA, kernelBB, kernelAB;
                    if (l==(nhist_[1]-1) || l==0) {
                       kernelAA = kernel(pos,2.*invNormKernelAA,dfuncAA);
                       kernelBB = kernel(pos,2.*invNormKernelBB,dfuncBB);
                       kernelAB = kernel(pos,2.*invNormKernelAB,dfuncAB);
                    } else {
                       kernelAA = kernel(pos,invNormKernelAA,dfuncAA);
                       kernelBB = kernel(pos,invNormKernelBB,dfuncBB);
                       kernelAB = kernel(pos,invNormKernelAB,dfuncAB);
                    }
         
                    gofrAA[k][h] += kernelAA*spin[i]*spin[j]/2.;
                    gofrBB[k][h] += kernelBB*(1-spin[i])*(1-spin[j])/2.;
                    gofrAB[k][h] += kernelAB*(spin[i]+spin[j]-2*spin[i]*spin[j])/2.;
         
                    Vector value1AA = dfuncAA[0]*distance_versor;
                    Vector value1BB = dfuncBB[0]*distance_versor;
                    Vector value1AB = dfuncAB[0]*distance_versor;
                    Vector value2_mol1AA = dfuncAA[1]*der_mol1;
                    Vector value2_mol1BB = dfuncBB[1]*der_mol1;
                    Vector value2_mol1AB = dfuncAB[1]*der_mol1;
                    //log.printf("dfuncAA %f \n", dfuncAA[1]);
                    gofrPrimeCenterAA[i*nhist1_nhist2_+k*nhist_[1]+h] += value1AA*spin[i]*spin[j];
                    gofrPrimeCenterBB[i*nhist1_nhist2_+k*nhist_[1]+h] += value1BB*(1-spin[i])*(1-spin[j]);
                    gofrPrimeCenterAB[i*nhist1_nhist2_+k*nhist_[1]+h] += value1AB*(spin[i]+spin[j]-2*spin[i]*spin[j]);
                    gofrPrimeStartAA[i*nhist1_nhist2_+k*nhist_[1]+h] +=  value2_mol1AA*spin[i]*spin[j];
                    gofrPrimeStartBB[i*nhist1_nhist2_+k*nhist_[1]+h] +=  value2_mol1BB*(1-spin[i])*(1-spin[j]);
                    gofrPrimeStartAB[i*nhist1_nhist2_+k*nhist_[1]+h] +=  value2_mol1AB*(spin[i]+spin[j]-2*spin[i]*spin[j]);
                    //log.printf("derivStart value2_mol1AA x %f , y %f , z %f \n", value2_mol1AA[0], value2_mol1AA[1], value2_mol1AA[2]);
                    //log.printf("derivStart value2_mol1BB x %f , y %f , z %f \n", value2_mol1BB[0], value2_mol1BB[1], value2_mol1BB[2]);
                    //log.printf("derivStart value2_mol1AB x %f , y %f , z %f \n", value2_mol1AB[0], value2_mol1AB[1], value2_mol1AB[2]);
                    gofrPrimeEndAA[i*nhist1_nhist2_+k*nhist_[1]+h] -=  value2_mol1AA*spin[i]*spin[j];
                    gofrPrimeEndBB[i*nhist1_nhist2_+k*nhist_[1]+h] -=  value2_mol1BB*(1-spin[i])*(1-spin[j]);
                    gofrPrimeEndAB[i*nhist1_nhist2_+k*nhist_[1]+h] -=  value2_mol1AB*(spin[i]+spin[j]-2*spin[i]*spin[j]);
         
                    gofrPrimeVector1StartAA[i*nhist1_nhist2_+k*nhist_[1]+h] +=  kernelAA*spin[j]*deriv_spin[i];
                    gofrPrimeVector2StartAA[i*nhist1_nhist2_+k*nhist_[1]+h] +=  kernelAA*spin[j]*deriv_spin[i+2*number_molecules];
                    gofrPrimeVector3StartAA[i*nhist1_nhist2_+k*nhist_[1]+h] +=  kernelAA*spin[j]*deriv_spin[i+4*number_molecules];
                    gofrPrimeVector1StartBB[i*nhist1_nhist2_+k*nhist_[1]+h] += -kernelBB*(1-spin[j])*deriv_spin[i];
                    gofrPrimeVector2StartBB[i*nhist1_nhist2_+k*nhist_[1]+h] += -kernelBB*(1-spin[j])*deriv_spin[i+2*number_molecules];
                    gofrPrimeVector3StartBB[i*nhist1_nhist2_+k*nhist_[1]+h] += -kernelBB*(1-spin[j])*deriv_spin[i+4*number_molecules];
                    gofrPrimeVector1StartAB[i*nhist1_nhist2_+k*nhist_[1]+h] +=  kernelAB*(1-2*spin[j])*deriv_spin[i];
                    gofrPrimeVector2StartAB[i*nhist1_nhist2_+k*nhist_[1]+h] +=  kernelAB*(1-2*spin[j])*deriv_spin[i+2*number_molecules];
                    gofrPrimeVector3StartAB[i*nhist1_nhist2_+k*nhist_[1]+h] +=  kernelAB*(1-2*spin[j])*deriv_spin[i+4*number_molecules];
         
                    Tensor vv1AA(value1AA, distance);
                    Tensor vv1BB(value1BB, distance);
                    Tensor vv1AB(value1AB, distance);
                    Tensor vv2_mol1AA(value2_mol1AA, mol_vector1);
                    Tensor vv2_mol1BB(value2_mol1BB, mol_vector1);
                    Tensor vv2_mol1AB(value2_mol1AB, mol_vector1);
                    Tensor vv3_vector1_AA(kernelAA*spin[j]*deriv_spin[i],vectors1[i]);
                    Tensor vv3_vector2_AA(kernelAA*spin[j]*deriv_spin[i+2*number_molecules],vectors2[i]);
                    Tensor vv3_vector3_AA(kernelAA*spin[j]*deriv_spin[i+4*number_molecules],vectors3[i]);
                    Tensor vv3_vector1_BB(-kernelBB*(1-spin[j])*deriv_spin[i],vectors1[i]);
                    Tensor vv3_vector2_BB(-kernelBB*(1-spin[j])*deriv_spin[i+2*number_molecules],vectors2[i]);
                    Tensor vv3_vector3_BB(-kernelBB*(1-spin[j])*deriv_spin[i+4*number_molecules],vectors3[i]);
                    Tensor vv3_vector1_AB(kernelAB*(1-2*spin[j])*deriv_spin[i],vectors1[i]);
                    Tensor vv3_vector2_AB(kernelAB*(1-2*spin[j])*deriv_spin[i+2*number_molecules],vectors2[i]);
                    Tensor vv3_vector3_AB(kernelAB*(1-2*spin[j])*deriv_spin[i+4*number_molecules],vectors3[i]);
         
                    gofrVirialAA[k][h] += (vv1AA/2.+vv2_mol1AA)*spin[i]*spin[j] + vv3_vector1_AA + vv3_vector2_AA + vv3_vector3_AA;
                    gofrVirialBB[k][h] += (vv1BB/2.+vv2_mol1BB)*(1-spin[i])*(1-spin[j]) + vv3_vector1_BB + vv3_vector2_BB + vv3_vector3_BB;
                    gofrVirialAB[k][h] += (vv1AB/2.+vv2_mol1AB)*(spin[i]+spin[j]-2*spin[i]*spin[j]) + vv3_vector1_AB + vv3_vector2_AB + vv3_vector3_AB;
                  }
                }
              }
	    }
	  }
        }
      }
    }
  } 
  if(!serial){
    comm.Sum(gofrAA);
    comm.Sum(gofrBB);
    comm.Sum(gofrAB);
    if (!doNotCalculateDerivatives() ) {
       comm.Sum(gofrVirialAA);
       comm.Sum(gofrVirialBB);
       comm.Sum(gofrVirialAB);
    }
  }
  if (doAverageGofr) {
     if (!doNotCalculateDerivatives()) error("Cannot calculate derivatives or bias using the AVERAGE_GOFR option");
     double factor;
     if (averageGofrTau==0 || iteration < averageGofrTau) {
        iteration += 1;
        factor = 1./( (double) iteration );
     } else factor = 2./((double) averageGofrTau + 1.);
     for(unsigned i=0;i<nhist_[0];++i){
        for(unsigned j=0;j<nhist_[1];++j){
           avgGofrAA[i][j] += (gofrAA[i][j]-avgGofrAA[i][j])*factor;
           avgGofrBB[i][j] += (gofrBB[i][j]-avgGofrBB[i][j])*factor;
           avgGofrAB[i][j] += (gofrAB[i][j]-avgGofrAB[i][j])*factor;
           gofrAA[i][j] = avgGofrAA[i][j];
           gofrBB[i][j] = avgGofrBB[i][j];
           gofrAB[i][j] = avgGofrAB[i][j];
        }
     }
  }
  // Output of gofr
  if (doOutputGofr && (getStep()%outputStride==0)) outputGofr(gofrAA,gofrBB,gofrAB);
  // Construct integrand
  Matrix<double> integrandAA(nhist_[0],nhist_[1]);
  Matrix<double> integrandBB(nhist_[0],nhist_[1]);
  Matrix<double> integrandAB(nhist_[0],nhist_[1]);
  Matrix<double> logGofrAAx1sqr(nhist_[0],nhist_[1]);
  Matrix<double> logGofrBBx1sqr(nhist_[0],nhist_[1]);
  Matrix<double> logGofrABx1sqr(nhist_[0],nhist_[1]);
  for(unsigned i=0;i<nhist_[0];++i){
     for(unsigned j=0;j<nhist_[1];++j){
        if (gofrAA[i][j]<1.e-10) {
           integrandAA[i][j] = x1sqr[i];
        } else {
           logGofrAAx1sqr[i][j] = std::log(gofrAA[i][j])*x1sqr[i];
           integrandAA[i][j] = gofrAA[i][j]*logGofrAAx1sqr[i][j]+(-gofrAA[i][j]+1)*x1sqr[i];
        }
        if (gofrBB[i][j]<1.e-10) {
           integrandBB[i][j] = x1sqr[i];
        } else {
           logGofrBBx1sqr[i][j] = std::log(gofrBB[i][j])*x1sqr[i];
           integrandBB[i][j] = gofrBB[i][j]*logGofrBBx1sqr[i][j]+(-gofrBB[i][j]+1)*x1sqr[i];
        }
        if (gofrAB[i][j]<1.e-10) {
           integrandAB[i][j] = x1sqr[i];
        } else {
           logGofrABx1sqr[i][j] = std::log(gofrAB[i][j])*x1sqr[i];
           integrandAB[i][j] = gofrAB[i][j]*logGofrABx1sqr[i][j]+(-gofrAB[i][j]+1)*x1sqr[i];
        }
     }
  }
  vector<double> delta(2);
  delta[0]=deltar;
  delta[1]=deltaCosAngle;
  double prefactorAA=(2*pi/volumeOfAngles)*densityA*densityA/density;
  double prefactorBB=(2*pi/volumeOfAngles)*densityB*densityB/density;
  double prefactorAB=(4*pi/volumeOfAngles)*densityA*densityB/density;
  double pairAAvalue=-prefactorAA*integrate(integrandAA,delta);
  double pairBBvalue=-prefactorBB*integrate(integrandBB,delta);
  double pairABvalue=-prefactorAB*integrate(integrandAB,delta);
  double oneBody = 0.;
  if (one_body) {
    oneBody += (8./2. - std::log(densityA*deBroglie3))*number_A_molecules/number_molecules;
    oneBody += (8./2. - std::log(densityB*deBroglie3))*number_B_molecules/number_molecules;
  }
  // Derivatives
  vector<Vector> derivAA(getNumberOfAtoms());
  vector<Vector> derivBB(getNumberOfAtoms());
  vector<Vector> derivAB(getNumberOfAtoms());
  vector<Vector> derivOneBody(getNumberOfAtoms());
  Tensor virialAA, virialBB, virialAB, virialOneBody;
  if (!doNotCalculateDerivatives() ) {
    Tensor gofrPrefactorVirial;
    if (doneigh) {
       for(unsigned int k=0;k<nl->getNumberOfLocalAtoms();k+=1) {
         unsigned index=nl->getIndexOfLocalAtom(k);
         // Center atom
         unsigned start_atom=index+center_lista.size();
         unsigned end_atom=index+2*center_lista.size();
         unsigned vector1_start_atom=index+3*center_lista.size();
         unsigned vector1_end_atom=index+4*center_lista.size();
         unsigned vector2_start_atom=index+5*center_lista.size();
         unsigned vector2_end_atom=index+6*center_lista.size();
         unsigned vector3_start_atom=index+7*center_lista.size();
         unsigned vector3_end_atom=index+8*center_lista.size();
         Matrix<Vector> integrandDerivativesAA(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesBB(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesAB(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesStartAA(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesStartBB(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesStartAB(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesEndAA(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesEndBB(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesEndAB(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesVector1StartAA(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesVector2StartAA(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesVector3StartAA(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesVector1StartBB(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesVector2StartBB(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesVector3StartBB(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesVector1StartAB(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesVector2StartAB(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesVector3StartAB(nhist_[0],nhist_[1]);
         gofrPrefactorVirial += Tensor(deriv_spin[index],vectors1[index])+Tensor(deriv_spin[index+2*number_molecules],vectors2[index])+Tensor(deriv_spin[index+4*number_molecules],vectors3[index]);
         for(unsigned i=0;i<nhist_[0];++i){
           for(unsigned j=0;j<nhist_[1];++j){
             if (gofrAA[i][j]>1.e-10) {
               integrandDerivativesAA[i][j] = gofrPrimeCenterAA[index*nhist1_nhist2_+i*nhist_[1]+j]*logGofrAAx1sqr[i][j];
               integrandDerivativesStartAA[i][j] = gofrPrimeStartAA[index*nhist1_nhist2_+i*nhist_[1]+j]*logGofrAAx1sqr[i][j];
               integrandDerivativesEndAA[i][j] = gofrPrimeEndAA[index*nhist1_nhist2_+i*nhist_[1]+j]*logGofrAAx1sqr[i][j];
               integrandDerivativesVector1StartAA[i][j] = 
                 (gofrPrimeVector1StartAA[index*nhist1_nhist2_+i*nhist_[1]+j]-gofrAA[i][j]*(2./number_A_molecules)*deriv_spin[index])*logGofrAAx1sqr[i][j];
               integrandDerivativesVector2StartAA[i][j] = 
                 (gofrPrimeVector2StartAA[index*nhist1_nhist2_+i*nhist_[1]+j]-gofrAA[i][j]*(2./number_A_molecules)*deriv_spin[index+2*number_molecules])*logGofrAAx1sqr[i][j];
               integrandDerivativesVector3StartAA[i][j] = 
                 (gofrPrimeVector3StartAA[index*nhist1_nhist2_+i*nhist_[1]+j]-gofrAA[i][j]*(2./number_A_molecules)*deriv_spin[index+4*number_molecules])*logGofrAAx1sqr[i][j];
             }
             if (gofrBB[i][j]>1.e-10) {
               integrandDerivativesBB[i][j] = gofrPrimeCenterBB[index*nhist1_nhist2_+i*nhist_[1]+j]*logGofrBBx1sqr[i][j];
               integrandDerivativesStartBB[i][j] = gofrPrimeStartBB[index*nhist1_nhist2_+i*nhist_[1]+j]*logGofrBBx1sqr[i][j];
               integrandDerivativesEndBB[i][j] = gofrPrimeEndBB[index*nhist1_nhist2_+i*nhist_[1]+j]*logGofrBBx1sqr[i][j];
               integrandDerivativesVector1StartBB[i][j] = 
                 (gofrPrimeVector1StartBB[index*nhist1_nhist2_+i*nhist_[1]+j]+gofrBB[i][j]*(2./number_B_molecules)*deriv_spin[index])*logGofrBBx1sqr[i][j];
               integrandDerivativesVector2StartBB[i][j] = 
                 (gofrPrimeVector2StartBB[index*nhist1_nhist2_+i*nhist_[1]+j]+gofrBB[i][j]*(2./number_B_molecules)*deriv_spin[index+2*number_molecules])*logGofrBBx1sqr[i][j];
               integrandDerivativesVector3StartBB[i][j] = 
                 (gofrPrimeVector3StartBB[index*nhist1_nhist2_+i*nhist_[1]+j]+gofrBB[i][j]*(2./number_B_molecules)*deriv_spin[index+4*number_molecules])*logGofrBBx1sqr[i][j];
             }
             if (gofrAB[i][j]>1.e-10) {
               integrandDerivativesAB[i][j] = gofrPrimeCenterAB[index*nhist1_nhist2_+i*nhist_[1]+j]*logGofrABx1sqr[i][j];
               integrandDerivativesStartAB[i][j] = gofrPrimeStartAB[index*nhist1_nhist2_+i*nhist_[1]+j]*logGofrABx1sqr[i][j];
               integrandDerivativesEndAB[i][j] = gofrPrimeEndAB[index*nhist1_nhist2_+i*nhist_[1]+j]*logGofrABx1sqr[i][j];
               integrandDerivativesVector1StartAB[i][j] = 
                 (gofrPrimeVector1StartAB[index*nhist1_nhist2_+i*nhist_[1]+j]-gofrAB[i][j]*((number_molecules-2.*number_A_molecules)/(number_A_molecules*number_B_molecules))*deriv_spin[index])*logGofrABx1sqr[i][j];
               integrandDerivativesVector2StartAB[i][j] = 
                 (gofrPrimeVector2StartAB[index*nhist1_nhist2_+i*nhist_[1]+j]-gofrAB[i][j]*((number_molecules-2.*number_A_molecules)/(number_A_molecules*number_B_molecules))*deriv_spin[index+2*number_molecules])*logGofrABx1sqr[i][j];
               integrandDerivativesVector3StartAB[i][j] = 
                 (gofrPrimeVector3StartAB[index*nhist1_nhist2_+i*nhist_[1]+j]-gofrAB[i][j]*((number_molecules-2.*number_A_molecules)/(number_A_molecules*number_B_molecules))*deriv_spin[index+4*number_molecules])*logGofrABx1sqr[i][j];
             }
           }
         }
         derivAA[index] = -prefactorAA*integrate(integrandDerivativesAA,delta);
         derivBB[index] = -prefactorBB*integrate(integrandDerivativesBB,delta);
         derivAB[index] = -prefactorAB*integrate(integrandDerivativesAB,delta);
         derivAA[start_atom] = -prefactorAA*integrate(integrandDerivativesStartAA,delta);
         derivBB[start_atom] = -prefactorBB*integrate(integrandDerivativesStartBB,delta);
         derivAB[start_atom] = -prefactorAB*integrate(integrandDerivativesStartAB,delta);
         derivAA[end_atom] = -prefactorAA*integrate(integrandDerivativesEndAA,delta);
         derivBB[end_atom] = -prefactorBB*integrate(integrandDerivativesEndBB,delta);
         derivAB[end_atom] = -prefactorAB*integrate(integrandDerivativesEndAB,delta);
         derivAA[vector1_start_atom] = -prefactorAA*integrate(integrandDerivativesVector1StartAA,delta)
                                       -prefactorAA*(2./number_A_molecules)*deriv_spin[index]*integrate(integrandAA,delta);
         derivBB[vector1_start_atom] = -prefactorBB*integrate(integrandDerivativesVector1StartBB,delta)
                                       -prefactorBB*(2./number_B_molecules)*(-deriv_spin[index])*integrate(integrandBB,delta);
         derivAB[vector1_start_atom] = -prefactorAB*integrate(integrandDerivativesVector1StartAB,delta)
                                       -prefactorAB*((number_molecules-2.*number_A_molecules)/(number_A_molecules*number_B_molecules))*deriv_spin[index]*integrate(integrandAB,delta);
         if (one_body) {
           derivOneBody[vector1_start_atom] += (6./2. - std::log(densityA*deBroglie3))*deriv_spin[index]/number_molecules
                                            +(6./2. - std::log(densityB*deBroglie3))*(-deriv_spin[index])/number_molecules;
         }
         derivAA[vector1_end_atom] = -derivAA[vector1_start_atom];
         derivBB[vector1_end_atom] = -derivBB[vector1_start_atom];
         derivAB[vector1_end_atom] = -derivAB[vector1_start_atom];
         derivOneBody[vector1_end_atom] = -derivOneBody[vector1_start_atom];
         derivAA[vector2_start_atom] = -prefactorAA*integrate(integrandDerivativesVector2StartAA,delta)
                                       -prefactorAA*(2./number_A_molecules)*deriv_spin[index+2*number_molecules]*integrate(integrandAA,delta);
         derivBB[vector2_start_atom] = -prefactorBB*integrate(integrandDerivativesVector2StartBB,delta)
                                       -prefactorBB*(2./number_B_molecules)*(-deriv_spin[index+2*number_molecules])*integrate(integrandBB,delta);
         derivAB[vector2_start_atom] = -prefactorAB*integrate(integrandDerivativesVector2StartAB,delta)
                                       -prefactorAB*((number_molecules-2.*number_A_molecules)/(number_A_molecules*number_B_molecules))*deriv_spin[index+2*number_molecules]*integrate(integrandAB,delta);
         if (one_body) {
           derivOneBody[vector2_start_atom] += (6./2. - std::log(densityA*deBroglie3))*deriv_spin[index+2*number_molecules]/number_molecules
                                            +(6./2. - std::log(densityB*deBroglie3))*(-deriv_spin[index+2*number_molecules])/number_molecules;
         }
         derivAA[vector2_end_atom] = -derivAA[vector2_start_atom];
         derivBB[vector2_end_atom] = -derivBB[vector2_start_atom];
         derivAB[vector2_end_atom] = -derivAB[vector2_start_atom];
         derivOneBody[vector2_end_atom] = -derivOneBody[vector2_start_atom];
         derivAA[vector3_start_atom] = -prefactorAA*integrate(integrandDerivativesVector3StartAA,delta)
                                       -prefactorAA*(2./number_A_molecules)*deriv_spin[index+4*number_molecules]*integrate(integrandAA,delta);
         derivBB[vector3_start_atom] = -prefactorBB*integrate(integrandDerivativesVector3StartBB,delta)
                                       -prefactorBB*(2./number_B_molecules)*(-deriv_spin[index+4*number_molecules])*integrate(integrandBB,delta);
         derivAB[vector3_start_atom] = -prefactorAB*integrate(integrandDerivativesVector3StartAB,delta)
                                       -prefactorAB*((number_molecules-2.*number_A_molecules)/(number_A_molecules*number_B_molecules))*deriv_spin[index+4*number_molecules]*integrate(integrandAB,delta);
         if (one_body) {
           derivOneBody[vector3_start_atom] += (6./2. - std::log(densityA*deBroglie3))*deriv_spin[index+4*number_molecules]/number_molecules
                                            +(6./2. - std::log(densityB*deBroglie3))*(-deriv_spin[index+4*number_molecules])/number_molecules;
         }
         derivAA[vector3_end_atom] = -derivAA[vector3_start_atom];
         derivBB[vector3_end_atom] = -derivBB[vector3_start_atom];
         derivAB[vector3_end_atom] = -derivAB[vector3_start_atom];
         derivOneBody[vector3_end_atom] = -derivOneBody[vector3_start_atom];
       }
    } else {
       for(unsigned int k=rank;k<center_lista.size();k+=stride) {
         // Center atom
         unsigned start_atom=k+center_lista.size();
         unsigned end_atom=k+2*center_lista.size();
         unsigned vector1_start_atom=k+3*center_lista.size();
         unsigned vector1_end_atom=k+4*center_lista.size();
         unsigned vector2_start_atom=k+5*center_lista.size();
         unsigned vector2_end_atom=k+6*center_lista.size();
         unsigned vector3_start_atom=k+7*center_lista.size();
         unsigned vector3_end_atom=k+8*center_lista.size();
         Matrix<Vector> integrandDerivativesAA(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesBB(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesAB(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesStartAA(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesStartBB(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesStartAB(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesEndAA(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesEndBB(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesEndAB(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesVector1StartAA(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesVector2StartAA(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesVector3StartAA(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesVector1StartBB(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesVector2StartBB(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesVector3StartBB(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesVector1StartAB(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesVector2StartAB(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesVector3StartAB(nhist_[0],nhist_[1]);
         gofrPrefactorVirial += Tensor(deriv_spin[k],vectors1[k])+Tensor(deriv_spin[k+2*number_molecules],vectors2[k])+Tensor(deriv_spin[k+4*number_molecules],vectors3[k]);
         for(unsigned i=0;i<nhist_[0];++i){
           for(unsigned j=0;j<nhist_[1];++j){
             if (gofrAA[i][j]>1.e-10) {
               integrandDerivativesAA[i][j] = gofrPrimeCenterAA[k*nhist1_nhist2_+i*nhist_[1]+j]*logGofrAAx1sqr[i][j];
               integrandDerivativesStartAA[i][j] = gofrPrimeStartAA[k*nhist1_nhist2_+i*nhist_[1]+j]*logGofrAAx1sqr[i][j];
               integrandDerivativesEndAA[i][j] = gofrPrimeEndAA[k*nhist1_nhist2_+i*nhist_[1]+j]*logGofrAAx1sqr[i][j];
               integrandDerivativesVector1StartAA[i][j] = 
                 (gofrPrimeVector1StartAA[k*nhist1_nhist2_+i*nhist_[1]+j]-gofrAA[i][j]*(2./number_A_molecules)*deriv_spin[k])*logGofrAAx1sqr[i][j];
               integrandDerivativesVector2StartAA[i][j] = 
                 (gofrPrimeVector2StartAA[k*nhist1_nhist2_+i*nhist_[1]+j]-gofrAA[i][j]*(2./number_A_molecules)*deriv_spin[k+2*number_molecules])*logGofrAAx1sqr[i][j];
               integrandDerivativesVector3StartAA[i][j] = 
                 (gofrPrimeVector3StartAA[k*nhist1_nhist2_+i*nhist_[1]+j]-gofrAA[i][j]*(2./number_A_molecules)*deriv_spin[k+4*number_molecules])*logGofrAAx1sqr[i][j];
             }
             if (gofrBB[i][j]>1.e-10) {
               integrandDerivativesBB[i][j] = gofrPrimeCenterBB[k*nhist1_nhist2_+i*nhist_[1]+j]*logGofrBBx1sqr[i][j];
               integrandDerivativesStartBB[i][j] = gofrPrimeStartBB[k*nhist1_nhist2_+i*nhist_[1]+j]*logGofrBBx1sqr[i][j];
               integrandDerivativesEndBB[i][j] = gofrPrimeEndBB[k*nhist1_nhist2_+i*nhist_[1]+j]*logGofrBBx1sqr[i][j];
               integrandDerivativesVector1StartBB[i][j] = 
                 (gofrPrimeVector1StartBB[k*nhist1_nhist2_+i*nhist_[1]+j]+gofrBB[i][j]*(2./number_B_molecules)*deriv_spin[k])*logGofrBBx1sqr[i][j];
               integrandDerivativesVector2StartBB[i][j] = 
                 (gofrPrimeVector2StartBB[k*nhist1_nhist2_+i*nhist_[1]+j]+gofrBB[i][j]*(2./number_B_molecules)*deriv_spin[k+2*number_molecules])*logGofrBBx1sqr[i][j];
               integrandDerivativesVector3StartBB[i][j] = 
                 (gofrPrimeVector3StartBB[k*nhist1_nhist2_+i*nhist_[1]+j]+gofrBB[i][j]*(2./number_B_molecules)*deriv_spin[k+4*number_molecules])*logGofrBBx1sqr[i][j];
             }
             if (gofrAB[i][j]>1.e-10) {
               integrandDerivativesAB[i][j] = gofrPrimeCenterAB[k*nhist1_nhist2_+i*nhist_[1]+j]*logGofrABx1sqr[i][j];
               integrandDerivativesStartAB[i][j] = gofrPrimeStartAB[k*nhist1_nhist2_+i*nhist_[1]+j]*logGofrABx1sqr[i][j];
               integrandDerivativesEndAB[i][j] = gofrPrimeEndAB[k*nhist1_nhist2_+i*nhist_[1]+j]*logGofrABx1sqr[i][j];
               integrandDerivativesVector1StartAB[i][j] = 
                 (gofrPrimeVector1StartAB[k*nhist1_nhist2_+i*nhist_[1]+j]-gofrAB[i][j]*((number_molecules-2.*number_A_molecules)/(number_A_molecules*number_B_molecules))*deriv_spin[k])*logGofrABx1sqr[i][j];
               integrandDerivativesVector2StartAB[i][j] = 
                 (gofrPrimeVector2StartAB[k*nhist1_nhist2_+i*nhist_[1]+j]-gofrAB[i][j]*((number_molecules-2.*number_A_molecules)/(number_A_molecules*number_B_molecules))*deriv_spin[k+2*number_molecules])*logGofrABx1sqr[i][j];
               integrandDerivativesVector3StartAB[i][j] = 
                 (gofrPrimeVector3StartAB[k*nhist1_nhist2_+i*nhist_[1]+j]-gofrAB[i][j]*((number_molecules-2.*number_A_molecules)/(number_A_molecules*number_B_molecules))*deriv_spin[k+4*number_molecules])*logGofrABx1sqr[i][j];
             }
           }
         }
         derivAA[k] = -prefactorAA*integrate(integrandDerivativesAA,delta);
         derivBB[k] = -prefactorBB*integrate(integrandDerivativesBB,delta);
         derivAB[k] = -prefactorAB*integrate(integrandDerivativesAB,delta);
         derivAA[start_atom] = -prefactorAA*integrate(integrandDerivativesStartAA,delta);
         derivBB[start_atom] = -prefactorBB*integrate(integrandDerivativesStartBB,delta);
         derivAB[start_atom] = -prefactorAB*integrate(integrandDerivativesStartAB,delta);
         derivAA[end_atom] = -prefactorAA*integrate(integrandDerivativesEndAA,delta);
         derivBB[end_atom] = -prefactorBB*integrate(integrandDerivativesEndBB,delta);
         derivAB[end_atom] = -prefactorAB*integrate(integrandDerivativesEndAB,delta);
         derivAA[vector1_start_atom] = -prefactorAA*integrate(integrandDerivativesVector1StartAA,delta)
                                       -prefactorAA*(2./number_A_molecules)*deriv_spin[k]*integrate(integrandAA,delta);
         derivBB[vector1_start_atom] = -prefactorBB*integrate(integrandDerivativesVector1StartBB,delta)
                                       -prefactorBB*(2./number_B_molecules)*(-deriv_spin[k])*integrate(integrandBB,delta);
         derivAB[vector1_start_atom] = -prefactorAB*integrate(integrandDerivativesVector1StartAB,delta)
                                       -prefactorAB*((number_molecules-2.*number_A_molecules)/(number_A_molecules*number_B_molecules))*deriv_spin[k]*integrate(integrandAB,delta);
         if (one_body) {
           derivOneBody[vector1_start_atom] += (6./2. - std::log(densityA*deBroglie3))*deriv_spin[k]/number_molecules
                                       +(6./2. - std::log(densityB*deBroglie3))*(-deriv_spin[k])/number_molecules;
         }
         derivAA[vector1_end_atom] = -derivAA[vector1_start_atom];
         derivBB[vector1_end_atom] = -derivBB[vector1_start_atom];
         derivAB[vector1_end_atom] = -derivAB[vector1_start_atom];
         derivOneBody[vector1_end_atom] = -derivOneBody[vector1_start_atom];
         derivAA[vector2_start_atom] = -prefactorAA*integrate(integrandDerivativesVector2StartAA,delta)
                                       -prefactorAA*(2./number_A_molecules)*deriv_spin[k+2*number_molecules]*integrate(integrandAA,delta);
         derivBB[vector2_start_atom] = -prefactorBB*integrate(integrandDerivativesVector2StartBB,delta)
                                       -prefactorBB*(2./number_B_molecules)*(-deriv_spin[k+2*number_molecules])*integrate(integrandBB,delta);
         derivAB[vector2_start_atom] = -prefactorAB*integrate(integrandDerivativesVector2StartAB,delta)
                                       -prefactorAB*((number_molecules-2.*number_A_molecules)/(number_A_molecules*number_B_molecules))*deriv_spin[k+2*number_molecules]*integrate(integrandAB,delta);
         if (one_body) {
           derivOneBody[vector2_start_atom] += (6./2. - std::log(densityA*deBroglie3))*deriv_spin[k+2*number_molecules]/number_molecules
                                       +(6./2. - std::log(densityB*deBroglie3))*(-deriv_spin[k+2*number_molecules])/number_molecules;
         }
         derivAA[vector2_end_atom] = -derivAA[vector2_start_atom];
         derivBB[vector2_end_atom] = -derivBB[vector2_start_atom];
         derivAB[vector2_end_atom] = -derivAB[vector2_start_atom];
         derivOneBody[vector2_end_atom] = -derivOneBody[vector2_start_atom];
         derivAA[vector3_start_atom] = -prefactorAA*integrate(integrandDerivativesVector3StartAA,delta)
                                       -prefactorAA*(2./number_A_molecules)*deriv_spin[k+4*number_molecules]*integrate(integrandAA,delta);
         derivBB[vector3_start_atom] = -prefactorBB*integrate(integrandDerivativesVector3StartBB,delta)
                                       -prefactorBB*(2./number_B_molecules)*(-deriv_spin[k+4*number_molecules])*integrate(integrandBB,delta);
         derivAB[vector3_start_atom] = -prefactorAB*integrate(integrandDerivativesVector3StartAB,delta)
                                       -prefactorAB*((number_molecules-2.*number_A_molecules)/(number_A_molecules*number_B_molecules))*deriv_spin[k+4*number_molecules]*integrate(integrandAB,delta);
         if (one_body) {
           derivOneBody[vector3_start_atom] += (6./2. - std::log(densityA*deBroglie3))*deriv_spin[k+4*number_molecules]/number_molecules
                                       +(6./2. - std::log(densityB*deBroglie3))*(-deriv_spin[k+4*number_molecules])/number_molecules;
         }
         derivAA[vector3_end_atom] = -derivAA[vector3_start_atom];
         derivBB[vector3_end_atom] = -derivBB[vector3_start_atom];
         derivAB[vector3_end_atom] = -derivAB[vector3_start_atom];
         derivOneBody[vector3_end_atom] = -derivOneBody[vector3_start_atom];
       }
    }
    if(!serial){
      comm.Sum(derivAA);
      comm.Sum(derivBB);
      comm.Sum(derivAB);
      comm.Sum(derivOneBody);
    }
    // Virial of positions
    // Construct virial integrand
    Matrix<Tensor> integrandVirialAA(nhist_[0],nhist_[1]);
    Matrix<Tensor> integrandVirialBB(nhist_[0],nhist_[1]);
    Matrix<Tensor> integrandVirialAB(nhist_[0],nhist_[1]);
    for(unsigned i=0;i<nhist_[0];++i){
       for(unsigned j=0;j<nhist_[1];++j){
          if (gofrAA[i][j]>1.e-10) {
             integrandVirialAA[i][j] = 
               (gofrVirialAA[i][j]-gofrAA[i][j]*(2./number_A_molecules)*gofrPrefactorVirial)*logGofrAAx1sqr[i][j];
          }
          if (gofrBB[i][j]>1.e-10) {
             integrandVirialBB[i][j] = 
               (gofrVirialBB[i][j]+gofrBB[i][j]*(2./number_B_molecules)*gofrPrefactorVirial)*logGofrBBx1sqr[i][j];
          }
          if (gofrAB[i][j]>1.e-10) {
             integrandVirialAB[i][j] = 
               (gofrVirialAB[i][j]-gofrAB[i][j]*((number_molecules-2.*number_A_molecules)/(number_A_molecules*number_B_molecules))*gofrPrefactorVirial)*logGofrABx1sqr[i][j];
          }
      }
    }
    // Integrate virial
    virialAA = -prefactorAA*integrate(integrandVirialAA,delta);
    virialBB = -prefactorBB*integrate(integrandVirialBB,delta);
    virialAB = -prefactorAB*integrate(integrandVirialAB,delta);
    // Virial of volume
    // Construct virial integrand
    Matrix<double> integrandVirialVolumeAA(nhist_[0],nhist_[1]);
    Matrix<double> integrandVirialVolumeBB(nhist_[0],nhist_[1]);
    Matrix<double> integrandVirialVolumeAB(nhist_[0],nhist_[1]);
    for(unsigned i=0;i<nhist_[0];++i){
       for(unsigned j=0;j<nhist_[1];++j){
          integrandVirialVolumeAA[i][j] = (-gofrAA[i][j]+1)*x1sqr[i];
          integrandVirialVolumeBB[i][j] = (-gofrBB[i][j]+1)*x1sqr[i];
          integrandVirialVolumeAB[i][j] = (-gofrAB[i][j]+1)*x1sqr[i];
       }
    }
    // Integrate virial
    virialAA += -prefactorAA*integrate(integrandVirialVolumeAA,delta)*Tensor::identity();
    virialBB += -prefactorBB*integrate(integrandVirialVolumeBB,delta)*Tensor::identity();
    virialAB += -prefactorAB*integrate(integrandVirialVolumeAB,delta)*Tensor::identity();
    // Virial of one body
    if (one_body) {
      virialOneBody = (6./2. - std::log(densityA*deBroglie3))*gofrPrefactorVirial/number_molecules;
      virialOneBody += (6./2. + std::log(densityB*deBroglie3))*gofrPrefactorVirial/number_molecules;
      // Virial of volume
      virialOneBody -= (number_A_molecules/number_molecules)*Tensor::identity();
      virialOneBody -= (number_B_molecules/number_molecules)*Tensor::identity();
    }
  }
  if (do_pairs) {
    Value* pairAA=getPntrToComponent("pairAA");
    Value* pairAB=getPntrToComponent("pairAB");
    Value* pairBB=getPntrToComponent("pairBB");
    Value* full=getPntrToComponent("full");
    pairAA->set(pairAAvalue);
    pairAB->set(pairABvalue);
    pairBB->set(pairBBvalue);
    full->set(pairAAvalue+pairABvalue+pairBBvalue+oneBody);
    if (!doNotCalculateDerivatives() ) {
      for(unsigned j=0;j<getNumberOfAtoms();++j) setAtomsDerivatives (pairAA,j,derivAA[j]);
      for(unsigned j=0;j<getNumberOfAtoms();++j) setAtomsDerivatives (pairAB,j,derivAB[j]);
      for(unsigned j=0;j<getNumberOfAtoms();++j) setAtomsDerivatives (pairBB,j,derivBB[j]);
      for(unsigned j=0;j<getNumberOfAtoms();++j) setAtomsDerivatives (full,j,derivAA[j]+derivAB[j]+derivBB[j]+derivOneBody[j]);
      //for(unsigned j=0;j<getNumberOfAtoms();++j) log.printf("derivAA x y z %f %f %f \n",derivAA[j][0],derivAA[j][1],derivAA[j][2]);
      setBoxDerivatives  (pairAA,virialAA);
      setBoxDerivatives  (pairBB,virialBB);
      setBoxDerivatives  (pairAB,virialAB);
      setBoxDerivatives  (full,virialAA+virialBB+virialAB+virialOneBody);
    }
  } else {
    setValue(pairAAvalue+pairABvalue+pairBBvalue+oneBody);
    for(unsigned j=0;j<getNumberOfAtoms();++j) setAtomsDerivatives (j,derivAA[j]+derivAB[j]+derivBB[j]+derivOneBody[j]);
    setBoxDerivatives  (virialAA+virialBB+virialAB+virialOneBody);
  } 
}

double PairOrientationalEntropyConformer::kernel(vector<double> distance, double invNormKernel, vector<double>&der)const{
  // Gaussian function and derivative
  double result = invNormKernel*std::exp(-distance[0]*distance[0]/twoSigma1Sqr-distance[1]*distance[1]/twoSigma2Sqr) ;
  der[0] = -distance[0]*result/sigma1Sqr;
  der[1] = -distance[1]*result/sigma2Sqr;
  return result;
}

double PairOrientationalEntropyConformer::integrate(Matrix<double> integrand, vector<double> delta)const{
  // Trapezoid rule
  double result = 0.;
  for(unsigned i=1;i<(nhist_[0]-1);++i){
     for(unsigned j=1;j<(nhist_[1]-1);++j){
        result += integrand[i][j];
     }
  }
  // Edges
  for(unsigned i=1;i<(nhist_[0]-1);++i){
     result += 0.5*integrand[i][0];
     result += 0.5*integrand[i][nhist_[1]-1];
  }
  for(unsigned j=1;j<(nhist_[1]-1);++j){
     result += 0.5*integrand[0][j];
     result += 0.5*integrand[nhist_[0]-1][j];
  }
  // Corners
  result += 0.25*integrand[0][0];
  result += 0.25*integrand[nhist_[0]-1][0];
  result += 0.25*integrand[0][nhist_[1]-1];
  result += 0.25*integrand[nhist_[0]-1][nhist_[1]-1];
  // Spacing
  result *= delta[0]*delta[1];
  return result;
}

Vector PairOrientationalEntropyConformer::integrate(Matrix<Vector> integrand, vector<double> delta)const{
  // Trapezoid rule
  Vector result;
  for(unsigned i=1;i<(nhist_[0]-1);++i){
     for(unsigned j=1;j<(nhist_[1]-1);++j){
        result += integrand[i][j];
     }
  }
  // Edges
  for(unsigned i=1;i<(nhist_[0]-1);++i){
     result += 0.5*integrand[i][0];
     result += 0.5*integrand[i][nhist_[1]-1];
  }
  for(unsigned j=1;j<(nhist_[1]-1);++j){
     result += 0.5*integrand[0][j];
     result += 0.5*integrand[nhist_[0]-1][j];
  }
  // Corners
  result += 0.25*integrand[0][0];
  result += 0.25*integrand[nhist_[0]-1][0];
  result += 0.25*integrand[0][nhist_[1]-1];
  result += 0.25*integrand[nhist_[0]-1][nhist_[1]-1];
  // Spacing
  result *= delta[0]*delta[1];
  return result;
}

Tensor PairOrientationalEntropyConformer::integrate(Matrix<Tensor> integrand, vector<double> delta)const{
  // Trapezoid rule
  Tensor result;
  for(unsigned i=1;i<(nhist_[0]-1);++i){
     for(unsigned j=1;j<(nhist_[1]-1);++j){
        result += integrand[i][j];
     }
  }
  // Edges
  for(unsigned i=1;i<(nhist_[0]-1);++i){
     result += 0.5*integrand[i][0];
     result += 0.5*integrand[i][nhist_[1]-1];
  }
  for(unsigned j=1;j<(nhist_[1]-1);++j){
     result += 0.5*integrand[0][j];
     result += 0.5*integrand[nhist_[0]-1][j];
  }
  // Corners
  result += 0.25*integrand[0][0];
  result += 0.25*integrand[nhist_[0]-1][0];
  result += 0.25*integrand[0][nhist_[1]-1];
  result += 0.25*integrand[nhist_[0]-1][nhist_[1]-1];
  // Spacing
  result *= delta[0]*delta[1];
  return result;
}

void PairOrientationalEntropyConformer::outputGofr(Matrix<double> gofrAA, Matrix<double> gofrBB, Matrix<double> gofrAB) {
  for(unsigned i=0;i<nhist_[0];++i){
     for(unsigned j=0;j<nhist_[1];++j){
        gofrOfile.printField("r",x1[i]).printField("theta",x2[j]).printField("gofrAA",gofrAA[i][j]).printField("gofrBB",gofrBB[i][j]).printField("gofrAB",gofrAB[i][j]).printField();
     }
     gofrOfile.printf("\n");
  }
  gofrOfile.printf("\n");
  gofrOfile.printf("\n");
}

void PairOrientationalEntropyConformer::outputXYZ(std::vector<double> spin) {
  outputFileXYZ.printf("%d \n \n",spin.size());
  for(unsigned i=0;i<spin.size();++i){
    outputFileXYZ.printf("X %f %f %f %f \n", getPosition(i)[0]*10,getPosition(i)[1]*10,getPosition(i)[2]*10,spin[i]);
  }
}


}
}
