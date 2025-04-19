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
#include "core/ActionRegister.h"
#include "tools/NeighborListParallel.h"
#include "tools/Communicator.h"
#include "tools/Tools.h"
#include "tools/IFile.h"

#include <string>
#include <math.h>

using namespace std;

namespace PLMD{
namespace colvar{

class PairOrientationalEntropyTwoSpecies : public Colvar {
  bool pbc, serial, invalidateList, firsttime, doneigh;
  bool do_pairs;
  NeighborListParallel *nl;
  vector<AtomNumber> centera_lista,starta_lista,enda_lista;
  vector<AtomNumber> centerb_lista,startb_lista,endb_lista;
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
  // One body
  bool one_body;
  double deBroglie3;
public:
  explicit PairOrientationalEntropyTwoSpecies(const ActionOptions&);
  ~PairOrientationalEntropyTwoSpecies();
  virtual void calculate();
  virtual void prepare();
  static void registerKeywords( Keywords& keys );
};

PLUMED_REGISTER_ACTION(PairOrientationalEntropyTwoSpecies,"PAIR_ORIENTATIONAL_ENTROPY_TWO_SPECIES")

void PairOrientationalEntropyTwoSpecies::registerKeywords( Keywords& keys ){
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
  keys.add("atoms","CENTER_A","Center atoms");
  keys.add("atoms","START_A","Start point of vector defining orientation");
  keys.add("atoms","END_A","End point of vector defining orientation");
  keys.add("atoms","CENTER_B","Center atoms");
  keys.add("atoms","START_B","Start point of vector defining orientation");
  keys.add("atoms","END_B","End point of vector defining orientation");
  keys.add("compulsory","MAXR","1","Maximum distance for the radial distribution function ");
  keys.add("optional","NHIST","Number of bins in the rdf ");
  keys.add("compulsory","SIGMA","0.1","Width of gaussians ");
  keys.add("optional","REFERENCE_GOFR_FNAME","the name of the file with the reference g(r)");
  keys.add("optional","TEMPERATURE","Temperature in Kelvin. It is compulsory when keyword ONE_BODY is used");
  keys.add("optional","MASS","Mass in g/mol. It is compulsory when keyword ONE_BODY is used");
  keys.addFlag("ONE_BODY",false,"Add the one body term (S = 5/2 - ln(dens*deBroglie^3) ) to the entropy");
  keys.addOutputComponent("pairAA","INDIVIDUAL_PAIRS","Pair AA contribution to the multicomponent pair entropy");
  keys.addOutputComponent("pairAB","INDIVIDUAL_PAIRS","Pair AB contribution to the multicomponent pair entropy");
  keys.addOutputComponent("pairBB","INDIVIDUAL_PAIRS","Pair BB contribution to the multicomponent pair entropy");
  keys.addOutputComponent("full","INDIVIDUAL_PAIRS","Total multicomponent pair entropy");
}

PairOrientationalEntropyTwoSpecies::PairOrientationalEntropyTwoSpecies(const ActionOptions&ao):
PLUMED_COLVAR_INIT(ao),
pbc(true),
serial(false),
do_pairs(false),
invalidateList(true),
firsttime(true)
{

  parseFlag("SERIAL",serial);

  parseAtomList("CENTER_A",centera_lista);
  parseAtomList("START_A",starta_lista);
  parseAtomList("END_A",enda_lista);
  if(centera_lista.size()!=starta_lista.size()) error("Number of atoms in START_A must be equal to the number of atoms in CENTER_A");
  if(centera_lista.size()!=enda_lista.size()) error("Number of atoms in START_A must be equal to the number of atoms in CENTER_A");

  parseAtomList("CENTER_B",centerb_lista);
  parseAtomList("START_B",startb_lista);
  parseAtomList("END_B",endb_lista);
  if(centerb_lista.size()!=startb_lista.size()) error("Number of atoms in START_B must be equal to the number of atoms in CENTER_A");
  if(centerb_lista.size()!=endb_lista.size()) error("Number of atoms in START_B must be equal to the number of atoms in CENTER_A");

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

  checkRead();

  // Neighbor lists
  if (doneigh) {
    vector<AtomNumber> center_lista;
    center_lista.reserve ( centera_lista.size() + centerb_lista.size()  );
    center_lista.insert (center_lista.end(), centera_lista.begin(), centera_lista.end() );
    center_lista.insert (center_lista.end(), centerb_lista.begin(), centerb_lista.end() );
    nl= new NeighborListParallel(center_lista,pbc,getPbc(),comm,log,nl_cut,nl_full_list,nl_st,nl_skin);
    log.printf("  using neighbor lists with\n");
    log.printf("  cutoff %f, and skin %f\n",nl_cut,nl_skin);
    if(nl_st>=0){
      log.printf("  update every %d steps\n",nl_st);
    } else {
      log.printf("  checking every step for dangerous builds and rebuilding as needed\n");
    }
  }
  atomsToRequest.reserve ( 3*centera_lista.size() + 3*centerb_lista.size()  );
  atomsToRequest.insert (atomsToRequest.end(), centera_lista.begin(), centera_lista.end() );
  atomsToRequest.insert (atomsToRequest.end(), centerb_lista.begin(), centerb_lista.end() );
  atomsToRequest.insert (atomsToRequest.end(), starta_lista.begin(), starta_lista.end() );
  atomsToRequest.insert (atomsToRequest.end(), startb_lista.begin(), startb_lista.end() );
  atomsToRequest.insert (atomsToRequest.end(), enda_lista.begin(), enda_lista.end() );
  atomsToRequest.insert (atomsToRequest.end(), endb_lista.begin(), endb_lista.end() );
  requestAtoms(atomsToRequest);
  log.printf(" atoms to request are %d \n",atomsToRequest.size());

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

PairOrientationalEntropyTwoSpecies::~PairOrientationalEntropyTwoSpecies(){
  if (doneigh) {
     nl->printStats();
     delete nl;
  }
  if (doOutputGofr) gofrOfile.close();
}

void PairOrientationalEntropyTwoSpecies::prepare(){
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
void PairOrientationalEntropyTwoSpecies::calculate()
{
  // Number of molecules
  double number_A_molecules = centera_lista.size() ;
  double number_B_molecules = centerb_lista.size() ;
  double number_molecules = number_A_molecules + number_B_molecules ;
  // Data structures
  Matrix<double> gofrAA(nhist_[0],nhist_[1]);
  Matrix<double> gofrBB(nhist_[0],nhist_[1]);
  Matrix<double> gofrAB(nhist_[0],nhist_[1]);
  // Some of these vectors are too long...
  vector<Vector> gofrPrimeCenterAA(nhist_[0]*nhist_[1]*number_molecules);
  vector<Vector> gofrPrimeCenterBB(nhist_[0]*nhist_[1]*number_molecules);
  vector<Vector> gofrPrimeCenterAB(nhist_[0]*nhist_[1]*number_molecules);
  vector<Vector> gofrPrimeStartAA(nhist_[0]*nhist_[1]*number_molecules);
  vector<Vector> gofrPrimeStartBB(nhist_[0]*nhist_[1]*number_molecules);
  vector<Vector> gofrPrimeStartAB(nhist_[0]*nhist_[1]*number_molecules);
  vector<Vector> gofrPrimeEndAA(nhist_[0]*nhist_[1]*number_molecules);
  vector<Vector> gofrPrimeEndBB(nhist_[0]*nhist_[1]*number_molecules);
  vector<Vector> gofrPrimeEndAB(nhist_[0]*nhist_[1]*number_molecules);
  Matrix<Tensor> gofrVirialAA(nhist_[0],nhist_[1]);
  Matrix<Tensor> gofrVirialBB(nhist_[0],nhist_[1]);
  Matrix<Tensor> gofrVirialAB(nhist_[0],nhist_[1]);
  // Calculate volume and density
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
      vector<Vector> centerPositions(getPositions().begin(),getPositions().begin() + number_molecules);
      nl->update(centerPositions);
    }
    for(unsigned int i=0;i<nl->getNumberOfLocalAtoms();i+=1) {
       unsigned index=nl->getIndexOfLocalAtom(i);
       unsigned atom1_mol1;
       unsigned atom2_mol1;
       atom1_mol1=index+1*number_molecules;
       atom2_mol1=index+2*number_molecules;
       std::vector<unsigned> neighbors=nl->getNeighbors(index);
       Vector position_index=getPosition(index);
       Vector mol_vector1=pbcDistance(getPosition(atom1_mol1),getPosition(atom2_mol1)); 
       double norm_v1 = std::sqrt(mol_vector1[0]*mol_vector1[0]+mol_vector1[1]*mol_vector1[1]+mol_vector1[2]*mol_vector1[2]);
       double inv_v1=1./norm_v1;
       double inv_v1_sqr=inv_v1*inv_v1;
       // Loop over neighbors
       for(unsigned int j=0;j<neighbors.size();j+=1) {  
          unsigned neighbor=neighbors[j];
          if(getAbsoluteIndex(index)==getAbsoluteIndex(neighbor)) continue;
          Vector distance;
          if(pbc){
           distance=pbcDistance(position_index,getPosition(neighbor));
          } else {
           distance=delta(position_index,getPosition(neighbor));
          }
          double d2;
          if ( (d2=distance[0]*distance[0])<rcut2 && (d2+=distance[1]*distance[1])<rcut2 && (d2+=distance[2]*distance[2])<rcut2) {
             double distanceModulo=std::sqrt(d2);
             Vector distance_versor = distance / distanceModulo;
             unsigned bin=std::floor(distanceModulo/deltar);
             unsigned atom1_mol2;
             unsigned atom2_mol2;
             atom1_mol2=neighbor+1*number_molecules;
             atom2_mol2=neighbor+2*number_molecules;
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
                  vector<double> dfunc(2);
                  double kernel_val;
                  // Three cases
                  if (index<number_A_molecules && neighbor<number_A_molecules) {
                  // AA
                    if (l==(nhist_[1]-1) || l==0) {
                      kernel_val = kernel(pos,2.*invNormKernelAA,dfunc);
                    } else {
                      kernel_val = kernel(pos,invNormKernelAA,dfunc);
                    }
                    gofrAA[k][h] += kernel_val/2.; // Divided by 2 since it will be summed twice
                    Vector value1 = dfunc[0]*distance_versor;
                    Vector value2_mol1 = dfunc[1]*der_mol1;
                    gofrPrimeCenterAA[index*nhist1_nhist2_+k*nhist_[1]+h] += value1;
                    gofrPrimeStartAA[index*nhist1_nhist2_+k*nhist_[1]+h] +=  value2_mol1;
                    gofrPrimeEndAA[index*nhist1_nhist2_+k*nhist_[1]+h] -=  value2_mol1;
                    Tensor vv1(value1, distance);
                    Tensor vv2_mol1(value2_mol1, mol_vector1);
                    gofrVirialAA[k][h] += (vv1/2.+vv2_mol1);
                  } else if (index>number_A_molecules && neighbor>number_A_molecules) {
                  // BB
                    if (l==(nhist_[1]-1) || l==0) {
                      kernel_val = kernel(pos,2.*invNormKernelBB,dfunc);
                    } else {
                      kernel_val = kernel(pos,invNormKernelBB,dfunc);
                    }
                    gofrBB[k][h] += kernel_val/2.; // Divided by 2 since it will be summed twice
                    Vector value1 = dfunc[0]*distance_versor;
                    Vector value2_mol1 = dfunc[1]*der_mol1;
                    gofrPrimeCenterBB[index*nhist1_nhist2_+k*nhist_[1]+h] += value1;
                    gofrPrimeStartBB[index*nhist1_nhist2_+k*nhist_[1]+h] +=  value2_mol1;
                    gofrPrimeEndBB[index*nhist1_nhist2_+k*nhist_[1]+h] -=  value2_mol1;
                    Tensor vv1(value1, distance);
                    Tensor vv2_mol1(value2_mol1, mol_vector1);
                    gofrVirialBB[k][h] += (vv1/2.+vv2_mol1);
                  } else {
                  // AB
                    if (l==(nhist_[1]-1) || l==0) {
                      kernel_val = kernel(pos,2.*invNormKernelAB,dfunc);
                    } else {
                      kernel_val = kernel(pos,invNormKernelAB,dfunc);
                    }
                    gofrAB[k][h] += kernel_val/2.; // Divided by 2 since it will be summed twice
                    Vector value1 = dfunc[0]*distance_versor;
                    Vector value2_mol1 = dfunc[1]*der_mol1;
                    gofrPrimeCenterAB[index*nhist1_nhist2_+k*nhist_[1]+h] += value1;
                    gofrPrimeStartAB[index*nhist1_nhist2_+k*nhist_[1]+h] +=  value2_mol1;
                    gofrPrimeEndAB[index*nhist1_nhist2_+k*nhist_[1]+h] -=  value2_mol1;
                    Tensor vv1(value1, distance);
                    Tensor vv2_mol1(value2_mol1, mol_vector1);
                    gofrVirialAB[k][h] += (vv1/2.+vv2_mol1);
                  }
	       }
             }
           }
        }
     }
  } else {
    for(unsigned int i=rank;i<number_molecules;i+=stride) {
       unsigned index=i;
       unsigned atom1_mol1;
       unsigned atom2_mol1;
       atom1_mol1=index+1*number_molecules;
       atom2_mol1=index+2*number_molecules;
       Vector position_index=getPosition(index);
       Vector mol_vector1=pbcDistance(getPosition(atom1_mol1),getPosition(atom2_mol1)); 
       double norm_v1 = std::sqrt(mol_vector1[0]*mol_vector1[0]+mol_vector1[1]*mol_vector1[1]+mol_vector1[2]*mol_vector1[2]);
       double inv_v1=1./norm_v1;
       double inv_v1_sqr=inv_v1*inv_v1;
       // Loop over all molecules
       for(unsigned int j=0;j<number_molecules;j+=1) {  
          unsigned neighbor=j;
          if(getAbsoluteIndex(index)==getAbsoluteIndex(neighbor)) continue;
          Vector distance;
          if(pbc){
           distance=pbcDistance(position_index,getPosition(neighbor));
          } else {
           distance=delta(position_index,getPosition(neighbor));
          }
          double d2;
          if ( (d2=distance[0]*distance[0])<rcut2 && (d2+=distance[1]*distance[1])<rcut2 && (d2+=distance[2]*distance[2])<rcut2) {
             double distanceModulo=std::sqrt(d2);
             Vector distance_versor = distance / distanceModulo;
             unsigned bin=std::floor(distanceModulo/deltar);
             unsigned atom1_mol2;
             unsigned atom2_mol2;
             atom1_mol2=neighbor+1*number_molecules;
             atom2_mol2=neighbor+2*number_molecules;
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
                  vector<double> dfunc(2);
                  double kernel_val;
                  // Three cases
                  if (index<number_A_molecules && neighbor<number_A_molecules) {
                  // AA
                    if (l==(nhist_[1]-1) || l==0) {
                      kernel_val = kernel(pos,2.*invNormKernelAA,dfunc);
                    } else {
                      kernel_val = kernel(pos,invNormKernelAA,dfunc);
                    }
                    gofrAA[k][h] += kernel_val/2.; // Divided by 2 since it will be summed twice
                    Vector value1 = dfunc[0]*distance_versor;
                    Vector value2_mol1 = dfunc[1]*der_mol1;
                    gofrPrimeCenterAA[index*nhist1_nhist2_+k*nhist_[1]+h] += value1;
                    gofrPrimeStartAA[index*nhist1_nhist2_+k*nhist_[1]+h] +=  value2_mol1;
                    gofrPrimeEndAA[index*nhist1_nhist2_+k*nhist_[1]+h] -=  value2_mol1;
                    Tensor vv1(value1, distance);
                    Tensor vv2_mol1(value2_mol1, mol_vector1);
                    gofrVirialAA[k][h] += (vv1/2.+vv2_mol1);
                  } else if (index>number_A_molecules && neighbor>number_A_molecules) {
                  // BB
                    if (l==(nhist_[1]-1) || l==0) {
                      kernel_val = kernel(pos,2.*invNormKernelBB,dfunc);
                    } else {
                      kernel_val = kernel(pos,invNormKernelBB,dfunc);
                    }
                    gofrBB[k][h] += kernel_val/2.; // Divided by 2 since it will be summed twice
                    Vector value1 = dfunc[0]*distance_versor;
                    Vector value2_mol1 = dfunc[1]*der_mol1;
                    gofrPrimeCenterBB[index*nhist1_nhist2_+k*nhist_[1]+h] += value1;
                    gofrPrimeStartBB[index*nhist1_nhist2_+k*nhist_[1]+h] +=  value2_mol1;
                    gofrPrimeEndBB[index*nhist1_nhist2_+k*nhist_[1]+h] -=  value2_mol1;
                    Tensor vv1(value1, distance);
                    Tensor vv2_mol1(value2_mol1, mol_vector1);
                    gofrVirialBB[k][h] += (vv1/2.+vv2_mol1);
                  } else {
                  // AB
                    if (l==(nhist_[1]-1) || l==0) {
                      kernel_val = kernel(pos,2.*invNormKernelAB,dfunc);
                    } else {
                      kernel_val = kernel(pos,invNormKernelAB,dfunc);
                    }
                    gofrAB[k][h] += kernel_val/2.; // Divided by 2 since it will be summed twice
                    Vector value1 = dfunc[0]*distance_versor;
                    Vector value2_mol1 = dfunc[1]*der_mol1;
                    gofrPrimeCenterAB[index*nhist1_nhist2_+k*nhist_[1]+h] += value1;
                    gofrPrimeStartAB[index*nhist1_nhist2_+k*nhist_[1]+h] +=  value2_mol1;
                    gofrPrimeEndAB[index*nhist1_nhist2_+k*nhist_[1]+h] -=  value2_mol1;
                    Tensor vv1(value1, distance);
                    Tensor vv2_mol1(value2_mol1, mol_vector1);
                    gofrVirialAB[k][h] += (vv1/2.+vv2_mol1);
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
  Tensor virialAA, virialBB, virialAB, virialOneBody;
  if (!doNotCalculateDerivatives() ) {
    if (doneigh) {
       for(unsigned int k=0;k<nl->getNumberOfLocalAtoms();k+=1) {
         unsigned index=nl->getIndexOfLocalAtom(k);
         // Center atom
         unsigned start_atom=index+number_molecules;
         unsigned end_atom=index+2*number_molecules;
         Matrix<Vector> integrandDerivativesAA(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesBB(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesAB(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesStartAA(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesStartBB(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesStartAB(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesEndAA(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesEndBB(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesEndAB(nhist_[0],nhist_[1]);
         for(unsigned i=0;i<nhist_[0];++i){
           for(unsigned j=0;j<nhist_[1];++j){
             if (gofrAA[i][j]>1.e-10) {
               integrandDerivativesAA[i][j] = gofrPrimeCenterAA[index*nhist1_nhist2_+i*nhist_[1]+j]*logGofrAAx1sqr[i][j];
               integrandDerivativesStartAA[i][j] = gofrPrimeStartAA[index*nhist1_nhist2_+i*nhist_[1]+j]*logGofrAAx1sqr[i][j];
               integrandDerivativesEndAA[i][j] = gofrPrimeEndAA[index*nhist1_nhist2_+i*nhist_[1]+j]*logGofrAAx1sqr[i][j];
             }
             if (gofrBB[i][j]>1.e-10) {
               integrandDerivativesBB[i][j] = gofrPrimeCenterBB[index*nhist1_nhist2_+i*nhist_[1]+j]*logGofrBBx1sqr[i][j];
               integrandDerivativesStartBB[i][j] = gofrPrimeStartBB[index*nhist1_nhist2_+i*nhist_[1]+j]*logGofrBBx1sqr[i][j];
               integrandDerivativesEndBB[i][j] = gofrPrimeEndBB[index*nhist1_nhist2_+i*nhist_[1]+j]*logGofrBBx1sqr[i][j];
             }
             if (gofrAB[i][j]>1.e-10) {
               integrandDerivativesAB[i][j] = gofrPrimeCenterAB[index*nhist1_nhist2_+i*nhist_[1]+j]*logGofrABx1sqr[i][j];
               integrandDerivativesStartAB[i][j] = gofrPrimeStartAB[index*nhist1_nhist2_+i*nhist_[1]+j]*logGofrABx1sqr[i][j];
               integrandDerivativesEndAB[i][j] = gofrPrimeEndAB[index*nhist1_nhist2_+i*nhist_[1]+j]*logGofrABx1sqr[i][j];
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
       }
    } else {
       for(unsigned int k=rank;k<number_molecules;k+=stride) {
         // Center atom
         unsigned start_atom=k+number_molecules;
         unsigned end_atom=k+2*number_molecules;
         Matrix<Vector> integrandDerivativesAA(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesBB(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesAB(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesStartAA(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesStartBB(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesStartAB(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesEndAA(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesEndBB(nhist_[0],nhist_[1]);
         Matrix<Vector> integrandDerivativesEndAB(nhist_[0],nhist_[1]);
         for(unsigned i=0;i<nhist_[0];++i){
           for(unsigned j=0;j<nhist_[1];++j){
             if (gofrAA[i][j]>1.e-10) {
               integrandDerivativesAA[i][j] = gofrPrimeCenterAA[k*nhist1_nhist2_+i*nhist_[1]+j]*logGofrAAx1sqr[i][j];
               integrandDerivativesStartAA[i][j] = gofrPrimeStartAA[k*nhist1_nhist2_+i*nhist_[1]+j]*logGofrAAx1sqr[i][j];
               integrandDerivativesEndAA[i][j] = gofrPrimeEndAA[k*nhist1_nhist2_+i*nhist_[1]+j]*logGofrAAx1sqr[i][j];
             }
             if (gofrBB[i][j]>1.e-10) {
               integrandDerivativesBB[i][j] = gofrPrimeCenterBB[k*nhist1_nhist2_+i*nhist_[1]+j]*logGofrBBx1sqr[i][j];
               integrandDerivativesStartBB[i][j] = gofrPrimeStartBB[k*nhist1_nhist2_+i*nhist_[1]+j]*logGofrBBx1sqr[i][j];
               integrandDerivativesEndBB[i][j] = gofrPrimeEndBB[k*nhist1_nhist2_+i*nhist_[1]+j]*logGofrBBx1sqr[i][j];
             }
             if (gofrAB[i][j]>1.e-10) {
               integrandDerivativesAB[i][j] = gofrPrimeCenterAB[k*nhist1_nhist2_+i*nhist_[1]+j]*logGofrABx1sqr[i][j];
               integrandDerivativesStartAB[i][j] = gofrPrimeStartAB[k*nhist1_nhist2_+i*nhist_[1]+j]*logGofrABx1sqr[i][j];
               integrandDerivativesEndAB[i][j] = gofrPrimeEndAB[k*nhist1_nhist2_+i*nhist_[1]+j]*logGofrABx1sqr[i][j];
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
       }
    }
    if(!serial){
      comm.Sum(derivAA);
      comm.Sum(derivBB);
      comm.Sum(derivAB);
    }
    // Virial of positions
    // Construct virial integrand
    Matrix<Tensor> integrandVirialAA(nhist_[0],nhist_[1]);
    Matrix<Tensor> integrandVirialBB(nhist_[0],nhist_[1]);
    Matrix<Tensor> integrandVirialAB(nhist_[0],nhist_[1]);
    for(unsigned i=0;i<nhist_[0];++i){
       for(unsigned j=0;j<nhist_[1];++j){
          if (gofrAA[i][j]>1.e-10) {
             integrandVirialAA[i][j] = gofrVirialAA[i][j]*logGofrAAx1sqr[i][j];
          }
          if (gofrBB[i][j]>1.e-10) {
             integrandVirialBB[i][j] = gofrVirialBB[i][j]*logGofrBBx1sqr[i][j];
          }
          if (gofrAB[i][j]>1.e-10) {
             integrandVirialAB[i][j] = gofrVirialAB[i][j]*logGofrABx1sqr[i][j];
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
      for(unsigned j=0;j<getNumberOfAtoms();++j) setAtomsDerivatives (full,j,derivAA[j]+derivAB[j]+derivBB[j]);
      //for(unsigned j=0;j<getNumberOfAtoms();++j) log.printf("derivAA x y z %f %f %f \n",derivAA[j][0],derivAA[j][1],derivAA[j][2]);
      setBoxDerivatives  (pairAA,virialAA);
      setBoxDerivatives  (pairBB,virialBB);
      setBoxDerivatives  (pairAB,virialAB);
      setBoxDerivatives  (full,virialAA+virialBB+virialAB+virialOneBody);
    }
  } else {
    setValue(pairAAvalue+pairABvalue+pairBBvalue+oneBody);
    if (!doNotCalculateDerivatives() ) {
      for(unsigned j=0;j<getNumberOfAtoms();++j) setAtomsDerivatives (j,derivAA[j]+derivAB[j]+derivBB[j]);
      setBoxDerivatives  (virialAA+virialBB+virialAB+virialOneBody);
    }
  } 
}

double PairOrientationalEntropyTwoSpecies::kernel(vector<double> distance, double invNormKernel, vector<double>&der)const{
  // Gaussian function and derivative
  double result = invNormKernel*std::exp(-distance[0]*distance[0]/twoSigma1Sqr-distance[1]*distance[1]/twoSigma2Sqr) ;
  der[0] = -distance[0]*result/sigma1Sqr;
  der[1] = -distance[1]*result/sigma2Sqr;
  return result;
}

double PairOrientationalEntropyTwoSpecies::integrate(Matrix<double> integrand, vector<double> delta)const{
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

Vector PairOrientationalEntropyTwoSpecies::integrate(Matrix<Vector> integrand, vector<double> delta)const{
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

Tensor PairOrientationalEntropyTwoSpecies::integrate(Matrix<Tensor> integrand, vector<double> delta)const{
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

void PairOrientationalEntropyTwoSpecies::outputGofr(Matrix<double> gofrAA, Matrix<double> gofrBB, Matrix<double> gofrAB) {
  for(unsigned i=0;i<nhist_[0];++i){
     for(unsigned j=0;j<nhist_[1];++j){
        gofrOfile.printField("r",x1[i]).printField("theta",x2[j]).printField("gofrAA",gofrAA[i][j]).printField("gofrBB",gofrBB[i][j]).printField("gofrAB",gofrAB[i][j]).printField();
     }
     gofrOfile.printf("\n");
  }
  gofrOfile.printf("\n");
  gofrOfile.printf("\n");
}

}
}
