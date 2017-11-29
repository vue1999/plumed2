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

#include <string>
#include <math.h>

using namespace std;

namespace PLMD{
namespace colvar{

//+PLUMEDOC COLVAR PAIRENTROPY
/*
Calculate the global pair entropy using the expression:
\f[
s=-2\pi\rho k_B \int\limits_0^{r_{\mathrm{max}}} \left [ g(r) \ln g(r) - g(r) + 1 \right ] r^2 dr .
\f]
where \f$ g(r) $\f is the pair distribution function and \f$ r_{\mathrm{max}} $\f is a cutoff in the integration (MAXR).
For the integration the interval from 0 to  \f$ r_{\mathrm{max}} $\f is partitioned in NHIST equal intervals. 
To make the calculation of \f$ g(r) $\f differentiable, the following function is used:
\f[
g(r) = \frac{1}{4 \pi \rho r^2} \sum\limits_{j} \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-(r-r_{ij})^2/(2\sigma^2)} ,
\f]
where \f$ \rho $\f is the density and \f$ sigma $\f is a broadening parameter (SIGMA).  
\par Example)
The following input tells plumed to calculate the pair entropy of atoms 1-250 with themselves.
\verbatim
PAIRENTROPY ...
 LABEL=s2
 GROUPA=1-250
 MAXR=0.65
 SIGMA=0.025
 NHIST=100
 NLIST
 NL_CUTOFF=0.75
 NL_STRIDE=10
... PAIRENTROPY
\endverbatim
*/
//+ENDPLUMEDOC

class PairOrientationalEntropy : public Colvar {
  bool pbc, serial, invalidateList, firsttime, doneigh;
  NeighborListParallel *nl;
  vector<AtomNumber> center_lista,start_lista,end_lista;
  std::vector<PLMD::AtomNumber> atomsToRequest;
  double maxr;
  vector<int> nhist_;
  int nhist1_nhist2_;
  vector<double> sigma_;
  double rcut2;
  double invTwoPiSigma1Sigma2, sigma1Sqr, sigma2Sqr, twoSigma1Sqr,twoSigma2Sqr, invNormKernel;
  double deltar, deltaAngle, deltaCosAngle;
  unsigned deltaBin, deltaBinAngle;
  // Integration routines
  double integrate(Matrix<double> integrand, vector<double> delta)const;
  Vector integrate(Matrix<Vector> integrand, vector<double> delta)const;
  Tensor integrate(Matrix<Tensor> integrand, vector<double> delta)const;
  vector<double> x1, x2, x1sqr, x2sqr;
  // Kernel to calculate g(r)
  double kernel(vector<double> distance, vector<double>&der)const;
  // Output gofr and integrand
  void outputGofr(Matrix<double> gofr, const char* fileName);
  void outputIntegrand(vector<double> integrand);
  int outputStride;
  bool doOutputGofr, doOutputIntegrand;
  // Average gofr
  Matrix<double> avgGofr;
  unsigned iteration;
  bool doAverageGofr;
public:
  explicit PairOrientationalEntropy(const ActionOptions&);
  ~PairOrientationalEntropy();
// active methods:
  virtual void calculate();
  virtual void prepare();
  static void registerKeywords( Keywords& keys );
};

PLUMED_REGISTER_ACTION(PairOrientationalEntropy,"PAIR_ORIENTATIONAL_ENTROPY")

void PairOrientationalEntropy::registerKeywords( Keywords& keys ){
  Colvar::registerKeywords(keys);
  keys.addFlag("SERIAL",false,"Perform the calculation in serial - for debug purpose");
  keys.addFlag("NLIST",false,"Use a neighbour list to speed up the calculation");
  keys.addFlag("OUTPUT_GOFR",false,"Output g(r)");
  keys.addFlag("AVERAGE_GOFR",false,"Average g(r) over time");
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
}

PairOrientationalEntropy::PairOrientationalEntropy(const ActionOptions&ao):
PLUMED_COLVAR_INIT(ao),
pbc(true),
serial(false),
invalidateList(true),
firsttime(true)
{

  parseFlag("SERIAL",serial);

  parseAtomList("CENTER",center_lista);
  parseAtomList("START",start_lista);
  parseAtomList("END",end_lista);
  if(center_lista.size()!=start_lista.size()) error("Number of atoms in START must be equal to the number of atoms in CENTER");
  if(center_lista.size()!=end_lista.size()) error("Number of atoms in START must be equal to the number of atoms in CENTER");

  bool nopbc=!pbc;
  parseFlag("NOPBC",nopbc);
  pbc=!nopbc;

// neighbor list stuff
  doneigh=false;
  bool nl_reduced_list=false;
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

  addValueWithDerivatives(); setNotPeriodic();

  parse("MAXR",maxr);
  log.printf("  Integration in the interval from 0. to %f \n", maxr );

  parseVector("SIGMA",sigma_);
  if(sigma_.size() != 2) error("SIGMA keyword takes two input values");
  log.printf("  The pair distribution function is calculated with a Gaussian kernel with deviations %f and %f \n", sigma_[0], sigma_[1]);
  double rcut = maxr + 2*sigma_[0];  // 2*sigma is hard coded
  rcut2 = rcut*rcut;
  if(doneigh){
    if(nl_cut<rcut) error("NL_CUTOFF should be larger than MAXR + 2*SIGMA");
    nl_skin=nl_cut-rcut;
  }

  parseVector("NHIST",nhist_);
  if (nhist_.size()<1) {
     nhist_.resize(2);
     nhist_[0]=ceil(maxr/sigma_[0]) + 1; // Default value
     nhist_[1]=ceil(2./sigma_[1]) + 1; // Default value
  }
  if(nhist_.size() != 2) error("NHIST keyword takes two input values");
  nhist1_nhist2_=nhist_[0]*nhist_[1];
  log.printf("  The r-theta space is discretized using a grid of size %u times %u. \n", nhist_[0], nhist_[1] );
  log.printf("  The integration is performed with the trapezoid rule. \n");

  doOutputGofr=false;
  parseFlag("OUTPUT_GOFR",doOutputGofr);
  if (doOutputGofr) { 
     log.printf("  The g(r) will be written to a file \n.");
  }
  doOutputIntegrand=false;
  parseFlag("OUTPUT_INTEGRAND",doOutputIntegrand);
  if (doOutputIntegrand) {
     log.printf("  The integrand will be written to a file \n.");
  }
  outputStride=1;
  parse("OUTPUT_STRIDE",outputStride);
  if (outputStride!=1 && !doOutputGofr && !doOutputIntegrand) error("Cannot specify OUTPUT_STRIDE if OUTPUT_GOFR or OUTPUT_INTEGRAND not used");
  if (outputStride<1) error("The output stride specified with OUTPUT_STRIDE must be greater than or equal to one.");
  if (outputStride>1) log.printf("  The output stride to write g(r) or the integrand is %d \n", outputStride);

  doAverageGofr=false;
  parseFlag("AVERAGE_GOFR",doAverageGofr);
  if (doAverageGofr) {
     iteration = 1;
     log.printf("  The g(r) will be averaged over all frames");
     avgGofr.resize(nhist_[0],nhist_[1]);
  }

  checkRead();

  // Neighbor lists
  if (doneigh) {
    nl= new NeighborListParallel(center_lista,pbc,getPbc(),comm,log,nl_cut,nl_reduced_list,nl_st,nl_skin);
    log.printf("  using neighbor lists with\n");
    log.printf("  cutoff %f, and skin %f\n",nl_cut,nl_skin);
    if(nl_st>=0){
      log.printf("  update every %d steps\n",nl_st);
    } else {
      log.printf("  checking every step for dangerous builds and rebuilding as needed\n");
    }
  }
  atomsToRequest.reserve ( center_lista.size() + start_lista.size() + end_lista.size() );
  atomsToRequest.insert (atomsToRequest.end(), center_lista.begin(), center_lista.end() );
  atomsToRequest.insert (atomsToRequest.end(), start_lista.begin(), start_lista.end() );
  atomsToRequest.insert (atomsToRequest.end(), end_lista.begin(), end_lista.end() );
  requestAtoms(atomsToRequest);

  // Define heavily used expressions
  invTwoPiSigma1Sigma2 = (1./(2.*pi*sigma_[0]*sigma_[1]));
  sigma1Sqr = sigma_[0]*sigma_[0];
  sigma2Sqr = sigma_[1]*sigma_[1];
  twoSigma1Sqr = 2*sigma_[0]*sigma_[0];
  twoSigma2Sqr = 2*sigma_[1]*sigma_[1];
  deltar=maxr/(nhist_[0]-1);
  deltaCosAngle=2./(nhist_[1]-1);
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
     x2[i]=-1+deltaCosAngle*i;
     x2sqr[i]=x2[i]*x2[i];
  }
}

PairOrientationalEntropy::~PairOrientationalEntropy(){
  if (doneigh) {
     nl->printStats();
     delete nl;
  }
}

void PairOrientationalEntropy::prepare(){
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
void PairOrientationalEntropy::calculate()
{
  //clock_t begin_time = clock();
  // Define output quantities
  double pairEntropy;
  vector<Vector> deriv(getNumberOfAtoms());
  Tensor virial;
  // Define intermediate quantities
  Matrix<double> gofr(nhist_[0],nhist_[1]);
  //vector<vector<vector<Vector> > > gofrPrime(getNumberOfAtoms(),vector<vector<Vector> >(nhist_[1],vector <Vector>(nhist_[0])));
  vector<Vector> gofrPrimeCenter(nhist_[0]*nhist_[1]*center_lista.size());
  vector<Vector> gofrPrimeStart(nhist_[0]*nhist_[1]*center_lista.size());
  vector<Vector> gofrPrimeEnd(nhist_[0]*nhist_[1]*center_lista.size());
  Matrix<Tensor> gofrVirial(nhist_[0],nhist_[1]);
  // Calculate volume and density
  double volume=getBox().determinant();
  double density=center_lista.size()/volume;
  // Normalization of g(r)
  double normConstantBase = 2*pi*center_lista.size()*density;
  // Take into account "volume" of angles
  double volumeOfAngles = 2.;
  normConstantBase /= volumeOfAngles;
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
    for(unsigned int i=rank;i<center_lista.size();i+=stride) {
       // Loop over neighbors
       std::vector<unsigned> neighbors;
       neighbors=nl->getNeighbors(i);
       for(unsigned int j=0;j<neighbors.size();j+=1) {  
          double d2;
          vector<double> dfunc(2);
          Vector distance;
          Vector distance_versor;
          if(getAbsoluteIndex(i)==getAbsoluteIndex(neighbors[j])) continue;
          if(pbc){
             distance=pbcDistance(getPosition(i),getPosition(neighbors[j]));
          } else {
             distance=delta(getPosition(i),getPosition(neighbors[j]));
          }
          if ( (d2=distance[0]*distance[0])<rcut2 && (d2+=distance[1]*distance[1])<rcut2 && (d2+=distance[2]*distance[2])<rcut2) {
             double distanceModulo=std::sqrt(d2);
             Vector distance_versor = distance / distanceModulo;
             unsigned bin=std::floor(distanceModulo/deltar);
             unsigned atom1_mol1=i+center_lista.size();
             unsigned atom2_mol1=i+center_lista.size()+start_lista.size();
             unsigned atom1_mol2=neighbors[j]+center_lista.size();
             unsigned atom2_mol2=neighbors[j]+center_lista.size()+start_lista.size();
             Vector mol_vector1=pbcDistance(getPosition(atom1_mol1),getPosition(atom2_mol1)); 
             Vector mol_vector2=pbcDistance(getPosition(atom1_mol2),getPosition(atom2_mol2));
             double norm_v1 = std::sqrt(mol_vector1[0]*mol_vector1[0]+mol_vector1[1]*mol_vector1[1]+mol_vector1[2]*mol_vector1[2]);
             double norm_v2 = std::sqrt(mol_vector2[0]*mol_vector2[0]+mol_vector2[1]*mol_vector2[1]+mol_vector2[2]*mol_vector2[2]);
             double inv_v1=1./norm_v1;
             double inv_v2=1./norm_v2;
             double inv_v1_sqr=inv_v1*inv_v1;
             double inv_v1_inv_v2=inv_v1*inv_v2;
             double cosAngle=dotProduct(mol_vector1,mol_vector2)*inv_v1*inv_v2;
             unsigned binAngle=std::floor((cosAngle+1.)/deltaCosAngle);
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
             Vector der_mol1=mol_vector2*inv_v1_inv_v2-cosAngle*mol_vector1*inv_v1_sqr;
             for(int k=minBin;k<maxBin+1;k+=1) {
               invNormKernel=invTwoPiSigma1Sigma2/(normConstantBase*x1sqr[k]);
               vector<double> pos(2);
               pos[0]=x1[k]-distanceModulo;
               for(int l=minBinAngle;l<maxBinAngle+1;l+=1) {
                  double theta=-1+deltaCosAngle*l;
                  pos[1]=theta-cosAngle;
                  // Include periodic effects
                  int h;
                  if (l<0) {
                     h=-l;
                  } else if (l>(nhist_[1]-1)) {
                     h=2*nhist_[1]-l-2;
                  } else {
                     h=l;
                  }
                  Vector value1;
                  Vector value2_mol1;
                  Vector value2_mol2;
                  if (l==(nhist_[1]-1) || l==0) {
                     gofr[k][h] += kernel(pos, dfunc);
                     value1 = 2*dfunc[0]*distance_versor;
                     value2_mol1 = 2*dfunc[1]*der_mol1;
                  } else {
                     gofr[k][h] += kernel(pos, dfunc)/2.;
                     value1 = dfunc[0]*distance_versor;
                     value2_mol1 = dfunc[1]*der_mol1;
                  }
                  gofrPrimeCenter[i*nhist1_nhist2_+k*nhist_[1]+h] += value1;
                  gofrPrimeStart[i*nhist1_nhist2_+k*nhist_[1]+h] +=  value2_mol1;
                  gofrPrimeEnd[i*nhist1_nhist2_+k*nhist_[1]+h] -=  value2_mol1;
                  Tensor vv1(value1, distance);
                  Tensor vv2_mol1(value2_mol1, mol_vector1);
                  gofrVirial[k][h] += vv1/2.+vv2_mol1;
               }
             }
           }
        }
     }
  } else {
    for(unsigned int i=rank;i<center_lista.size();i+=stride) {
      for(unsigned int j=0;j<center_lista.size();j+=1) {
        double d2;
        vector<double> dfunc(2);
        Vector distance;
        Vector distance_versor;
        if(getAbsoluteIndex(i)==getAbsoluteIndex(j)) continue;
        if(pbc){
         distance=pbcDistance(getPosition(i),getPosition(j));
        } else {
         distance=delta(getPosition(i),getPosition(j));
        }
        if ( (d2=distance[0]*distance[0])<rcut2 && (d2+=distance[1]*distance[1])<rcut2 && (d2+=distance[2]*distance[2])<rcut2) {
          double distanceModulo=std::sqrt(d2);
          Vector distance_versor = distance / distanceModulo;
          unsigned bin=std::floor(distanceModulo/deltar);
          unsigned atom1_mol1=i+center_lista.size();
          unsigned atom2_mol1=i+center_lista.size()+start_lista.size();
          unsigned atom1_mol2=j+center_lista.size();
          unsigned atom2_mol2=j+center_lista.size()+start_lista.size();
          Vector mol_vector1=pbcDistance(getPosition(atom1_mol1),getPosition(atom2_mol1)); 
          Vector mol_vector2=pbcDistance(getPosition(atom1_mol2),getPosition(atom2_mol2));
          double norm_v1 = std::sqrt(mol_vector1[0]*mol_vector1[0]+mol_vector1[1]*mol_vector1[1]+mol_vector1[2]*mol_vector1[2]);
          double norm_v2 = std::sqrt(mol_vector2[0]*mol_vector2[0]+mol_vector2[1]*mol_vector2[1]+mol_vector2[2]*mol_vector2[2]);
          double inv_v1=1./norm_v1;
          double inv_v2=1./norm_v2;
          double inv_v1_sqr=inv_v1*inv_v1;
          //double inv_v2_sqr=inv_v2*inv_v2;
          double inv_v1_inv_v2=inv_v1*inv_v2;
          double cosAngle=dotProduct(mol_vector1,mol_vector2)*inv_v1*inv_v2;
          //double angle=acos(cosAngle);
          //log.printf("Angle %f radians %f degrees \n", angle, angle*180./pi);
          unsigned binAngle=std::floor((cosAngle+1.)/deltaCosAngle);
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
          //log.printf("minBinAngle %d maxBinAngle %d binAngle %d deltaBinAngle %d \n",minBinAngle,maxBinAngle,binAngle,deltaBinAngle);
          Vector der_mol1=mol_vector2*inv_v1_inv_v2-cosAngle*mol_vector1*inv_v1_sqr;
          for(int k=minBin;k<maxBin+1;k+=1) {
            //double normConstant=normConstantBase*x1sqr[k];
            invNormKernel=invTwoPiSigma1Sigma2/(normConstantBase*x1sqr[k]);
            vector<double> pos(2);
            pos[0]=x1[k]-distanceModulo;
            for(int l=minBinAngle;l<maxBinAngle+1;l+=1) {
               double theta=-1+deltaCosAngle*l;
               pos[1]=theta-cosAngle;
               // Include periodic effects
               int h;
               if (l<0) {
                  h=-l;
               } else if (l>(nhist_[1]-1)) {
                  h=2*nhist_[1]-l-2;
               } else {
                  h=l;
               }
               Vector value1;
               Vector value2_mol1;
               Vector value2_mol2;
               if (l==(nhist_[1]-1) || l==0) {
                  gofr[k][h] += kernel(pos, dfunc);
                  value1 = 2*dfunc[0]*distance_versor;
                  value2_mol1 = 2*dfunc[1]*der_mol1;
                  //value2_mol2 = 2*dfunc[1]*(mol_vector1*inv_v1_inv_v2-cosAngle*mol_vector2*inv_v2_sqr)/normConstant;
               } else {
                  gofr[k][h] += kernel(pos, dfunc)/2.;
                  value1 = dfunc[0]*distance_versor;
                  value2_mol1 = dfunc[1]*der_mol1;
                  //value2_mol2 = dfunc[1]*(mol_vector1*inv_v1_inv_v2-cosAngle*mol_vector2*inv_v2_sqr)/normConstant;
               }
               /*
               gofrPrime[k][h][i] += value1;
               gofrPrime[k][h][j] -= value1;
               gofrPrime[k][h][atom1_mol1] += value2_mol1;
               gofrPrime[k][h][atom2_mol1] -= value2_mol1;
               gofrPrime[k][h][atom1_mol2] += value2_mol2;
               gofrPrime[k][h][atom2_mol2] -= value2_mol2; 
               */
               /*
               gofrPrime[i*nhist_[0]*nhist_[1]+h*nhist_[0]+k] += value1;
               gofrPrime[atom1_mol1*nhist_[0]*nhist_[1]+h*nhist_[0]+k] += value2_mol1;
               gofrPrime[atom2_mol1*nhist_[0]*nhist_[1]+h*nhist_[0]+k] -= value2_mol1;
               */
               gofrPrimeCenter[i*nhist1_nhist2_+k*nhist_[1]+h] += value1;
               gofrPrimeStart[i*nhist1_nhist2_+k*nhist_[1]+h] +=  value2_mol1;
               gofrPrimeEnd[i*nhist1_nhist2_+k*nhist_[1]+h] -=  value2_mol1;
               //gofrPrime[j*nhist_[0]*nhist_[1]+h*nhist_[0]+k] -= value1;
               //gofrPrime[atom1_mol2*nhist_[0]*nhist_[1]+h*nhist_[0]+k] += value2_mol2;
               //gofrPrime[atom2_mol2*nhist_[0]*nhist_[1]+h*nhist_[0]+k] -= value2_mol2;
               Tensor vv1(value1, distance);
               Tensor vv2_mol1(value2_mol1, mol_vector1);
               //Tensor vv2_mol2(value2_mol2, mol_vector2);
               gofrVirial[k][h] += vv1/2.+vv2_mol1; //+vv2_mol2;
            }
          }
        }
      }
    }
  }
  //std::cout << "Main loop: " << float( clock () - begin_time ) << "\n";
  //begin_time = clock();
  if(!serial){
    comm.Sum(&gofr[0][0],nhist_[0]*nhist_[1]);
    if (!doNotCalculateDerivatives() ) {
       comm.Sum(&gofrVirial[0][0],nhist_[0]*nhist_[1]);
    }
  }
  //std::cout << "Communication: " <<  float( clock () - begin_time ) << "\n";
  //begin_time = clock();
  if (doAverageGofr) {
     for(unsigned i=0;i<nhist_[0];++i){
        for(unsigned j=0;j<nhist_[1];++j){
           avgGofr[i][j] += (gofr[i][j]-avgGofr[i][j])/( (double) iteration);
           gofr[i][j] = avgGofr[i][j];
        }
     }
     iteration += 1;
  }
  // Output of gofr
  if (doOutputGofr && (getStep()%outputStride==0) && rank==0) outputGofr(gofr,"gofr.txt");
  // Construct integrand
  Matrix<double> integrand(nhist_[0],nhist_[1]);
  Matrix<double> logGofrx1sqr(nhist_[0],nhist_[1]);
  for(unsigned i=0;i<nhist_[0];++i){
     for(unsigned j=0;j<nhist_[1];++j){
        logGofrx1sqr[i][j] = std::log(gofr[i][j])*x1sqr[i];
        if (gofr[i][j]<1.e-10) {
           integrand[i][j] = x1sqr[i];
        } else {
           integrand[i][j] = gofr[i][j]*logGofrx1sqr[i][j]+(-gofr[i][j]+1)*x1sqr[i];
        }
     }
  }
  vector<double> delta(2);
  delta[0]=deltar;
  delta[1]=deltaCosAngle;
  double TwoPiDensityVolAngles=(2*pi/volumeOfAngles)*density;
  pairEntropy=-TwoPiDensityVolAngles*integrate(integrand,delta);
  //std::cout << "Integrand and integration: " << float( clock () - begin_time ) << "\n";
  //begin_time = clock();
  // Derivatives
  if (!doNotCalculateDerivatives() ) {
    for(unsigned int k=rank;k<center_lista.size();k+=stride) {
      // Center atom
      unsigned start_atom=k+center_lista.size();
      unsigned end_atom=k+center_lista.size()+start_lista.size();
      Matrix<Vector> integrandDerivatives(nhist_[0],nhist_[1]);
      Matrix<Vector> integrandDerivativesStart(nhist_[0],nhist_[1]);
      Matrix<Vector> integrandDerivativesEnd(nhist_[0],nhist_[1]);
      for(unsigned i=0;i<nhist_[0];++i){
        for(unsigned j=0;j<nhist_[1];++j){
          if (gofr[i][j]>1.e-10) {
            integrandDerivatives[i][j] = gofrPrimeCenter[k*nhist1_nhist2_+i*nhist_[1]+j]*logGofrx1sqr[i][j];
            integrandDerivativesStart[i][j] = gofrPrimeStart[k*nhist1_nhist2_+i*nhist_[1]+j]*logGofrx1sqr[i][j];
            integrandDerivativesEnd[i][j] = gofrPrimeEnd[k*nhist1_nhist2_+i*nhist_[1]+j]*logGofrx1sqr[i][j];
          }
        }
      }
      deriv[k] = -TwoPiDensityVolAngles*integrate(integrandDerivatives,delta);
      deriv[start_atom] = -TwoPiDensityVolAngles*integrate(integrandDerivativesStart,delta);
      deriv[end_atom] = -TwoPiDensityVolAngles*integrate(integrandDerivativesEnd,delta);
    }
    if(!serial){
      comm.Sum(&deriv[0][0],3*getNumberOfAtoms());
    }
    // Virial of positions
    // Construct virial integrand
    Matrix<Tensor> integrandVirial(nhist_[0],nhist_[1]);
    for(unsigned i=0;i<nhist_[0];++i){
       for(unsigned j=0;j<nhist_[1];++j){
          if (gofr[i][j]>1.e-10) {
             integrandVirial[i][j] = gofrVirial[i][j]*logGofrx1sqr[i][j];
          }
      }
    }
    // Integrate virial
    virial = -TwoPiDensityVolAngles*integrate(integrandVirial,delta);
    // Virial of volume
    // Construct virial integrand
    Matrix<double> integrandVirialVolume(nhist_[0],nhist_[1]);
    for(unsigned i=0;i<nhist_[0];++i){
       for(unsigned j=0;j<nhist_[1];++j){
          integrandVirialVolume[i][j] = (-gofr[i][j]+1)*x1sqr[i];
       }
    }
    // Integrate virial
    virial += -TwoPiDensityVolAngles*integrate(integrandVirialVolume,delta)*Tensor::identity();
  }
  //std::cout << "Derivatives integration: " << float( clock () - begin_time ) << "\n";
  // Assign output quantities
  for(unsigned i=0;i<deriv.size();++i) setAtomsDerivatives(i,deriv[i]);
  setValue           (pairEntropy);
  setBoxDerivatives  (virial);
}

double PairOrientationalEntropy::kernel(vector<double> distance,vector<double>&der)const{
  // Gaussian function and derivative
  double result = invNormKernel*std::exp(-distance[0]*distance[0]/twoSigma1Sqr-distance[1]*distance[1]/twoSigma2Sqr) ;
  //double result = invTwoPiSigma1Sigma2*std::exp(-distance[0]*distance[0]/twoSigma1Sqr-distance[1]*distance[1]/twoSigma2Sqr) ;
  der[0] = -distance[0]*result/sigma1Sqr;
  der[1] = -distance[1]*result/sigma2Sqr;
  return result;
}

double PairOrientationalEntropy::integrate(Matrix<double> integrand, vector<double> delta)const{
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

Vector PairOrientationalEntropy::integrate(Matrix<Vector> integrand, vector<double> delta)const{
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

Tensor PairOrientationalEntropy::integrate(Matrix<Tensor> integrand, vector<double> delta)const{
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

void PairOrientationalEntropy::outputGofr(Matrix<double> gofr, const char* fileName) {
  PLMD::OFile gofrOfile;
  gofrOfile.open(fileName);
  for(unsigned i=0;i<nhist_[0];++i){
     double r=deltar*i;
     for(unsigned j=0;j<nhist_[1];++j){
        double theta=-1+deltaCosAngle*j;
        gofrOfile.printField("r",r).printField("theta",theta).printField("gofr",gofr[i][j]).printField();
     }
     gofrOfile.printf("\n");
  }
  gofrOfile.close();
}

void PairOrientationalEntropy::outputIntegrand(vector<double> integrand) {
  PLMD::OFile gofrOfile;
  gofrOfile.open("integrand.txt");
  for(unsigned i=0;i<integrand.size();++i){
     double r=deltar*(i+0.5);
     gofrOfile.printField("r",r).printField("integrand",integrand[i]).printField();
  }
  gofrOfile.close();
}

}
}
