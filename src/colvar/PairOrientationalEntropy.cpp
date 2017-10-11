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
#include "tools/NeighborList.h"
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
  bool pbc;
  bool serial;
  NeighborList *nl;
  bool invalidateList;
  bool firsttime;
  bool doOutputGofr;
  bool doOutputIntegrand;
  double maxr, sigma;
  unsigned nhist;
  vector<int> nhist_;
  vector<double> sigma_;
  double rcut2;
  double invSqrt2piSigma, sigmaSqr2, sigmaSqr;
  double deltar, deltaAngle;
  unsigned deltaBin, deltaBinAngle;
  // Integration routines
  double integrate(vector<double> integrand, double delta)const;
  Vector integrate(vector<Vector> integrand, double delta)const;
  Tensor integrate(vector<Tensor> integrand, double delta)const;
  // Kernel to calculate g(r)
  double kernel(vector<double> distance, double&der)const;
  // Output gofr and integrand
  void outputGofr(Matrix<double> gofr);
  void outputIntegrand(vector<double> integrand);
  vector<AtomNumber> center_lista,start_lista,end_lista;
  std::vector<PLMD::AtomNumber> atomsToRequest;
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
  keys.addFlag("PAIR",false,"Pair only 1st element of the 1st group with 1st element in the second, etc");
  keys.addFlag("NLIST",false,"Use a neighbour list to speed up the calculation");
  keys.addFlag("OUTPUT_GOFR",false,"Output g(r)");
  keys.addFlag("OUTPUT_INTEGRAND",false,"Output integrand");
  keys.add("optional","NL_CUTOFF","The cutoff for the neighbour list");
  keys.add("optional","NL_STRIDE","The frequency with which we are updating the atoms in the neighbour list");
  keys.add("atoms","CENTER","Center atoms");
  keys.add("atoms","START","Start point of vector defining orientation");
  keys.add("atoms","END","End point of vector defining orientation");
  keys.add("compulsory","MAXR","1","Maximum distance for the radial distribution function ");
  keys.add("compulsory","NHIST","1","Number of bins in the rdf ");
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
  parseFlag("OUTPUT_GOFR",doOutputGofr);
  parseFlag("OUTPUT_INTEGRAND",doOutputIntegrand);

  parseAtomList("CENTER",center_lista);
  parseAtomList("START",start_lista);
  parseAtomList("END",end_lista);

  bool nopbc=!pbc;
  parseFlag("NOPBC",nopbc);
  pbc=!nopbc;

// pair stuff
  bool dopair=false;
  parseFlag("PAIR",dopair);

// neighbor list stuff
  bool doneigh=false;
  double nl_cut=0.0;
  int nl_st=0;
  parseFlag("NLIST",doneigh);
  if(doneigh){
   parse("NL_CUTOFF",nl_cut);
   if(nl_cut<=0.0) error("NL_CUTOFF should be explicitly specified and positive");
   parse("NL_STRIDE",nl_st);
   if(nl_st<=0) error("NL_STRIDE should be explicitly specified and positive");
  }

  addValueWithDerivatives(); setNotPeriodic();
  if(doneigh)  nl= new NeighborList(center_lista,pbc,getPbc(),nl_cut,nl_st);
  else         nl= new NeighborList(center_lista,pbc,getPbc());

  atomsToRequest.reserve ( center_lista.size() + start_lista.size() + end_lista.size() );
  atomsToRequest.insert (atomsToRequest.end(), center_lista.begin(), center_lista.end() );
  atomsToRequest.insert (atomsToRequest.end(), start_lista.begin(), start_lista.end() );
  atomsToRequest.insert (atomsToRequest.end(), end_lista.begin(), end_lista.end() );
  requestAtoms(atomsToRequest);

  if(pbc) log.printf("  using periodic boundary conditions\n");
  else    log.printf("  without periodic boundary conditions\n");
  if(dopair) log.printf("  with PAIR option\n");
  if(doneigh){
   log.printf("  using neighbor lists with\n");
   log.printf("  update every %d steps and cutoff %f\n",nl_st,nl_cut);
  }

  parse("MAXR",maxr);
  log.printf("Integration in the interval from 0. to %f nm. \n", maxr );
  parseVector("NHIST",nhist_);
  if(nhist_.size() != 2) error("NHIST keyword takes two input values");
  nhist =nhist_[0];
  log.printf("The interval is partitioned in %u equal parts in r and %u equal parts in tehta. The integration is performed with the trapezoid rule. \n", nhist_[0], nhist_[1] );
  parseVector("SIGMA",sigma_);
  if(sigma_.size() != 2) error("SIGMA keyword takes two input values");
  sigma=sigma_[0];
  log.printf("The pair distribution function is calculated with a Gaussian kernel with deviations %f and %f \n", sigma_[0], sigma_[1]);
  double rcut = maxr + 3*sigma_[0];
  rcut2 = (maxr + 3*sigma)*(maxr + 3*sigma_[0]);  // 3*sigma is hard coded
  if(doneigh){
    if(nl_cut<rcut) error("NL_CUTOFF should be larger than MAXR + 3*SIGMA");
  }

  checkRead();

  // Define heavily used expressions
  double sqrt2piSigma = std::sqrt(2*pi)*sigma;
  invSqrt2piSigma = 1./sqrt2piSigma;
  sigmaSqr2 = 2.*sigma*sigma;
  sigmaSqr = sigma*sigma;
  deltar=maxr/nhist_[0];
  deltaAngle=pi/nhist_[1];
  deltaBin = std::floor(3*sigma_[0]/deltar); // 3*sigma is hard coded
  deltaBinAngle = std::floor(3*sigma_[1]/deltaAngle); // 3*sigma is hard coded
}

PairOrientationalEntropy::~PairOrientationalEntropy(){
  delete nl;
}

void PairOrientationalEntropy::prepare(){
  if(nl->getStride()>0){
    requestAtoms(atomsToRequest);
    if(firsttime || (getStep()%nl->getStride()==0)){
      invalidateList=true;
      firsttime=false;
    }else{
      invalidateList=false;
      if(getExchangeStep()) error("Neighbor lists should be updated on exchange steps - choose a NL_STRIDE which divides the exchange stride!");
    }
    if(getExchangeStep()) firsttime=true;
  }
}

// calculator
void PairOrientationalEntropy::calculate()
{
  // Define output quantities
  double pairEntropy;
  vector<Vector> deriv(getNumberOfAtoms());
  Tensor virial;
  // Define intermediate quantities
  Matrix<double> gofr(nhist_[0],nhist_[1]);
  //vector<double> logGofr(nhist);
  //Matrix<Vector> gofrPrime(nhist,getNumberOfAtoms());
  //vector<Tensor> gofrVirial(nhist);
  // Setup neighbor list and parallelization
  if(nl->getStride()>0 && invalidateList){
    nl->update(getPositions());
  }
  unsigned stride=comm.Get_size();
  unsigned rank=comm.Get_rank();
  if(serial){
    stride=1;
    rank=0;
  }else{
    stride=comm.Get_size();
    rank=comm.Get_rank();
  }
  // Loop over neighbors
  const unsigned nn=nl->size();
  for(unsigned int i=rank;i<nn;i+=stride) {
    double dfunc, d2;
    Vector distance;
    Vector distance_versor;
    unsigned i0=nl->getClosePair(i).first;
    unsigned i1=nl->getClosePair(i).second;
    if(getAbsoluteIndex(i0)==getAbsoluteIndex(i1)) continue;
    if(pbc){
     distance=pbcDistance(getPosition(i0),getPosition(i1));
    } else {
     distance=delta(getPosition(i0),getPosition(i1));
    }
    if ( (d2=distance[0]*distance[0])<rcut2 && (d2+=distance[1]*distance[1])<rcut2 && (d2+=distance[2]*distance[2])<rcut2) {
      double distanceModulo=std::sqrt(d2);
      Vector distance_versor = distance / distanceModulo;
      unsigned bin=std::floor(distanceModulo/deltar);
      unsigned atom1_mol1=i0+center_lista.size();
      unsigned atom2_mol1=i0+center_lista.size()+start_lista.size();
      unsigned atom1_mol2=i1+center_lista.size();
      unsigned atom2_mol2=i1+center_lista.size()+start_lista.size();;
      Vector mol_vector1=pbcDistance(getPosition(atom1_mol1),getPosition(atom2_mol1));
      Vector mol_vector2=pbcDistance(getPosition(atom1_mol2),getPosition(atom2_mol2));
      double norm_v1 = std::sqrt(mol_vector1[0]*mol_vector1[0]+mol_vector1[1]*mol_vector1[1]+mol_vector1[2]*mol_vector1[2]);
      double norm_v2 = std::sqrt(mol_vector2[0]*mol_vector2[0]+mol_vector2[1]*mol_vector2[1]+mol_vector2[2]*mol_vector2[2]);
      double inv_v1=1./norm_v1;
      double inv_v2=1./norm_v2;
      double cosAngle=dotProduct(mol_vector1,mol_vector2)*inv_v1*inv_v2;
      double angle=acos(cosAngle);
      unsigned binAngle=std::floor(angle/deltaAngle);
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
      if (minBinAngle < 0) minBinAngle=0;
      if (minBinAngle > (nhist_[1]-1)) minBinAngle=nhist_[1]-1;
      maxBinAngle=binAngle +  deltaBinAngle;
      if (maxBinAngle > (nhist_[1]-1)) maxBinAngle=nhist_[1]-1;
      for(int k=minBin;k<maxBin+1;k+=1) {
        double x=deltar*(k+0.5);
        for(int l=minBinAngle;l<maxBinAngle+1;l+=1) {
           double theta=deltaAngle*(l+0.5);
           vector<double> pos(2);
           pos[0]=x-distanceModulo;
           pos[1]=theta-angle;
           gofr[k][l] += kernel(pos, dfunc);
        }
        //gofr[k] += kernel(x-distanceModulo, dfunc);
        //Vector value = dfunc * distance_versor;
        //gofrPrime[k][i0] += value;
        //gofrPrime[k][i1] -= value;
        //Tensor vv(value, distance);
        //gofrVirial[k] += vv;
      }
    }
  }
  if(!serial){
    comm.Sum(&gofr[0][0],nhist_[0]*nhist_[1]);
    //comm.Sum(&gofrPrime[0][0],nhist*getNumberOfAtoms());
    //comm.Sum(&gofrVirial[0],nhist);
  }
  // Calculate volume and density
  double volume=getBox().determinant();
  double density=getNumberOfAtoms()/volume;
  // Normalize g(r)
  /*
  double normConstantBase = 2*pi*getNumberOfAtoms()*density;
  for(unsigned j=0;j<nhist;++j){
    double x=deltar*(j+0.5);
    double normConstant = normConstantBase*x*x;
    gofr[j][0] /= normConstant;
    //log.printf(" gofr after %f %f \n",x, gofr[j]);
    //gofrVirial[j] /= normConstant;
    //for(unsigned k=0;k<getNumberOfAtoms();++k){
    //  gofrPrime[j][k] /= normConstant;
    //}
  }
  */
  // Output of gofr
  if (doOutputGofr && rank==0) outputGofr(gofr);
  // Construct integrand
  vector<double> integrand(nhist);
  /*
  for(unsigned j=0;j<nhist;++j){
    double x=deltar*(j+0.5);
    logGofr[j] = std::log(gofr[j]);
    if (gofr[j]<1.e-10) {
      integrand[j] = x*x;
    } else {
      integrand[j] = (gofr[j]*logGofr[j]-gofr[j]+1)*x*x;
    }
  }
  */
  // Output of integrands
  //if (doOutputIntegrand && rank==0) outputIntegrand(integrand);
  // Integrate to obtain pair entropy;
  pairEntropy = -2*pi*density*integrate(integrand,deltar);
  /*
  // Construct integrand and integrate derivatives
  if (!doNotCalculateDerivatives() ) {
    for(unsigned int j=rank;j<getNumberOfAtoms();j+=stride) {
      vector<Vector> integrandDerivatives(nhist);
      for(unsigned k=0;k<nhist;++k){
        double x=deltar*(k+0.5);
        if (gofr[k]>1.e-10) {
          integrandDerivatives[k] = gofrPrime[k][j]*logGofr[k]*x*x;
        }
      }
      // Integrate
      deriv[j] = -2*pi*density*integrate(integrandDerivatives,deltar);
    }
    comm.Sum(&deriv[0][0],3*getNumberOfAtoms());
    // Virial of positions
    // Construct virial integrand
    vector<Tensor> integrandVirial(nhist);
    for(unsigned j=0;j<nhist;++j){
      double x=deltar*(j+0.5);
      if (gofr[j]>1.e-10) {
        integrandVirial[j] = gofrVirial[j]*logGofr[j]*x*x;
      }
    }
    // Integrate virial
    virial = -2*pi*density*integrate(integrandVirial,deltar);
    // Virial of volume
    // Construct virial integrand
    vector<double> integrandVirialVolume(nhist);
    for(unsigned j=0;j<nhist;j+=1) {
      double x=deltar*(j+0.5);
      integrandVirialVolume[j] = (-gofr[j]+1)*x*x;
    }
    // Integrate virial
    virial += -2*pi*density*integrate(integrandVirialVolume,deltar)*Tensor::identity();
  }
  */
  // Assign output quantities
  //for(unsigned i=0;i<deriv.size();++i) setAtomsDerivatives(i,deriv[i]);
  setValue           (pairEntropy);
  //setBoxDerivatives  (virial);
}

double PairOrientationalEntropy::kernel(vector<double> distance,double&der)const{
  // Gaussian function and derivative
  //double result = invSqrt2piSigma*std::exp(-distance*distance/sigmaSqr2) ;
  //der = -distance*result/sigmaSqr;
  double result = (1./(2.*sigma_[0]*sigma_[1]))*std::exp(-distance[0]*distance[0]/(2*sigma_[0]*sigma_[0])-distance[1]*distance[1]/(2*sigma_[1]*sigma_[1])) ;
  der = 0.;
  return result;
}

double PairOrientationalEntropy::integrate(vector<double> integrand, double delta)const{
  // Trapezoid rule
  double result = 0.;
  for(unsigned i=1;i<(integrand.size()-1);++i){
    result += integrand[i];
  }
  result += 0.5*integrand[0];
  result += 0.5*integrand[integrand.size()-1];
  result *= delta;
  return result;
}

Vector PairOrientationalEntropy::integrate(vector<Vector> integrand, double delta)const{
  // Trapezoid rule
  Vector result;
  for(unsigned i=1;i<(integrand.size()-1);++i){
      result += integrand[i];
  }
  result += 0.5*integrand[0];
  result += 0.5*integrand[integrand.size()-1];
  result *= delta;
  return result;
}

Tensor PairOrientationalEntropy::integrate(vector<Tensor> integrand, double delta)const{
  // Trapezoid rule
  Tensor result;
  for(unsigned i=1;i<(integrand.size()-1);++i){
      result += integrand[i];
  }
  result += 0.5*integrand[0];
  result += 0.5*integrand[integrand.size()-1];
  result *= delta;
  return result;
}

void PairOrientationalEntropy::outputGofr(Matrix<double> gofr) {
  PLMD::OFile gofrOfile;
  gofrOfile.open("gofr.txt");
  for(unsigned i=0;i<nhist_[0];++i){
     double r=deltar*(i+0.5);
     for(unsigned j=0;j<nhist_[1];++j){
        double theta=deltaAngle*(j+0.5);
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
