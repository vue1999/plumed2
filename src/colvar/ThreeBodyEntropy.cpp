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

class ThreeBodyEntropy : public Colvar {
  bool pbc;
  bool serial;
  NeighborList *nl;
  bool invalidateList;
  bool firsttime;
  bool doOutputGofr;
  bool doOutputIntegrand;
  double maxr, sigma;
  unsigned nhist;
  double rcut2;
  double invSqrt2piSigma, sigmaSqr2, sigmaSqr;
  double deltaAngle;
  unsigned deltaBin;
  // Integration routines
  double integrate(vector<double> integrand, double delta)const;
  Vector integrate(vector<Vector> integrand, double delta)const;
  Tensor integrate(vector<Tensor> integrand, double delta)const;
  // Kernel to calculate g(r)
  double kernel(double distance, double&der)const;
  // Output gofr and integrand
  void outputGofr(vector<double> gofr);
  void outputIntegrand(vector<double> integrand);
public:
  explicit ThreeBodyEntropy(const ActionOptions&);
  ~ThreeBodyEntropy();
// active methods:
  virtual void calculate();
  virtual void prepare();
  static void registerKeywords( Keywords& keys );
};

PLUMED_REGISTER_ACTION(ThreeBodyEntropy,"THREE_BODY_ENTROPY")

void ThreeBodyEntropy::registerKeywords( Keywords& keys ){
  Colvar::registerKeywords(keys);
  keys.addFlag("SERIAL",false,"Perform the calculation in serial - for debug purpose");
  keys.addFlag("PAIR",false,"Pair only 1st element of the 1st group with 1st element in the second, etc");
  keys.addFlag("NLIST",false,"Use a neighbour list to speed up the calculation");
  keys.addFlag("OUTPUT_GOFR",false,"Output g(r)");
  //keys.addFlag("OUTPUT_INTEGRAND",false,"Output integrand");
  keys.add("optional","NL_CUTOFF","The cutoff for the neighbour list");
  keys.add("optional","NL_STRIDE","The frequency with which we are updating the atoms in the neighbour list");
  keys.add("atoms","GROUP","List of atoms");
  //keys.add("atoms","GROUPB","Second list of atoms (if empty, N*(N-1)/2 pairs in GROUPA are counted)");
  keys.add("compulsory","MAXR","1","Maximum distance for the radial distribution function ");
  keys.add("compulsory","NHIST","1","Number of bins in the rdf ");
  keys.add("compulsory","SIGMA","0.1","Width of gaussians ");
}

ThreeBodyEntropy::ThreeBodyEntropy(const ActionOptions&ao):
PLUMED_COLVAR_INIT(ao),
pbc(true),
serial(false),
invalidateList(true),
firsttime(true)
{

  parseFlag("SERIAL",serial);
  parseFlag("OUTPUT_GOFR",doOutputGofr);
  //parseFlag("OUTPUT_INTEGRAND",doOutputIntegrand);

  vector<AtomNumber> ga_lista,gb_lista;
  parseAtomList("GROUP",ga_lista);
  //parseAtomList("GROUPB",gb_lista);

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
  if(ga_lista.size()>0){
    if(doneigh)  nl= new NeighborList(ga_lista,pbc,getPbc(),nl_cut,nl_st);
    else         nl= new NeighborList(ga_lista,pbc,getPbc());
  }

  requestAtoms(nl->getFullAtomList());

  log.printf("  between a group of %u atoms\n",static_cast<unsigned>(ga_lista.size()));
  log.printf("  group:\n");
  for(unsigned int i=0;i<ga_lista.size();++i){
   if ( (i+1) % 25 == 0 ) log.printf("  \n");
   log.printf("  %d", ga_lista[i].serial());
  }
  log.printf("  \n");
  if(pbc) log.printf("  using periodic boundary conditions\n");
  else    log.printf("  without periodic boundary conditions\n");
  if(dopair) log.printf("  with PAIR option\n");
  if(doneigh){
   log.printf("  using neighbor lists with\n");
   log.printf("  update every %d steps and cutoff %f\n",nl_st,nl_cut);
  }

  parse("MAXR",maxr);
  log.printf("Integration in the interval from 0. to %f nm. \n", maxr );
  parse("NHIST",nhist);
  log.printf("The interval is partitioned in %u equal parts and the integration is perfromed with the trapezoid rule. \n", nhist );
  parse("SIGMA",sigma);
  log.printf("The pair distribution function is calculated with a Gaussian kernel with deviation %f nm. \n", sigma);
  double rcut = maxr; // + 3*sigma;
  rcut2 = maxr*maxr; //(maxr + 3*sigma)*(maxr + 3*sigma);  // 3*sigma is hard coded
  if(doneigh){
    if(nl_cut<rcut) error("NL_CUTOFF should be larger than MAXR + 3*SIGMA");
  }

  checkRead();

  // Define heavily used expressions
  double sqrt2piSigma = std::sqrt(2*pi)*sigma;
  invSqrt2piSigma = 1./sqrt2piSigma;
  sigmaSqr2 = 2.*sigma*sigma;
  sigmaSqr = sigma*sigma;
  deltaAngle=2./nhist;
  deltaBin = std::floor(3*sigma/deltaAngle); // 3*sigma is hard coded
}

ThreeBodyEntropy::~ThreeBodyEntropy(){
  delete nl;
}

void ThreeBodyEntropy::prepare(){
  if(nl->getStride()>0){
    if(firsttime || (getStep()%nl->getStride()==0)){
      requestAtoms(nl->getFullAtomList());
      invalidateList=true;
      firsttime=false;
    }else{
      requestAtoms(nl->getReducedAtomList());
      invalidateList=false;
      if(getExchangeStep()) error("Neighbor lists should be updated on exchange steps - choose a NL_STRIDE which divides the exchange stride!");
    }
    if(getExchangeStep()) firsttime=true;
  }
}

// calculator
void ThreeBodyEntropy::calculate()
{
  // Define output quantities
  double threeBodyEntropy;
  vector<Vector> deriv(getNumberOfAtoms());
  vector<double> angleHistogram(nhist);
  vector<double> angleHistogram2(nhist);
  Tensor virial;
  int numberOfTriplets = 0;
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
  // Loop over central atoms
  for(unsigned int i=rank;i<getNumberOfAtoms();i+=stride) {
    double dfunc, d2_ij, d2_ik;
    //Vector distance_versor;
    vector<unsigned> neighbors=nl->getNeighbors(i);
    unsigned neighNumber=neighbors.size();
    Vector ri=getPosition(i);
    Vector distance_ij, distance_ik;
    // First loop over neighbors
    for(unsigned int j=0;j<(neighNumber-1);j+=1) {
       unsigned j0=neighbors[j];
       //if(getAbsoluteIndex(i)==getAbsoluteIndex(j0)) continue;
       if(pbc){ distance_ij=pbcDistance(ri,getPosition(j0));
       } else { distance_ij=delta(ri,getPosition(j0)); }
       if ( (d2_ij=distance_ij[0]*distance_ij[0])<rcut2 && (d2_ij+=distance_ij[1]*distance_ij[1])<rcut2 && (d2_ij+=distance_ij[2]*distance_ij[2])<rcut2) {
          double inv_d_ij=1./std::sqrt(d2_ij);
          // Second loop over neighbors
          for(unsigned int k=j+1;k<neighNumber;k+=1) {
             unsigned k0=neighbors[k];
             //if(getAbsoluteIndex(j0)==getAbsoluteIndex(k0)) continue;
             if(pbc){ distance_ik=pbcDistance(ri,getPosition(k0));
             } else { distance_ik=delta(ri,getPosition(k0)); }
             if ( (d2_ik=distance_ik[0]*distance_ik[0])<rcut2 && (d2_ik+=distance_ik[1]*distance_ik[1])<rcut2 && (d2_ik+=distance_ik[2]*distance_ik[2])<rcut2) {
                double inv_d_ik=1./std::sqrt(d2_ik);
                double angle=dotProduct(distance_ij,distance_ik)*inv_d_ij*inv_d_ik;
                //log.printf("Angle of %d, %d, %d is %f \n", i, j0, k0, angle);
                unsigned bin=std::floor((angle+1.)/deltaAngle);
                int minBin, maxBin; // These cannot be unsigned
                // Only consider contributions to g(r) of atoms less than n*sigma bins apart from the actual distance
                minBin=bin - deltaBin;
                //if (minBin < 0) minBin=0;
                //if (minBin > (nhist-1)) minBin=nhist-1;
                maxBin=bin +  deltaBin;
                //if (maxBin > (nhist-1)) maxBin=nhist-1;
                for(int l=minBin;l<maxBin+1;l+=1) {
                   int h;
                   if (l<0) {
                      h=-l-1;
                   } else if (l>(nhist-1)) {
                      h=2*nhist-l-1;
                   } else {
                      h=l;
                   }
                   double x=deltaAngle*(l+0.5)-1;
                   angleHistogram[h] += kernel(x-angle, dfunc);
                }
                ++numberOfTriplets;
                /*
                // Reflect values at boundaries
                if (bin<deltaBin) {
                   //log.printf("bin is %d and angle is %f \n",bin, angle);
                   for(int l=0;l<deltaBin;l+=1) {
                      //log.printf("l is %d and deltaBin is %d and angle is %f \n",l);
                      double x=deltaAngle*(l+0.5)-1;
                      double angleReflect = -angle-2.;
                      angleHistogram[l] += kernel(x-angleReflect, dfunc);
                   }
                } else if ((nhist-bin-1)<deltaBin) {
                   //log.printf("bin is %d and angle is %f \n",bin, angle);
                   for(int l=(nhist-deltaBin);l<nhist;l+=1) {
                      //log.printf("l is %d and deltaBin is %d and angle is %f \n",l,deltaBin,angle);
                      double x=deltaAngle*(l+0.5)-1;
                      double angleReflect = -angle+2.;
                      angleHistogram[l] += kernel(x-angleReflect, dfunc);
                   }
                }
                */
             }
          }
       }
    }
  }
  if(!serial){
    comm.Sum(&angleHistogram[0],nhist);
    comm.Sum(numberOfTriplets);
  }
  double norm=1./numberOfTriplets;
  // This is the right one
  //double volume=getBox().determinant();
  //double density=getNumberOfAtoms()/volume;
  //double norm=(getNumberOfAtoms()-1)*(getNumberOfAtoms()-2)/(8.*pi*pi*density*density*numberOfTriplets);
  //log.printf( "Norm %f, density %f, and triplets %d \n", norm,density,numberOfTriplets);
  vector<double> integrand(nhist);
  for(int l=0;l<nhist;l+=1) {
     double x=deltaAngle*(l+0.5)-1;
     angleHistogram[l] *= norm;
     //log.printf("angle histo %f %f \n",x,angleHistogram[l]);
     if (angleHistogram[l]<1.e-10) {
        integrand[l]=-angleHistogram[l];
     } else {
        integrand[l]=angleHistogram[l]*std::log(angleHistogram[l]/0.5);
     }
  }
  if (doOutputGofr && rank==0) outputGofr(angleHistogram);
  //double coordinationNumber2=8.*pi*pi*density*density*integrate(angleHistogram,deltaAngle);
  //log.printf( "CoordNumber2 %f \n", coordinationNumber2);
  //threeBodyEntropy=-(4./3.)*density*density*pi*pi*integrate(integrand,deltaAngle);
  threeBodyEntropy=-integrate(integrand,deltaAngle);
    /*
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
      int minBin, maxBin; // These cannot be unsigned
      // Only consider contributions to g(r) of atoms less than n*sigma bins apart from the actual distance
      minBin=bin - deltaBin;
      if (minBin < 0) minBin=0;
      if (minBin > (nhist-1)) minBin=nhist-1;
      maxBin=bin +  deltaBin;
      if (maxBin > (nhist-1)) maxBin=nhist-1;
      for(int k=minBin;k<maxBin+1;k+=1) {
        double x=deltar*(k+0.5);
        gofr[k] += kernel(x-distanceModulo, dfunc);
        Vector value = dfunc * distance_versor;
        gofrPrime[k][i0] += value;
        gofrPrime[k][i1] -= value;
        Tensor vv(value, distance);
        gofrVirial[k] += vv;
      }
    }
  }
  if(!serial){
    comm.Sum(&gofr[0],nhist);
  }
  */
  // Assign output quantities
  //for(unsigned i=0;i<deriv.size();++i) setAtomsDerivatives(i,deriv[i]);
  setValue           (threeBodyEntropy);
  //setBoxDerivatives  (virial);
}

double ThreeBodyEntropy::kernel(double distance,double&der)const{
  // Gaussian function and derivative
  double result = invSqrt2piSigma*std::exp(-distance*distance/sigmaSqr2) ;
  der = -distance*result/sigmaSqr;
  return result;
}

double ThreeBodyEntropy::integrate(vector<double> integrand, double delta)const{
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

Vector ThreeBodyEntropy::integrate(vector<Vector> integrand, double delta)const{
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

Tensor ThreeBodyEntropy::integrate(vector<Tensor> integrand, double delta)const{
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

void ThreeBodyEntropy::outputGofr(vector<double> gofr) {
  PLMD::OFile gofrOfile;
  gofrOfile.open("gofr.txt");
  for(unsigned i=0;i<gofr.size();++i){
     double r=deltaAngle*(i+0.5)-1;
     gofrOfile.printField("r",r).printField("gofr",gofr[i]).printField();
  }
  gofrOfile.close();
}

void ThreeBodyEntropy::outputIntegrand(vector<double> integrand) {
  PLMD::OFile gofrOfile;
  gofrOfile.open("integrand.txt");
  for(unsigned i=0;i<integrand.size();++i){
     double r=deltaAngle*(i+0.5);
     gofrOfile.printField("r",r).printField("integrand",integrand[i]).printField();
  }
  gofrOfile.close();
}


}
}
