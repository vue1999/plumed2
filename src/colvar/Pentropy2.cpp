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
#include "tools/NeighborList.h"
#include "tools/Communicator.h"
#include "tools/Tools.h" // Has pi

#include <string>

using namespace std;

namespace PLMD{
namespace colvar{

class Pentropy2 : public Colvar {
  bool pbc;
  bool serial;
  NeighborList *nl;
  bool invalidateList;
  bool firsttime;
  //SwitchingFunction switchingFunction;
  double maxr;
  unsigned nhist;
  double sigma;
  double TwoDivSqrtPi;
  double sqrt2;
  double sqrt2sigma;
  double invSqrt2sigma;

public:
  explicit Pentropy2(const ActionOptions&);
  ~Pentropy2();
// active methods:
  virtual void calculate();
  virtual void prepare();
  virtual double pairing(double distance,double&dfunc)const;
  static void registerKeywords( Keywords& keys );
};

PLUMED_REGISTER_ACTION(Pentropy2,"PENTROPY2")

void Pentropy2::registerKeywords( Keywords& keys ){
  Colvar::registerKeywords(keys);
  keys.addFlag("SERIAL",false,"Perform the calculation in serial - for debug purpose");
  keys.addFlag("PAIR",false,"Pair only 1st element of the 1st group with 1st element in the second, etc");
  keys.addFlag("NLIST",false,"Use a neighbour list to speed up the calculation");
  keys.add("optional","NL_CUTOFF","The cutoff for the neighbour list");
  keys.add("optional","NL_STRIDE","The frequency with which we are updating the atoms in the neighbour list");
  keys.add("atoms","GROUPA","First list of atoms");
  keys.add("atoms","GROUPB","Second list of atoms (if empty, N*(N-1)/2 pairs in GROUPA are counted)");
  keys.add("compulsory","MAXR","1","Maximum distance for the radial distribution function ");
  keys.add("compulsory","NHIST","1","Number of bins in the rdf ");
  keys.add("compulsory","SIGMA","0.1","Width of gaussians ");
}

Pentropy2::Pentropy2(const ActionOptions&ao):
PLUMED_COLVAR_INIT(ao),
pbc(true),
serial(false),
invalidateList(true),
firsttime(true)
{

  parseFlag("SERIAL",serial);

  vector<AtomNumber> ga_lista,gb_lista;
  parseAtomList("GROUPA",ga_lista);
  parseAtomList("GROUPB",gb_lista);

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
  if(gb_lista.size()>0){
    if(doneigh)  nl= new NeighborList(ga_lista, gb_lista, serial, dopair, pbc, getPbc(), comm, nl_cut, nl_st);
    else         nl= new NeighborList(ga_lista, gb_lista, serial, dopair, pbc, getPbc(), comm);
  } else {
    if(doneigh)  nl= new NeighborList(ga_lista, serial, pbc, getPbc(), comm, nl_cut, nl_st);
    else         nl= new NeighborList(ga_lista, serial, pbc, getPbc(), comm);
  }
  
  requestAtoms(nl->getFullAtomList());
 
  log.printf("  between two groups of %u and %u atoms\n",static_cast<unsigned>(ga_lista.size()),static_cast<unsigned>(gb_lista.size()));
  log.printf("  first group:\n");
  for(unsigned int i=0;i<ga_lista.size();++i){
   if ( (i+1) % 25 == 0 ) log.printf("  \n");
   log.printf("  %d", ga_lista[i].serial());
  }
  log.printf("  \n  second group:\n");
  for(unsigned int i=0;i<gb_lista.size();++i){
   if ( (i+1) % 25 == 0 ) log.printf("  \n");
   log.printf("  %d", gb_lista[i].serial());
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
  parse("NHIST",nhist);
  parse("SIGMA",sigma);

  checkRead();

  // Define heavily used expressions
  TwoDivSqrtPi = 2. / std::sqrt(pi) ;
  sqrt2 = std::sqrt(2);
  sqrt2sigma = sqrt2*sigma;
  invSqrt2sigma = 1./sqrt2sigma;
}

Pentropy2::~Pentropy2(){
  delete nl;
}

void Pentropy2::prepare(){
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
void Pentropy2::calculate()
{

 double entropy=0.;
 Tensor virial;
 double deltar=maxr/nhist;
 unsigned deltaBin = std::floor(3*sigma/deltar);
 vector<double> gofr(nhist);
 Matrix<Vector> gofrPrime(nhist,getNumberOfAtoms());
 vector<Vector> deriv(getNumberOfAtoms());
 double dfunc1=0.;
 double dfunc2=0.;
 Vector value;
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

const unsigned nn=nl->size();

 for(unsigned int i=rank;i<nn;i+=stride) {   
 
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
  
  double distanceModulo=distance.modulo();
  distance_versor=distance/distanceModulo;
  unsigned bin=std::floor(distanceModulo/deltar);
  // Only consider contributions to g(r) of atoms less than 5*sigma bins apart from the actual distance
  int minBin=bin - deltaBin;
  if (minBin < 0) minBin=0;
  int maxBin=bin +  deltaBin;
  if (maxBin > (nhist-1)) maxBin=nhist-1;
  double x;
  for(unsigned int j=minBin;j<(maxBin+1);j+=1) {   
        double xmin = j*deltar;
        double xmax = (j+1)*deltar;
        gofr[j] += pairing((xmax-distanceModulo)/sqrt2sigma, dfunc1) - pairing((xmin-distanceModulo)/sqrt2sigma, dfunc2);
	value =	invSqrt2sigma * (dfunc1-dfunc2) * distance_versor;
        gofrPrime[j][i0] += value;
        gofrPrime[j][i1] -= value;
  }
 }
 
 if(!serial){
   comm.Sum(gofr);
   comm.Sum(&gofrPrime[0][0],nhist*getNumberOfAtoms());
 }

 // Normalize g(r) and g'(r)
 double volume=getBox().determinant(); 
 double density=getNumberOfAtoms()/volume;
 double Nideal=(4./3.)*pi*density*pow(deltar,3);
 double normConstant;
 for(unsigned int j=0;j<nhist;j+=1) {   
   normConstant = getNumberOfAtoms()*Nideal*(pow(j+1,3)-pow(j,3)) ;
   gofr[j] /= normConstant ;
   for(unsigned int i=0;i<getNumberOfAtoms();i+=1) {
      gofrPrime[j][i] /= normConstant;
   }
 }

 // Construct integrand
 vector<double> integrand(nhist);
 for(unsigned int j=0;j<nhist;j+=1) {   
   if (gofr[j]<1.e-10) {
     integrand[j] = pow(j*deltar,2);
   } else {
     integrand[j] = (gofr[j]*std::log(gofr[j])-gofr[j]+1)*pow(j*deltar,2);
   }
 }

 // Integrate
 entropy += (integrand[0] + integrand[nhist-1])/2.;
 for(unsigned int j=1;j<nhist-1;j+=1) {   
   entropy += integrand[j];
 }
 entropy *= deltar; // Integration delta
 entropy *= -2*pi*density; // Integral pre-factor

  // Derivatives
  for(unsigned int i=rank;i<getNumberOfAtoms();i+=stride) {
    // First and last point of the integration should be treated differently from the rest
    for(unsigned int j=0;j<nhist;j+=1) {   
      if (gofr[j]>1.e-10) {
        deriv[i] += (gofrPrime[j][i]*std::log(gofr[j]))*pow(deltar*j,2);
      }
    }
    deriv[i] *= deltar; // Integration delta
    deriv[i] *= -2*pi*density; // Integral pre-factor
    //Tensor vv(deriv[i],distances[i]);
    //virial-=vv;
  }


 if(!serial){
   if(!deriv.empty()) comm.Sum(&deriv[0][0],3*deriv.size());
   comm.Sum(virial);
 }

 for(unsigned i=0;i<deriv.size();++i) setAtomsDerivatives(i,deriv[i]);
 setValue           (entropy);
 setBoxDerivatives  (virial);

}

double Pentropy2::pairing(double distance,double&dfunc)const{
  // Error function and derivative
  double result = erf(distance);
  dfunc = TwoDivSqrtPi*std::exp(-distance*distance);
  return result;
}




}
}
