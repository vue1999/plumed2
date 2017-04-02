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
#include "tools/Tools.h" // Has pi
#include "tools/SwitchingFunction.h"

#include <string>

using namespace std;

namespace PLMD{
namespace colvar{

class Perovskite : public Colvar {
  bool pbc;
  bool serial;
  NeighborList *nl;
  bool invalidateList;
  bool firsttime;
  std::vector<PLMD::AtomNumber> atomsToRequest;
  SwitchingFunction switchingFunction1;
  SwitchingFunction switchingFunction2;
  SwitchingFunction switchingFunction3;
  SwitchingFunction coordSwitchingFunction1;
  SwitchingFunction coordSwitchingFunction2;
  SwitchingFunction coordSwitchingFunction3;
  vector<AtomNumber> center_lista, around_species1_lista, around_species2_lista, around_species3_lista, around_species_lista ;

public:
  explicit Perovskite(const ActionOptions&);
  ~Perovskite();
// active methods:
  virtual void calculate();
  virtual void prepare();
  static void registerKeywords( Keywords& keys );
};

PLUMED_REGISTER_ACTION(Perovskite,"PEROVSKITE")

void Perovskite::registerKeywords( Keywords& keys ){
  Colvar::registerKeywords(keys);
  keys.addFlag("SERIAL",false,"Perform the calculation in serial - for debug purpose");
  keys.addFlag("PAIR",false,"Pair only 1st element of the 1st group with 1st element in the second, etc");
  keys.addFlag("NLIST",false,"Use a neighbour list to speed up the calculation");
  keys.add("optional","NL_CUTOFF","The cutoff for the neighbour list 1");
  keys.add("optional","NL_STRIDE","The frequency with which we are updating the atoms in the neighbour list");
  keys.add("atoms","CENTER","Center atoms");
  keys.add("atoms","AROUND_SPECIES1","Around atoms 1");
  keys.add("atoms","AROUND_SPECIES2","Around atoms 1");
  keys.add("atoms","AROUND_SPECIES3","Around atoms 1");
  keys.add("optional","SWITCH1","Switching function 1.");
  keys.add("optional","SWITCH2","Switching function 1.");
  keys.add("optional","SWITCH3","Switching function 1.");
  keys.add("optional","COORD_SWITCH1","Coordination switching function 1.");
  keys.add("optional","COORD_SWITCH2","Coordination switching function 1.");
  keys.add("optional","COORD_SWITCH3","Coordination switching function 1.");
}

Perovskite::Perovskite(const ActionOptions&ao):
PLUMED_COLVAR_INIT(ao),
pbc(true),
serial(false),
invalidateList(true),
firsttime(true)
{

  parseFlag("SERIAL",serial);

  parseAtomList("CENTER",center_lista);
  parseAtomList("AROUND_SPECIES1",around_species1_lista);
  parseAtomList("AROUND_SPECIES2",around_species2_lista);
  parseAtomList("AROUND_SPECIES3",around_species3_lista);
  around_species_lista.reserve ( around_species1_lista.size() + around_species2_lista.size() +around_species3_lista.size() );
  around_species_lista.insert (  around_species_lista.end() , around_species1_lista.begin(),  around_species1_lista.end() );
  around_species_lista.insert (  around_species_lista.end() , around_species2_lista.begin(),  around_species2_lista.end() );
  around_species_lista.insert (  around_species_lista.end() , around_species3_lista.begin(),  around_species3_lista.end() );

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
  if(center_lista.size()>0 && around_species1_lista.size()>0  && around_species2_lista.size()>0 && around_species3_lista.size()>0){
    if(doneigh)  nl= new NeighborList(center_lista,around_species_lista,dopair,pbc,getPbc(),nl_cut,nl_st);
    else         nl= new NeighborList(center_lista,around_species_lista,dopair,pbc,getPbc());
  } else {
    error("CENTER, AROUND_SPECIES1, AROUND_SPECIES2, and AROUND_SPECIES3 should be explicitly defined.");
  }
  atomsToRequest.reserve ( center_lista.size() + around_species_lista.size() );
  atomsToRequest.insert (atomsToRequest.end(), center_lista.begin(), center_lista.end() );
  atomsToRequest.insert (atomsToRequest.end(), around_species_lista.begin(), around_species_lista.end() );
  requestAtoms(atomsToRequest);
  log.printf("  between four groups of %u, %u, %u and %u atoms\n",static_cast<unsigned>(center_lista.size()),static_cast<unsigned>(around_species1_lista.size()),static_cast<unsigned>(around_species2_lista.size()), static_cast<unsigned>(around_species3_lista.size()) );
  log.printf("  first group:\n");
  for(unsigned int i=0;i<center_lista.size();++i){
   if ( (i+1) % 25 == 0 ) log.printf("  \n");
   log.printf("  %d", center_lista[i].serial());
  }
  log.printf("  \n  second group:\n");
  for(unsigned int i=0;i<around_species1_lista.size();++i){
   if ( (i+1) % 25 == 0 ) log.printf("  \n");
   log.printf("  %d", around_species1_lista[i].serial());
  }
  log.printf("  \n  third group:\n");
  for(unsigned int i=0;i<around_species2_lista.size();++i){
   if ( (i+1) % 25 == 0 ) log.printf("  \n");
   log.printf("  %d", around_species2_lista[i].serial());
  }
  log.printf("  \n  fourth group:\n");
  for(unsigned int i=0;i<around_species3_lista.size();++i){
   if ( (i+1) % 25 == 0 ) log.printf("  \n");
   log.printf("  %d", around_species3_lista[i].serial());
  }
  log.printf("  \n");
  if(pbc) log.printf("  using periodic boundary conditions\n");
  else    log.printf("  without periodic boundary conditions\n");
  if(dopair) log.printf("  with PAIR option\n");
  if(doneigh){
   log.printf("  using neighbor lists with\n");
   log.printf("  update every %d steps and cutoff %f \n",nl_st,nl_cut);
  }

  string sw,errors;
  // Distance switching functions
  parse("SWITCH1",sw);
  if(sw.length()>0){
    switchingFunction1.set(sw,errors);
    if( errors.length()!=0 ) error("problem reading SWITCH1 keyword : " + errors );
  } else {
    error("No switching function 1 defined. Please define SWITCH1!");
  }
  parse("SWITCH2",sw);
  if(sw.length()>0){
    switchingFunction2.set(sw,errors);
    if( errors.length()!=0 ) error("problem reading SWITCH2 keyword : " + errors );
  } else {
    error("No switching function 2 defined. Please define SWITCH2!");
  }
  parse("SWITCH3",sw);
  if(sw.length()>0){
    switchingFunction3.set(sw,errors);
    if( errors.length()!=0 ) error("problem reading SWITCH3 keyword : " + errors );
  } else {
    error("No switching function 3 defined. Please define SWITCH3!");
  }
  // Coordination switching functions
  parse("COORD_SWITCH1",sw);
  if(sw.length()>0){
    coordSwitchingFunction1.set(sw,errors);
    if( errors.length()!=0 ) error("problem reading COORD_SWITCH1 keyword : " + errors );
  } else {
    error("No coordination switching function 1 defined. Please define COORD_SWITCH1!");
  }
  parse("COORD_SWITCH2",sw);
  if(sw.length()>0){
    coordSwitchingFunction2.set(sw,errors);
    if( errors.length()!=0 ) error("problem reading COORD_SWITCH2 keyword : " + errors );
  } else {
    error("No coordination switching function 2 defined. Please define COORD_SWITCH2!");
  }
  parse("COORD_SWITCH3",sw);
  if(sw.length()>0){
    coordSwitchingFunction3.set(sw,errors);
    if( errors.length()!=0 ) error("problem reading COORD_SWITCH3 keyword : " + errors );
  } else {
    error("No coordination switching function 3 defined. Please define COORD_SWITCH3!");
  }

  checkRead();
}

Perovskite::~Perovskite(){
  delete nl;
}

void Perovskite::prepare(){
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
void Perovskite::calculate()
{
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

double CVvalue=0.;
vector<Vector> deriv(getNumberOfAtoms());
Tensor virial;
const unsigned nn=center_lista.size();
 // Loop over center atoms
 for(unsigned int i=rank;i<nn;i+=stride) {   
   // Loop over around species
   double coordination1=0., coordination2=0., coordination3=0.;
   vector<Vector> deriv1(getNumberOfAtoms()), deriv2(getNumberOfAtoms()), deriv3(getNumberOfAtoms());
   Tensor virial1, virial2, virial3;
   std::vector<unsigned> neighbors;
   neighbors=nl->getNeighbors(i);
   for(unsigned int j=0;j<neighbors.size();j+=1) {   
     double dfunc;
     Vector distance;
    // i different from j
    if (atomsToRequest[i].serial()!=atomsToRequest[neighbors[j]].serial()) { 
       distance = pbcDistance(getPosition(i),getPosition(neighbors[j]));
       if ( (neighbors[j]>=center_lista.size()) && (neighbors[j]<(center_lista.size()+around_species1_lista.size())) ) {
         // Atom belongs to around species 1
         coordination1 += switchingFunction1.calculateSqr(distance.modulo2(),dfunc);
         Vector dd(dfunc*distance);
         Tensor vv(dd,distance);
         virial1 -= vv;
         deriv1[i]-=dd;
         deriv1[neighbors[j]]+=dd;
         //log.printf("Atom %u with atom %u - type 1: Coordination %f and distance %f \n", atomsToRequest[i].serial(), atomsToRequest[neighbors[j]].serial(), switchingFunction1.calculateSqr(distance.modulo2(),dfunc), distance.modulo());
       } else if ( (neighbors[j]>=(center_lista.size()+around_species1_lista.size()) ) && (neighbors[j]<(center_lista.size()+around_species1_lista.size()+around_species2_lista.size())) ){
         // Atom belongs to around species 2
         coordination2 += switchingFunction2.calculateSqr(distance.modulo2(),dfunc);
         Vector dd(dfunc*distance);
         Tensor vv(dd,distance);
         virial2 -= vv;
         deriv2[i]-=dd;
         deriv2[neighbors[j]]+=dd;
       } else if ( neighbors[j]>=(center_lista.size()+around_species1_lista.size()+around_species2_lista.size()) ){
         // Atom belongs to around species 3
         coordination3 += switchingFunction3.calculateSqr(distance.modulo2(),dfunc);
         Vector dd(dfunc*distance);
         Tensor vv(dd,distance);
         virial3 -= vv;
         deriv3[i]-=dd;
         deriv3[neighbors[j]]+=dd;
       }
     }
   }
   //log.printf("Atom %u: Coordination 1 %f 2 %f and 3 %f \n", atomsToRequest[i].serial(), coordination1, coordination2, coordination3 );
   double dfunc1,dfunc2,dfunc3;
   double CVvalue1 = coordSwitchingFunction1.calculate(coordination1,dfunc1);
   double CVvalue2 = coordSwitchingFunction2.calculate(coordination2,dfunc2);
   double CVvalue3 = coordSwitchingFunction3.calculate(coordination3,dfunc3);
   dfunc1 *= coordination1;
   dfunc2 *= coordination2;
   dfunc3 *= coordination3;
   CVvalue += CVvalue1*CVvalue2*CVvalue3;
   for(unsigned int j=0;j<getNumberOfAtoms();j+=1) {   
     for(unsigned int k=0;k<3;k+=1) {   
       deriv[j][k] += deriv1[j][k]*dfunc1*CVvalue2*CVvalue3 + CVvalue1*deriv2[j][k]*dfunc2*CVvalue3 + CVvalue1*CVvalue2*deriv3[j][k]*dfunc3;
     }
   }
   for(unsigned int j=0;j<3;j+=1) {   
     for(unsigned int k=0;k<3;k+=1) {   
       virial[j][k] += virial1[j][k]*dfunc1*CVvalue2*CVvalue3 + CVvalue1*virial2[j][k]*dfunc2*CVvalue3 + CVvalue1*CVvalue2*virial3[j][k]*dfunc3;
     }
   }
 }
 if(!serial){
   comm.Sum(CVvalue);
   if(!deriv.empty()) comm.Sum(&deriv[0][0],3*deriv.size());
   comm.Sum(virial);
 }
 setValue(CVvalue);
 for(unsigned i=0;i<deriv.size();++i) setAtomsDerivatives(i,deriv[i]);
 setBoxDerivatives  (virial);
}

}
}
