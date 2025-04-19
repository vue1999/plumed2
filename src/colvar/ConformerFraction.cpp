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
#include "tools/OFile.h"
#include "tools/SwitchingFunction.h"

#include <string>
#include <math.h>

using namespace std;

namespace PLMD{
namespace colvar{

class ConformerFraction : public Colvar {
  bool serial;
  vector<AtomNumber> center_lista;
  vector<AtomNumber> vector1start_lista,vector1end_lista;
  vector<AtomNumber> vector2start_lista,vector2end_lista;
  vector<AtomNumber> vector3start_lista,vector3end_lista;
  std::vector<PLMD::AtomNumber> atomsToRequest;
  double cutoff, cutoffsqr;
  bool doOutputXYZ;
  mutable PLMD::OFile outputFile;
  SwitchingFunction switchingFunction;
public:
  explicit ConformerFraction(const ActionOptions&);
  ~ConformerFraction();
  virtual void calculate();
  virtual void prepare();
  static void registerKeywords( Keywords& keys );
  void outputXYZ(std::vector<double>);
};

PLUMED_REGISTER_ACTION(ConformerFraction,"CONFORMERFRACTION")

void ConformerFraction::registerKeywords( Keywords& keys ){
  Colvar::registerKeywords(keys);
  keys.addFlag("SERIAL",false,"Perform the calculation in serial - for debug purpose");
  keys.add("atoms","CENTER","Center of molecule");
  keys.add("atoms","VECTOR1START","Start point of first vector");
  keys.add("atoms","VECTOR1END"  ,"End point of first vector");
  keys.add("atoms","VECTOR2START","Start point of second vector");
  keys.add("atoms","VECTOR2END"  ,"End point of second vector");
  keys.add("atoms","VECTOR3START","Start point of third vector");
  keys.add("atoms","VECTOR3END"  ,"End point of third vector");
  //keys.add("compulsory","CUTOFF","1","Cutoff");
  keys.add("compulsory","SWITCH","Swiching function");
  keys.addFlag("OUTPUT_XYZ",false,"Output xyz file");
  keys.addOutputComponent("fraction","COMPONENTS","fraction of up spins");
  keys.addOutputComponent("corr","COMPONENTS","corrrelation between up and down spins");
}

ConformerFraction::ConformerFraction(const ActionOptions&ao):
PLUMED_COLVAR_INIT(ao)
{
  parseFlag("SERIAL",serial);
  //parse("CUTOFF",cutoff);
  //cutoffsqr = cutoff*cutoff;
  parseAtomList("CENTER",center_lista);
  parseAtomList("VECTOR1START",vector1start_lista);
  parseAtomList("VECTOR1END",vector1end_lista);
  parseAtomList("VECTOR2START",vector2start_lista);
  parseAtomList("VECTOR2END",vector2end_lista);
  parseAtomList("VECTOR3START",vector3start_lista);
  parseAtomList("VECTOR3END",vector3end_lista);
  //if(center_lista.size()!=start1_lista.size()) error("Number of atoms in START1 must be equal to the number of atoms in CENTER");

  doOutputXYZ=false;
  parseFlag("OUTPUT_XYZ",doOutputXYZ);
  if (doOutputXYZ) { 
     log.printf("  An xyz file with the spin will be written \n");
     outputFile.link(*this);
     outputFile.open("spin.xyz");
  }

  string sw,errors;
  parse("SWITCH",sw);
  if(sw.length()>0) {
    switchingFunction.set(sw,errors);
    if( errors.length()!=0 ) error("problem reading SWITCH keyword : " + errors );
  }

  double cutoff = switchingFunction.get_dmax();
  cutoffsqr = cutoff*cutoff;

  addComponentWithDerivatives("fraction"); componentIsNotPeriodic("fraction");
  addComponentWithDerivatives("corr"); componentIsNotPeriodic("corr");

  checkRead();
  atomsToRequest.reserve ( 7*center_lista.size() );
  atomsToRequest.insert (atomsToRequest.end(), center_lista.begin(), center_lista.end() );
  atomsToRequest.insert (atomsToRequest.end(), vector1start_lista.begin(), vector1start_lista.end() );
  atomsToRequest.insert (atomsToRequest.end(), vector1end_lista.begin()  , vector1end_lista.end() );
  atomsToRequest.insert (atomsToRequest.end(), vector2start_lista.begin(), vector2start_lista.end() );
  atomsToRequest.insert (atomsToRequest.end(), vector2end_lista.begin()  , vector2end_lista.end() );
  atomsToRequest.insert (atomsToRequest.end(), vector3start_lista.begin(), vector3start_lista.end() );
  atomsToRequest.insert (atomsToRequest.end(), vector3end_lista.begin()  , vector3end_lista.end() );
  requestAtoms(atomsToRequest);
}

ConformerFraction::~ConformerFraction(){}

void ConformerFraction::prepare(){}

// calculator
void ConformerFraction::calculate()
{
  std::vector<Vector> deriv_spin(getNumberOfAtoms());
  int number_molecules = center_lista.size();
  std::vector<double> spin(number_molecules);
  double avgspin = 0.;
  for(int i=0;i<number_molecules;i+=1) {
    /*
    log.printf("atom1 x y z %f %f %f \n", getPosition(i)[0]*10., getPosition(i)[1]*10., getPosition(i)[2]*10.);
    log.printf("atom2 x y z %f %f %f \n", getPosition(i+1*number_molecules)[0]*10,getPosition(i+1*number_molecules)[1]*10,getPosition(i+1*number_molecules)[2]*10);
    */
    Vector vector1=pbcDistance(getPosition(i+1*number_molecules),getPosition(i+2*number_molecules));
    Vector vector2=pbcDistance(getPosition(i+3*number_molecules),getPosition(i+4*number_molecules));
    Vector vector3=pbcDistance(getPosition(i+5*number_molecules),getPosition(i+6*number_molecules));
    double norm_v1 = std::sqrt(vector1[0]*vector1[0]+vector1[1]*vector1[1]+vector1[2]*vector1[2]);
    double norm_v2 = std::sqrt(vector2[0]*vector2[0]+vector2[1]*vector2[1]+vector2[2]*vector2[2]);
    double norm_v3 = std::sqrt(vector3[0]*vector3[0]+vector3[1]*vector3[1]+vector3[2]*vector3[2]);
    double inv_norm_v1 = 1. / norm_v1;
    double inv_norm_v2 = 1. / norm_v2;
    double inv_norm_v3 = 1. / norm_v3;
    vector1 *= inv_norm_v1;
    vector2 *= inv_norm_v2;
    vector3 *= inv_norm_v3;
    //double inv_v3=1./norm_v3;
    Vector vector4=crossProduct(vector1,vector2); 
    //double norm_v4 = std::sqrt(vector4[0]*vector4[0]+vector4[1]*vector4[1]+vector4[2]*vector4[2]);
    spin[i] = dotProduct(vector3,vector4);
    //log.printf("%f \n", spin[i]);
    avgspin += spin[i];
    Vector deriv_vector1_x = inv_norm_v1*Vector(1,0,0) - vector1*vector1[0]*inv_norm_v1;
    Vector deriv_vector1_y = inv_norm_v1*Vector(0,1,0) - vector1*vector1[1]*inv_norm_v1;
    Vector deriv_vector1_z = inv_norm_v1*Vector(0,0,1) - vector1*vector1[2]*inv_norm_v1;
    Vector deriv_vector2_x = inv_norm_v2*Vector(1,0,0) - vector2*vector2[0]*inv_norm_v2;
    Vector deriv_vector2_y = inv_norm_v2*Vector(0,1,0) - vector2*vector2[1]*inv_norm_v2;
    Vector deriv_vector2_z = inv_norm_v2*Vector(0,0,1) - vector2*vector2[2]*inv_norm_v2;
    Vector deriv_vector3_x = inv_norm_v3*Vector(1,0,0) - vector3*vector3[0]*inv_norm_v3;
    Vector deriv_vector3_y = inv_norm_v3*Vector(0,1,0) - vector3*vector3[1]*inv_norm_v3;
    Vector deriv_vector3_z = inv_norm_v3*Vector(0,0,1) - vector3*vector3[2]*inv_norm_v3;
    double deriv_spin_vector1_x = dotProduct(crossProduct(deriv_vector1_x,vector2),vector3);
    double deriv_spin_vector1_y = dotProduct(crossProduct(deriv_vector1_y,vector2),vector3);
    double deriv_spin_vector1_z = dotProduct(crossProduct(deriv_vector1_z,vector2),vector3);
    deriv_spin[i+1*number_molecules][0] = -deriv_spin_vector1_x;
    deriv_spin[i+1*number_molecules][1] = -deriv_spin_vector1_y;
    deriv_spin[i+1*number_molecules][2] = -deriv_spin_vector1_z;
    deriv_spin[i+2*number_molecules][0] =  deriv_spin_vector1_x;
    deriv_spin[i+2*number_molecules][1] =  deriv_spin_vector1_y;
    deriv_spin[i+2*number_molecules][2] =  deriv_spin_vector1_z;
    double deriv_spin_vector2_x = dotProduct(crossProduct(vector1,deriv_vector2_x),vector3);
    double deriv_spin_vector2_y = dotProduct(crossProduct(vector1,deriv_vector2_y),vector3);
    double deriv_spin_vector2_z = dotProduct(crossProduct(vector1,deriv_vector2_z),vector3);
    deriv_spin[i+3*number_molecules][0] = -deriv_spin_vector2_x;
    deriv_spin[i+3*number_molecules][1] = -deriv_spin_vector2_y;
    deriv_spin[i+3*number_molecules][2] = -deriv_spin_vector2_z;
    deriv_spin[i+4*number_molecules][0] =  deriv_spin_vector2_x;
    deriv_spin[i+4*number_molecules][1] =  deriv_spin_vector2_y;
    deriv_spin[i+4*number_molecules][2] =  deriv_spin_vector2_z;
    double deriv_spin_vector3_x = dotProduct(vector4,deriv_vector3_x);
    double deriv_spin_vector3_y = dotProduct(vector4,deriv_vector3_y);
    double deriv_spin_vector3_z = dotProduct(vector4,deriv_vector3_z);
    deriv_spin[i+5*number_molecules][0] = -deriv_spin_vector3_x;
    deriv_spin[i+5*number_molecules][1] = -deriv_spin_vector3_y;
    deriv_spin[i+5*number_molecules][2] = -deriv_spin_vector3_z;
    deriv_spin[i+6*number_molecules][0] =  deriv_spin_vector3_x;
    deriv_spin[i+6*number_molecules][1] =  deriv_spin_vector3_y;
    deriv_spin[i+6*number_molecules][2] =  deriv_spin_vector3_z;
  }
  if (doOutputXYZ) { 
    outputXYZ(spin);
  }
  avgspin /= number_molecules;
  double fraction = (avgspin+1)/2.;
  Value* valuefraction=getPntrToComponent("fraction");
  valuefraction->set(fraction);
  for(unsigned i=0;i<deriv_spin.size();++i) {
    Vector deriv = deriv_spin[i] / (2*number_molecules);
    setAtomsDerivatives(valuefraction,i,deriv);
  }
  //setValue           (fraction);
  double corr=0.;
  double counter=0;
  std::vector<Vector> deriv(getNumberOfAtoms());
  /*
  std::vector<Vector> deriv_num(getNumberOfAtoms());
  std::vector<Vector> deriv_denom(getNumberOfAtoms());
  for(int i=0;i<number_molecules-1;i+=1) {
    for(int j=i+1;j<number_molecules;j+=1) {
      Vector distance = pbcDistance(getPosition(i),getPosition(j)); 
      double dist = distance.modulo();
      if (dist < cutoff) {
        double df;
        double sw = switchingFunction.calculateSqr( d2, df);
        corr += spin[i]*spin[j]*sw;
        counter += sw;
      }
    }
  }
  corr /= counter;
  */
  Value* valuecorr=getPntrToComponent("corr");
  valuecorr->set(corr);
  for(unsigned i=0;i<deriv.size();++i) setAtomsDerivatives(valuecorr,i,deriv[i]);
  //setBoxDerivatives  (virial);
}

void ConformerFraction::outputXYZ(std::vector<double> spin) {
  outputFile.printf("%d \n \n",spin.size());
  for(unsigned i=0;i<spin.size();++i){
    outputFile.printf("X %f %f %f %f \n", getPosition(i)[0]*10,getPosition(i)[1]*10,getPosition(i)[2]*10,spin[i]);
  }
}


}
}
