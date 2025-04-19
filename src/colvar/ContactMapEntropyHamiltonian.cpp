/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2012-2018 The plumed team
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
#include "tools/NeighborList.h"
#include "ActionRegister.h"
#include "tools/SwitchingFunction.h"
#include "tools/Matrix.h"
#include "tools/IFile.h"
#include "tools/OFile.h"

#include <string>
#include <cmath>
#include <memory>

//A namespace defines an area of code in which all identifiers are guaranteed to be unique e.g
// and one calls it by : std::cout << std::PLMD::doSomething(4, 3) << '\n';
using namespace std;


namespace PLMD {   
namespace colvar {

//+PLUMEDOC COLVAR CONTACTMAP
/*
Calculate the distances between a number of pairs of atoms and transform each distance by a switching function.

The transformed distance can be compared with a reference value in order to calculate the squared distance
between two contact maps. Each distance can also be weighted for a given value. CONTACTMAP can be used together
with \ref FUNCPATHMSD to define a path in the contactmap space.

The individual contact map distances related to each contact can be accessed as components
named `cm.contact-1`, `cm.contact-2`, etc, assuming that the label of the CONTACTMAP is `cm`.

\par Examples

The following example calculates switching functions based on the distances between atoms
1 and 2, 3 and 4 and 4 and 5. The values of these three switching functions are then output
to a file named colvar.

\plumedfile
CONTACTMAP ATOMS1=1,2 ATOMS2=3,4 ATOMS3=4,5 ATOMS4=5,6 SWITCH={RATIONAL R_0=1.5} LABEL=f1
PRINT ARG=f1.* FILE=colvar
\endplumedfile

The following example calculates the difference of the current contact map with respect
to a reference provided. In this case REFERENCE is the fraction of contact that is formed
(i.e. the distance between two atoms transformed with the SWITH), while R_0 is the contact
distance. WEIGHT gives the relative weight of each contact to the final distance measure.

\plumedfile
CONTACTMAP ...
ATOMS1=1,2 REFERENCE1=0.1 WEIGHT1=0.5
ATOMS2=3,4 REFERENCE2=0.5 WEIGHT2=1.0
ATOMS3=4,5 REFERENCE3=0.25 WEIGHT3=1.0
ATOMS4=5,6 REFERENCE4=0.0 WEIGHT4=0.5
SWITCH={RATIONAL R_0=1.5}
LABEL=cmap
CMDIST
... CONTACTMAP

PRINT ARG=cmap FILE=colvar
\endplumedfile

The next example calculates calculates fraction of native contacts (Q)
for Trp-cage mini-protein. R_0 is the distance at which the switch function is guaranteed to
be 1.0 – it doesn't really matter for Q and  should be something very small, like 1 A.
REF is the reference distance for the contact, e.g. the distance from a crystal structure.
LAMBDA is the tolerance for the distance – if set to 1.0, the contact would have to have exactly
the reference value to be formed; instead for lambda values of 1.5–1.8 are usually used to allow some slack.
BETA is the softness of the switch function, default is 50nm.
WEIGHT is the 1/(number of contacts) giving equal weight to each contact.

When using native contact Q switch function, please cite \cite best2013

\plumedfile
# Full example available in regtest/basic/rt72/

CONTACTMAP ...
ATOMS1=1,67 SWITCH1={Q R_0=0.01 BETA=50.0 LAMBDA=1.5 REF=0.4059} WEIGHT1=0.003597
ATOMS2=1,68 SWITCH2={Q R_0=0.01 BETA=50.0 LAMBDA=1.5 REF=0.4039} WEIGHT2=0.003597
ATOMS3=1,69 SWITCH3={Q R_0=0.01 BETA=50.0 LAMBDA=1.5 REF=0.3215} WEIGHT3=0.003597
[snip]
ATOMS275=183,213 SWITCH275={Q R_0=0.01 BETA=50.0 LAMBDA=1.5 REF=0.355} WEIGHT275=0.003597
ATOMS276=183,234 SWITCH276={Q R_0=0.01 BETA=50.0 LAMBDA=1.5 REF=0.428} WEIGHT276=0.003597
ATOMS277=183,250 SWITCH277={Q R_0=0.01 BETA=50.0 LAMBDA=1.5 REF=0.3832} WEIGHT277=0.003597
ATOMS278=197,220 SWITCH278={Q R_0=0.01 BETA=50.0 LAMBDA=1.5 REF=0.3827} WEIGHT278=0.003597
LABEL=cmap
SUM
... CONTACTMAP

PRINT ARG=cmap FILE=colvar
\endplumedfile
(See also \ref switchingfunction)

*/
//+ENDPLUMEDOC

//###########################
//When you define a class, you define a blueprint for an object, that is what an object of the class will consist of and what operations can be performed on such an object
//We declare objects of a class with exactly the same sort of declaration that we declare variables of basic types.
//e.g ContactMapEntropyHamiltonian contacts    for defining the object contacts of ContactMapEntropyHamiltonian class.
//The public data members of objects of a class can be accessed using the direct member access operator (.)
//Inheritance allows us to define a class in terms of another class, which makes it easier to create and maintain an application.

//When creating a class, instead of writing completely new data members and member functions, the programmer can designate that the new class should inherit the members of an existing class. This existing class is called the base class, and the new class is referred to as the derived class.

//##########################

class ContactMapEntropyHamiltonian : public Colvar {
private:
  bool pbc;
  bool serial;
  bool components;
  double beta;
  bool write_files; 
  std::unique_ptr<NeighborList> nl;
  SwitchingFunction sf;
  PLMD:: OFile EigenvalOFile,EigenvecOFile, MatrixOFile;
public:
  static void registerKeywords( Keywords& keys );
  explicit ContactMapEntropyHamiltonian(const ActionOptions&);
// active methods:
  virtual void calculate();
  void checkFieldsAllowed() {}

};

PLUMED_REGISTER_ACTION(ContactMapEntropyHamiltonian,"CONTACTMAPENTROPYHAMILTONIAN")

void ContactMapEntropyHamiltonian::registerKeywords( Keywords& keys ) {
  Colvar::registerKeywords( keys );
  keys.add("atoms","ATOMS","List of atoms");
  keys.add("optional","BETA","Inverse temperature");
  keys.addFlag("COMPONENTS",false,"calculate the x, y and z components of the distance separately and store them as label.x, label.y and label.z");
  keys.addOutputComponent("entropy","COMPONENTS","the entropy-component of the contact map ");
  keys.addOutputComponent("energy","COMPONENTS","the energy-component of the contact map");
  keys.add("optional","SWITCH","This keyword is used if you want to employ an alternative to the continuous swiching function defined above. "
           "The following provides information on the \\ref switchingfunction that are available. "
           "When this keyword is present you no longer need the NN, MM, D_0 and R_0 keywords.");
  keys.addFlag("SERIAL",false,"Perform the calculation in serial - for debug purpose");
  keys.addFlag("WRITE_FILES",false,"Perform the calculation in serial - for debug purpose");

}

ContactMapEntropyHamiltonian::ContactMapEntropyHamiltonian(const ActionOptions&ao):
  PLUMED_COLVAR_INIT(ao),
  pbc(true),
  serial(false)
{
  parseFlag("SERIAL",serial);
  parseFlag("COMPONENTS",components);
  if (components) log.printf("  The energy, enrtopy will be computed separately \n");  

  parseFlag("WRITE_FILES",write_files);
  bool nopbc=!pbc;
  parseFlag("NOPBC",nopbc);
  pbc=!nopbc;
  // Read in the atoms
  vector<AtomNumber> atoms_lista;
  parseAtomList("ATOMS",atoms_lista);
  //beta=0.4009623;
  beta=1;
  parse("BETA",beta);
  // Read in the switching function
  string sw,errors;
  parse("SWITCH",sw);
  if(sw.length()>0) {
    sf.set(sw,errors);
    if( errors.length()!=0 ) error("problem reading SWITCH keyword : " + errors );
  }
  if(components) {
    addComponentWithDerivatives("energy"); componentIsNotPeriodic("energy");
    addComponentWithDerivatives("entropy"); componentIsNotPeriodic("entropy");
  }else{
    addValueWithDerivatives(); setNotPeriodic();
  }
  requestAtoms(atoms_lista);
  checkRead();
}

void ContactMapEntropyHamiltonian::calculate() {

  std::vector<Vector> deriventropy(getNumberOfAtoms());
  std::vector<Vector> derivenergy(getNumberOfAtoms());
  unsigned stride=comm.Get_size();
  unsigned rank=comm.Get_rank();
  if(serial) {
    // when using components the parallelisation do not work
    stride=1;
    rank=0;
  } else {
    stride=comm.Get_size();
    rank=comm.Get_rank();
  }
  // Define matrices
  Matrix<double> A(getNumberOfAtoms(),getNumberOfAtoms());
  Matrix<double> derivAx(getNumberOfAtoms(),getNumberOfAtoms());
  Matrix<double> derivAy(getNumberOfAtoms(),getNumberOfAtoms());
  Matrix<double> derivAz(getNumberOfAtoms(),getNumberOfAtoms());
  Matrix<double> eigenvecs(getNumberOfAtoms(),getNumberOfAtoms());
  std::vector<double> eigvals(getNumberOfAtoms());

  //derivAx=0.;
  //derivAy=0.;
  //derivAz=0.;
  //A=-0.;
  // Set diagonal elements to one (only for selected elements, otherwise they are summed in comm.Sum
  for(unsigned i=rank; i<getNumberOfAtoms(); i+=stride) A[i][i]=1.; // Check if this is right
// sum over close pairs
  for(unsigned i=rank; i<(getNumberOfAtoms()-1); i+=stride) {
    for(unsigned j=i+1; j<getNumberOfAtoms(); j+=1) {
      Vector distance;
      if(pbc) {
        distance=pbcDistance(getPosition(i),getPosition(j));
      } else {
        distance=delta(getPosition(i),getPosition(j));
      }
      double dfunc=0.;
      A[i][j] = -sf.calculate(distance.modulo(), dfunc);
      A[j][i] = A[i][j];
      if(!doNotCalculateDerivatives() ) {
        derivAx[i][j] = -dfunc*distance[0];
        derivAy[i][j] = -dfunc*distance[1];
        derivAz[i][j] = -dfunc*distance[2];
        derivAx[j][i] = derivAx[i][j] ; 
        derivAy[j][i] = derivAy[i][j] ; 
        derivAz[j][i] = derivAz[i][j] ;
      }
    }
  }
  if(!serial) {
    comm.Sum(A);
    if(!doNotCalculateDerivatives() ) {
      comm.Sum(derivAx);
      comm.Sum(derivAy);
      comm.Sum(derivAz);
    }
  }

  //Diagonalize 
  diagMat(A,eigvals,eigenvecs);
  double Z=0.;
  std::vector<double> rho(eigvals.size());
  // Define rho 
  for (unsigned j=0; j<eigvals.size(); ++j) {
    Z+=std::exp(-beta*eigvals[j]);
  }
  for (unsigned j=0; j<eigvals.size(); ++j) {
    //rho[j]=eigvals[j];
    rho[j]=std::exp(-beta*eigvals[j])/Z;
  }
  //cout << "eigenvalues " << eigvals[2] << " \n";
  // Get the maximum eigevalue
  //double lambdas = eigvals[ getNumberOfAtoms() ];
  double MatEntropy=0.;
  double MatEnergy=0.;
  // Calculate entropy-energy
  for(unsigned j=0; j<eigvals.size(); ++j) {
    log.printf("rho %d %f \n",j,rho[j]);
    if (rho[j]>1.e-10) MatEntropy-=rho[j]*std::log(rho[j]);
    MatEnergy+=eigvals[j]*rho[j];
  }

  MatEnergy/=getNumberOfAtoms();
  MatEntropy/=getNumberOfAtoms();

  //setValue           (MatEntropy);
  // Calculate derivative of entropy 
  if(!doNotCalculateDerivatives() ) {
    for(unsigned i=rank; i<getNumberOfAtoms(); i+=stride) {
      for(unsigned j=0; j<eigvals.size(); ++j) {
        double der_rho_wrt_lambdaX=0., der_rho_wrt_lambdaY=0., der_rho_wrt_lambdaZ=0.;
        double hellmanFeynmanTOTX=0., hellmanFeynmanTOTY=0., hellmanFeynmanTOTZ=0.;
        for(unsigned k=0; k<eigvals.size(); ++k) {
          //double der_rho_wrt_Hi = 1;
          double hellmanFeynmanX=0., hellmanFeynmanY=0., hellmanFeynmanZ=0.;
          for(unsigned l=0; l<getNumberOfAtoms(); ++l) {
            //log.printf("HF %f \n",hellmanFeynmanX);
            if (l<i && l!=i) {
              hellmanFeynmanX += eigenvecs[k][l]*derivAx[l][i]*eigenvecs[k][i];
              hellmanFeynmanY += eigenvecs[k][l]*derivAy[l][i]*eigenvecs[k][i];
              hellmanFeynmanZ += eigenvecs[k][l]*derivAz[l][i]*eigenvecs[k][i];
            } else if (l!=i) {
              hellmanFeynmanX -= eigenvecs[k][l]*derivAx[l][i]*eigenvecs[k][i];
              hellmanFeynmanY -= eigenvecs[k][l]*derivAy[l][i]*eigenvecs[k][i];
              hellmanFeynmanZ -= eigenvecs[k][l]*derivAz[l][i]*eigenvecs[k][i];
            }
          }
          hellmanFeynmanX *= 2;
          hellmanFeynmanY *= 2;
          hellmanFeynmanZ *= 2;
          double der_rho_wrt_Hi;
          if (k==j){
            der_rho_wrt_Hi = -beta*(rho[k] - rho[k]*rho[k]);
          }else{
           der_rho_wrt_Hi = beta * rho[j] * rho[k];
          }
          der_rho_wrt_lambdaX += der_rho_wrt_Hi*hellmanFeynmanX;
          der_rho_wrt_lambdaY += der_rho_wrt_Hi*hellmanFeynmanY;
          der_rho_wrt_lambdaZ += der_rho_wrt_Hi*hellmanFeynmanZ;
          if (k==j) {
            hellmanFeynmanTOTX=hellmanFeynmanX;
            hellmanFeynmanTOTY=hellmanFeynmanY;
            hellmanFeynmanTOTZ=hellmanFeynmanZ;
          }
        }
        //deriv[i][0] -= (1.0+std::log(rho[j]))*hellmanFeynmanX*der_rho_wrt_Hi;             
        //deriv[i][1] -= (1.0+std::log(rho[j]))*hellmanFeynmanY*der_rho_wrt_Hi;
        //deriv[i][2] -= (1.0+std::log(rho[j]))*hellmanFeynmanZ*der_rho_wrt_Hi;
        deriventropy[i][0] -= (1.0+std::log(rho[j]))*der_rho_wrt_lambdaX;             
        deriventropy[i][1] -= (1.0+std::log(rho[j]))*der_rho_wrt_lambdaY;
        deriventropy[i][2] -= (1.0+std::log(rho[j]))*der_rho_wrt_lambdaZ;
        derivenergy[i][0] += hellmanFeynmanTOTX*rho[j]+eigvals[j]*der_rho_wrt_lambdaX;   
        derivenergy[i][1] += hellmanFeynmanTOTY*rho[j]+eigvals[j]*der_rho_wrt_lambdaY;
        derivenergy[i][2] += hellmanFeynmanTOTZ*rho[j]+eigvals[j]*der_rho_wrt_lambdaZ;
      }
      deriventropy[i][0] /= getNumberOfAtoms();
      deriventropy[i][1] /= getNumberOfAtoms();
      deriventropy[i][2] /= getNumberOfAtoms();
      derivenergy[i][0] /= getNumberOfAtoms();
      derivenergy[i][1] /= getNumberOfAtoms();
      derivenergy[i][2] /= getNumberOfAtoms();
    }
  }



  //if(!doNotCalculateDerivatives() ) {
  //  if(!serial) {
  //    comm.Sum(&deriv[0][0],3*deriv.size());
  //  }
  //  for(unsigned i=0; i<deriv.size(); ++i) setAtomsDerivatives(i,deriv[i]);
  // }

  if (components) {
    Value* ValueMatEntropy=getPntrToComponent("entropy");
    Value* ValueMatEnergy=getPntrToComponent("energy");
    ValueMatEnergy->set(MatEnergy);
    ValueMatEntropy->set(MatEntropy);
  } else {
    setValue           (MatEntropy);
  }


  //setValue           (MatEntropy);
  if(!doNotCalculateDerivatives() ) {
    if(!serial) {
      comm.Sum(&deriventropy[0][0],3*deriventropy.size());
      comm.Sum(&derivenergy[0][0],3*derivenergy.size());
    }
    if (components) {
      Value* ValueMatEntropy=getPntrToComponent("entropy");  
      Value* ValueMatEnergy=getPntrToComponent("energy");       
      for(unsigned i=0; i<deriventropy.size(); ++i) setAtomsDerivatives(ValueMatEntropy,i,deriventropy[i]);
      for(unsigned i=0; i<derivenergy.size(); ++i) setAtomsDerivatives(ValueMatEnergy,i,derivenergy[i]);
    }else{
      for(unsigned i=0; i<deriventropy.size(); ++i) setAtomsDerivatives(i,deriventropy[i]);
    }
  }
 


  if (write_files){
    EigenvalOFile.link(*this);
    EigenvalOFile.open("Eigenval.txt");
    //EigenvalOFile.printField("%s\n","");
    EigenvecOFile.link(*this);
    MatrixOFile.link(*this);
    EigenvecOFile.open("Eigenvec.txt");
    MatrixOFile.open("Matrix.txt");
    for(unsigned i=0; i<getNumberOfAtoms(); i++) {
      //print eigenval
      EigenvalOFile.printField("eigenvalues",rho[i]).printField();
      for (unsigned j=0; j<getNumberOfAtoms(); j++) {
        EigenvecOFile.printf("%f %s",eigenvecs[i][j],"");
        MatrixOFile.printf("%f ",A[i][j]);
      }
      EigenvecOFile.printf("%s\n","");
      MatrixOFile.printf("%s\n","");
    }
    EigenvalOFile.close();
    EigenvecOFile.close();
    MatrixOFile.close();
  } 
}

}
}
