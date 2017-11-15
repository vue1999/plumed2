/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2011-2017 The plumed team
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
#include "NeighborListParallel.h"
#include "Vector.h"
#include "Pbc.h"
#include "AtomNumber.h"
#include "Tools.h"
#include <vector>
#include <algorithm>
#include "Communicator.h"
#include "Log.h"

namespace PLMD {
using namespace std;

NeighborListParallel::NeighborListParallel(const vector<AtomNumber>& list0, const vector<AtomNumber>& list1,
                           const bool& do_pair, const bool& do_pbc, const Pbc& pbc, Communicator& cc, Log& log,
                           const double& distance, const unsigned& stride): reduced(false),
  do_pair_(do_pair), do_pbc_(do_pbc), pbc_(&pbc),
  distance_(distance), stride_(stride), mycomm(cc), mylog(log) 
{
// store full list of atoms needed
  fullatomlist_=list0;
  fullatomlist_.insert(fullatomlist_.end(),list1.begin(),list1.end());
  nlist0_=list0.size();
  nlist1_=list1.size();
  twolists_=true;
  if(!do_pair) {
    nallpairs_=nlist0_*nlist1_;
  } else {
    plumed_assert(nlist0_==nlist1_);
    nallpairs_=nlist0_;
  }
  lastupdate_=0;
}

NeighborListParallel::NeighborListParallel(const vector<AtomNumber>& list0, const bool& do_pbc,
                           const Pbc& pbc, Communicator& cc, Log& log, const double& distance,
                           const unsigned& stride): reduced(false),
  do_pbc_(do_pbc), pbc_(&pbc),
  distance_(distance), stride_(stride), mycomm(cc), mylog(log) {
  fullatomlist_=list0;
  nlist0_=list0.size();
  twolists_=false;
  nallpairs_=nlist0_*(nlist0_-1)/2;
  lastupdate_=0;
}

vector<AtomNumber>& NeighborListParallel::getFullAtomList() {
  return fullatomlist_;
}

void NeighborListParallel::update(const vector<Vector>& positions) {
  neighbors_.clear();
  unsigned mpi_rank=mycomm.Get_rank();
  unsigned mpi_stride=mycomm.Get_size();
  const double d2=distance_*distance_;
// check if positions array has the correct length
  plumed_assert(positions.size()==fullatomlist_.size());
  if (!twolists_) {
    for(unsigned int i=mpi_rank;i<(nlist0_-1);i+=mpi_stride) {
       for(unsigned int j=i+1;j<nlist0_;j+=1) {
          Vector distance;
          if(do_pbc_) {
            distance=pbc_->distance(positions[i],positions[j]);
          } else {
            distance=delta(positions[i],positions[j]);
          }
          double value=modulo2(distance);
          if(value<=d2) neighbors_.push_back(pair<unsigned,unsigned>(i,j));
       }
    }
  } else if(twolists_ && do_pair_) {
    for(unsigned int i=0;i<nlist0_;i+=1) {
       Vector distance;
       if(do_pbc_) {
         distance=pbc_->distance(positions[i],positions[nlist0_+i]);
       } else {
         distance=delta(positions[i],positions[nlist0_+i]);
       }
       double value=modulo2(distance);
       if(value<=d2) neighbors_.push_back(pair<unsigned,unsigned>(i,nlist0_+i));
    }
  } else if (twolists_ && !do_pair_) {
    for(unsigned int i=mpi_rank;i<nlist0_;i+=mpi_stride) {
       for(unsigned int j=0;j<nlist1_;j+=1) {
          Vector distance;
          if(do_pbc_) {
            distance=pbc_->distance(positions[i],positions[nlist0_+j]);
          } else {
            distance=delta(positions[i],positions[nlist0_+j]);
          }
          double value=modulo2(distance);
          if(value<=d2) neighbors_.push_back(pair<unsigned,unsigned>(i,nlist0_+j));
       }
    }
  }
  /*
  for(unsigned int i=mpi_rank; i<nallpairs_; i+=mpi_stride) {
  */
}

unsigned NeighborListParallel::getStride() const {
  return stride_;
}

unsigned NeighborListParallel::getLastUpdate() const {
  return lastupdate_;
}

void NeighborListParallel::setLastUpdate(unsigned step) {
  lastupdate_=step;
}

unsigned NeighborListParallel::size() const {
  return neighbors_.size();
}

pair<unsigned,unsigned> NeighborListParallel::getClosePair(unsigned i) const {
  return neighbors_[i];
}

vector<unsigned> NeighborListParallel::getNeighbors(unsigned index) {
  vector<unsigned> neighbors;
  for(unsigned int i=0; i<size(); ++i) {
    if(neighbors_[i].first==index)  neighbors.push_back(neighbors_[i].second);
    if(neighbors_[i].second==index) neighbors.push_back(neighbors_[i].first);
  }
  return neighbors;
}

}
