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

namespace PLMD {
using namespace std;

NeighborListParallel::NeighborListParallel(const vector<AtomNumber>& list0, const vector<AtomNumber>& list1,
                           const bool& do_pair, const bool& do_pbc, const Pbc& pbc, const Communicator& cc,
                           const double& distance, const unsigned& stride): reduced(false),
  do_pair_(do_pair), do_pbc_(do_pbc), pbc_(&pbc),
  distance_(distance), stride_(stride), comm(cc)
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
  initialize();
  lastupdate_=0;
}

NeighborListParallel::NeighborListParallel(const vector<AtomNumber>& list0, const bool& do_pbc,
                           const Pbc& pbc, const Communicator& cc, const double& distance,
                           const unsigned& stride): reduced(false),
  do_pbc_(do_pbc), pbc_(&pbc),
  distance_(distance), stride_(stride), comm(cc) {
  fullatomlist_=list0;
  nlist0_=list0.size();
  twolists_=false;
  nallpairs_=nlist0_*(nlist0_-1)/2;
  initialize();
  lastupdate_=0;
}

void NeighborListParallel::initialize() {
  neighbors_.clear();
  for(unsigned int i=0; i<nallpairs_; ++i) {
    neighbors_.push_back(getIndexPair(i));
  }
}

vector<AtomNumber>& NeighborListParallel::getFullAtomList() {
  return fullatomlist_;
}

pair<unsigned,unsigned> NeighborListParallel::getIndexPair(unsigned ipair) {
  pair<unsigned,unsigned> index;
  if(twolists_ && do_pair_) {
    index=pair<unsigned,unsigned>(ipair,ipair+nlist0_);
  } else if (twolists_ && !do_pair_) {
    index=pair<unsigned,unsigned>(ipair/nlist1_,ipair%nlist1_+nlist0_);
  } else if (!twolists_) {
    unsigned ii = nallpairs_-1-ipair;
    unsigned  K = unsigned(floor((sqrt(double(8*ii+1))+1)/2));
    unsigned jj = ii-K*(K-1)/2;
    index=pair<unsigned,unsigned>(nlist0_-1-K,nlist0_-1-jj);
  }
  return index;
}

void NeighborListParallel::update(const vector<Vector>& positions) {
  neighbors_.clear();
  std::vector<std::pair<unsigned,unsigned> > neighbors_rank_;
  unsigned rank=comm.Get_rank();
  unsigned stride=comm.Get_size();
  const double d2=distance_*distance_;
// check if positions array has the correct length
  plumed_assert(positions.size()==fullatomlist_.size());
// Parallelize
  for(unsigned int i=rank; i<nallpairs_; ++stride) {
    pair<unsigned,unsigned> index=getIndexPair(i);
    unsigned index0=index.first;
    unsigned index1=index.second;
    Vector distance;
    if(do_pbc_) {
      distance=pbc_->distance(positions[index0],positions[index1]);
    } else {
      distance=delta(positions[index0],positions[index1]);
    }
    double value=modulo2(distance);
    if(value<=d2) {neighbors_rank_.push_back(index);}
  }
// Allocate vector neighbors_
  unsigned neighbors_size = neighbors_rank_.size();
  comm.Sum(neighbors_size);
// Join neighbors_rank_ vectors to form one neighbors_ vector
  neighbors_.resize(neighbors_size);
  comm.Allgather(&neighbors_rank_[0].first,2*neighbors_rank_.size(),&neighbors_[0].first,2*neighbors_size);
  setRequestList();
}

void NeighborListParallel::setRequestList() {
  requestlist_.clear();
  for(unsigned int i=0; i<size(); ++i) {
    requestlist_.push_back(fullatomlist_[neighbors_[i].first]);
    requestlist_.push_back(fullatomlist_[neighbors_[i].second]);
  }
  Tools::removeDuplicates(requestlist_);
  reduced=false;
}

vector<AtomNumber>& NeighborListParallel::getReducedAtomList() {
  if(!reduced)for(unsigned int i=0; i<size(); ++i) {
      unsigned newindex0=0,newindex1=0;
      AtomNumber index0=fullatomlist_[neighbors_[i].first];
      AtomNumber index1=fullatomlist_[neighbors_[i].second];
// I exploit the fact that requestlist_ is an ordered vector
      auto p = std::find(requestlist_.begin(), requestlist_.end(), index0); plumed_assert(p!=requestlist_.end()); newindex0=p-requestlist_.begin();
      p = std::find(requestlist_.begin(), requestlist_.end(), index1); plumed_assert(p!=requestlist_.end()); newindex1=p-requestlist_.begin();
      neighbors_[i]=pair<unsigned,unsigned>(newindex0,newindex1);
    }
  reduced=true;
  return requestlist_;
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
