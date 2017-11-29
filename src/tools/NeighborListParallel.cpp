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
                           const bool& do_reduced_list, const double& distance, const int& stride, const double& skin): 
  do_pair_(do_pair), do_pbc_(do_pbc), pbc_(&pbc),
  distance_(distance), stride_(stride), mycomm(cc), mylog(log),
  skin_(skin), do_reduced_list_(do_reduced_list)
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
  positions_old_.resize(fullatomlist_.size());
  dangerousBuilds_=0;
  firsttime_=true;
  numberOfBuilds_=0;
  avgTotalNeighbors_=0.;
  maxLoadImbalance_=2.;
  avgLoadImbalance_=0.;
}

NeighborListParallel::NeighborListParallel(const vector<AtomNumber>& list0, const bool& do_pbc,
                           const Pbc& pbc, Communicator& cc, Log& log, const double& distance,
                           const bool& do_reduced_list, const int& stride, const double& skin):
  do_pbc_(do_pbc), pbc_(&pbc),
  distance_(distance), stride_(stride), mycomm(cc), mylog(log),
  skin_(skin) , do_reduced_list_(do_reduced_list)
{
  fullatomlist_=list0;
  nlist0_=list0.size();
  twolists_=false;
  nallpairs_=nlist0_*(nlist0_-1)/2;
  lastupdate_=0;
  positions_old_.resize(fullatomlist_.size());
  dangerousBuilds_=0;
  firsttime_=true;
  numberOfBuilds_=0;
  avgTotalNeighbors_=0.;
  maxLoadImbalance_=2.;
  avgLoadImbalance_=0.;
}

vector<AtomNumber>& NeighborListParallel::getFullAtomList() {
  return fullatomlist_;
}

bool NeighborListParallel::isListStillGood(const vector<Vector>& positions) {
  bool flag=true;
  plumed_assert(positions.size()==fullatomlist_.size());
  for(unsigned int i=0;i<fullatomlist_.size();i++) {
    Vector distance;
    if(do_pbc_) {
       distance=pbc_->distance(positions[i],positions_old_[i]);
    } else {
       distance=delta(positions[i],positions_old_[i]);
    }
    if (modulo(distance)>skin_) {
       flag=false;
       break;
    }
  }
  return flag;
}

void NeighborListParallel::printStats() {
  mylog.printf("Neighbor list statistics\n");
  mylog.printf("Total # of neighbors = %f \n", avgTotalNeighbors_);
  mylog.printf("Ave neighs/atom = %f \n", avgTotalNeighbors_ /(double) nlist0_);
  mylog.printf("Neighbor list builds = %d \n",numberOfBuilds_);
  mylog.printf("Dangerous builds = %d \n",dangerousBuilds_);
  mylog.printf("Average load imbalance (min/max) = %f \n",avgLoadImbalance_);
  mylog.printf("Maximum load imbalance (min/max) = %f \n",maxLoadImbalance_);
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
  // If needed, build a shared neighbor list between processors
  if (!do_reduced_list_ && mpi_stride>1) {
     //unsigned allNeighNum=neighbors_.size();
     //mycomm.Sum(allNeighNum);
     //std::cout << neighbors_.size() << " " << allNeighNum << "\n";
     std::vector<unsigned> neighbors_ranks_(mycomm.Get_size());
     unsigned neighNum = neighbors_.size();
     mycomm.Allgather(&neighNum,1,&neighbors_ranks_[0],1);
     unsigned allNeighNum=0;
     for(unsigned int i=0;i<mycomm.Get_size();i+=1) allNeighNum+=neighbors_ranks_[i];
     std::vector<std::pair<unsigned,unsigned> > all_neighbors_(allNeighNum);
     std::vector<unsigned> sum_neighbors_ranks_(mpi_stride);
     for(unsigned int i=1;i<mpi_stride;i+=1) {
        for(unsigned int j=0;j<i;j+=1) {
           sum_neighbors_ranks_[i] += neighbors_ranks_[j];
        }
     }
     for(unsigned int i=0;i<neighbors_.size();i+=1) {
        all_neighbors_[sum_neighbors_ranks_[mpi_rank]+i]=neighbors_[i];
     }
     mycomm.Sum(&all_neighbors_[0].first,2*allNeighNum);
     //mycomm.Allgather(&neighbors_[0].first,2*neighbors_.size(),&all_neighbors_[0].first,2*neighbors_.size());
     neighbors_ = all_neighbors_;
  }
  /*
  if (mpi_rank==0) {
     for(unsigned int i=0;i<neighbors_.size();i+=1) {
        std::cout << neighbors_[i].first << " " <<  neighbors_[i].second << "\n";
     }
  }
  */
  gatherStats(positions);
  // Store positions for checking
  for(unsigned int i=0;i<fullatomlist_.size();i++) {
     positions_old_[i]=positions[i];
  }
}

void NeighborListParallel::gatherStats(const vector<Vector>& positions) {
  // Check if rebuilt was dangerous
  if (!firsttime_ && !isListStillGood(positions)) {
     dangerousBuilds_++;
  }
  firsttime_=false;
  numberOfBuilds_++;
  std::vector<unsigned> neighbors_ranks_(mycomm.Get_size());
  unsigned neighNum = neighbors_.size();
  mycomm.Allgather(&neighNum,1,&neighbors_ranks_[0],1);
  unsigned allNeighNum=0;
  for(unsigned int i=0;i<mycomm.Get_size();i+=1) allNeighNum+=neighbors_ranks_[i];
  //for(unsigned int i=0;i<mpi_stride;i+=1) mylog.printf("core %d neighbors %d \n", i,neighbors_ranks_[i]);
  avgTotalNeighbors_ += (allNeighNum-avgTotalNeighbors_)/numberOfBuilds_;
  auto min_element_ = *std::min_element(neighbors_ranks_.begin(), neighbors_ranks_.end());
  auto max_element_ = *std::max_element(neighbors_ranks_.begin(), neighbors_ranks_.end());
  //mylog.printf("Value: Min %d max %d \n",min_element_,max_element_ );
  double loadImbalance=min_element_ / (double) max_element_;
  //mylog.printf("loadImbalance %f \n", loadImbalance);
  if (maxLoadImbalance_>loadImbalance) maxLoadImbalance_=loadImbalance;
  avgLoadImbalance_ += (loadImbalance-avgLoadImbalance_)/numberOfBuilds_;
}

int NeighborListParallel::getStride() const {
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
  if (!do_reduced_list_) plumed_merror("Cannot ask for individual pair when the the reduced neighbor list is not used.");
  return neighbors_[i];
}

vector<unsigned> NeighborListParallel::getNeighbors(unsigned index) {
  if (do_reduced_list_) plumed_merror("Cannot ask for all neighbors when the reduced neighbor list is used.");
  vector<unsigned> neighbors;
  for(unsigned int i=0; i<size(); ++i) {
    if(neighbors_[i].first==index)  neighbors.push_back(neighbors_[i].second);
    if(neighbors_[i].second==index) neighbors.push_back(neighbors_[i].first);
  }
  return neighbors;
}

}
