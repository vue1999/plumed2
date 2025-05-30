/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2014-2023 The plumed team
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
#ifndef __PLUMED_tools_OpenMP_h
#define __PLUMED_tools_OpenMP_h

#include <vector>

namespace PLMD {

namespace OpenMP {

/// Set number of threads that can be used by openMP
void setNumThreads(const unsigned nt);

/// Get number of threads that can be used by openMP
unsigned getNumThreads();

/// Returns a unique thread identification number within the current team
unsigned getThreadNum();

/// get cacheline size
unsigned getCachelineSize();

/// Get a reasonable number of threads so as to access to an array of size s located at x
template<typename T>
unsigned getGoodNumThreads(const T* /*getTheType*/,unsigned n) {
  // this is more or less the equivalent of writing "unsigned getGoodNumThreads<T>(unsigned n)"

  // a factor two is necessary since there is no guarantee that x is aligned
  // to cache line boundary
  unsigned m=n*sizeof(T)/(2*getCachelineSize());
  unsigned numThreads=getNumThreads();
  if(m>=numThreads) {
    m=numThreads;
  } else {
    //it is better to use either all the active threads or only one
    //this solves a performance problem as explained in issue #415
    m=1;
  }
  return m;
}

/// Get a reasonable number of threads so as to access to vector v
template<typename T>
unsigned getGoodNumThreads(const std::vector<T> & v) {
  if(v.size()==0) {
    return 1;
  } else {
    return getGoodNumThreads(&v[0],v.size());
  }
}

}//namespace OpenMP
}//namespace PLMD

#endif
