/***************************************************************************

Copyright (C) 2014  stefan.berke @ modular-audio-graphics.com

This source is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either
version 3.0 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this software; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA

****************************************************************************/

#ifndef RSOM_NO_CUDA

#ifndef CUDABACKEND_H
#define CUDABACKEND_H


#include "backend.h"


namespace RSOM {

/** Cuda backend for SOM class.
    See RSOM::Backend for documentation. */
class CudaBackend : public Backend
{
public:
    CudaBackend(Index max_threads);
    ~CudaBackend();

    std::string name() const;

    bool setMemory(Index sizex, Index sizey, Index dim);

    bool free();

    // --- upload data ---

    bool uploadMap(const Float * map);

    bool uploadIMap(const Index * imap);

    bool uploadVec(const Float * vec);

    // --- download data ---

    bool downloadMap(Float * map, Index z = 0, Index depth = 0);

    bool downloadIMap(Index * imap);

    bool downloadDMap(Float * dmap);

    bool downloadUMap(Float * dmap);

    // --- functions ---

    bool set(Index x, Index y, Index rx, Index ry, Float amp);

    bool setIMapValue(Index x, Index value);

    bool calcUMap();

    bool calcDMap(bool only_vacant = false, Float fixed_value = 0);

    bool calcDMap(Index x, Index y, Index w, Index h,
                  bool only_vacant = false, Float fixed_value = 0);

    bool getMinDMap(Index& index, Float& value);

    bool getMinDMap(Index& index, Float& value,
                    Index x, Index y, Index w, Index h);

    bool normalizeUMap(Float factor = 1);

    bool normalize(Float * map, Index size, Float factor = 1);

    bool debugFunc();

    // ------ public MEMBER ---------

    Index size, sizex, sizey, dim;

    Float
    /** 3d som map on device */
        * dev_map,
    /** 2d difference map */
        * dev_dmap,
    /** 2d neighbour difference */
        * dev_umap,
    /** one vector of length CudaBackend::dim used for questions */
        * dev_vec;
    Index
    /** 2d index map */
        * dev_imap,
    /** scratch space to have some bytes on device */
        * dev_scratch;

    /** maximum number of threads.
        @todo this probably is more complicated. */
    Index max_threads;

};


} // namespace RSOM

#endif // CUDABACKEND_H


#endif //#ifndef RSOM_NO_CUDA
