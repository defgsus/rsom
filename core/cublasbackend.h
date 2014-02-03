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
#ifndef CUBLASBACKEND_H
#define CUBLASBACKEND_H


#include "backend.h"
#include "cublas_v2.h"

namespace RSOM {

/** Cuda backend for SOM class. */
class CublasBackend : public Backend
{
public:
    CublasBackend(Index max_threads = 0);
    ~CublasBackend();

    std::string name() const;

    /** sets parameters and inits device memory.
        @return success. */
    bool setMemory(Index sizex, Index sizey, Index dim);

    /** free device memory, if any */
    bool free();

    // --- upload data ---

    bool uploadMap(const Float * map);

    bool uploadVec(const Float * vec);

    // --- download data ---

    bool downloadMap(Float * map);

    bool downloadDMap(Float * dmap);

    // --- functions ---

    /** adjust the neighbourhood around x,y, with radius rx,ry, to uploaded vector. */
    bool set(Index x, Index y, Index rx, Index ry, Float amp);

    /** Calculates the distance of each cell to the
        previously uploaded vector.
        Result can be requested via downloadDMap(). */
    bool calcDMap();

    /** return smallest dmap value in @p index. */
    bool getMinDMap(Index& index);

    bool debugFunc();

    // ------ public MEMBER ---------

    Index size, sizex, sizey, dim;

    static cublasHandle_t handle;

    Float
    /** 3d som map on device */
        * dev_map,
    /** 2d difference map */
        * dev_dmap,
    /** one vector of length CublasBackend::dim used for questions */
        * dev_vec;
    Index
    /** scratch space to find best match */
        * dev_idx;

    /** maximum number of threads.
        @todo this probably is more complicated. */
    Index max_threads;

};


} // namespace RSOM

#endif // CUBLASBACKEND_H