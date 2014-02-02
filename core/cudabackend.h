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

#ifndef CUDABACKEND_H
#define CUDABACKEND_H

#include "som_types.h"

struct cudaMemcpy3DParms;

namespace RSOM {

/** backend for SOM class. */
class CudaBackend
{
public:
    CudaBackend();
    ~CudaBackend();

    /** disconnect from map, free device memory, if any */
    bool free();

    /** sets parameters and inits device memory.
        @return success. */
    bool setMemory(Index sizex, Index sizey, Index dim);

    // --- upload data ---

    bool uploadMap(const Float * map);

    bool uploadVec(Float * vec);

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

    // ------ public MEMBER ---------

    Index size, sizex, sizey, dim,
        idx_threads,
        idx_stride;

    Float
    /** 3d som map on device */
        * dev_map,
    /** 2d difference map */
        * dev_dmap,
    /** one vector of length CudaBackend::dim used for questions */
        * dev_vec;
    Index
    /** scratch space to find best match */
        * dev_idx;

private:

    cudaMemcpy3DParms * p_upload_, * p_download_;
};


} // namespace RSOM

#endif // CUDABACKEND_H
