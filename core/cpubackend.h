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

#ifndef CPUBACKEND_H
#define CPUBACKEND_H

#include <vector>

#include "backend.h"


namespace RSOM {

/** Cpu backend for SOM class.
    Note that memory is copied to conform with CudaBackend. */
class CpuBackend : public Backend
{
public:
    CpuBackend();
    ~CpuBackend();

    std::string name() const { return "cpu"; }

    /** sets parameters and inits device memory.
        @return success. */
    bool setMemory(Index sizex, Index sizey, Index dim);

    /** free memory, if any */
    bool free();

    // --- upload data ---

    bool uploadMap(const Float * map);

    bool uploadIMap(const Index * imap);

    bool uploadVec(const Float * vec);

    // --- download data ---

    bool downloadMap(Float * map, Index z = 0, Index depth = 0);

    bool downloadIMap(Index * imap);

    bool downloadDMap(Float * dmap);

    // --- functions ---

    /** adjust the neighbourhood around x,y, with radius rx,ry, to uploaded vector. */
    bool set(Index x, Index y, Index rx, Index ry, Float amp);

    /** change the @p x th value in imap to @p value. */
    bool setIMapValue(Index x, Index value);

    /** Calculates the distance of each cell to the
        previously uploaded vector.
        Result can be requested via downloadDMap(). */
    bool calcDMap();

    /** return index to smallest dmap value in @p index. */
    bool getMinDMap(Index& index, bool );

    bool debugFunc();

    // ------ public MEMBER ---------

    Index size, sizex, sizey, dim;

    std::vector<Float>
    /** 3d som map */
        cpu_map,
    /** 2d difference map */
        cpu_dmap,
    /** one vector of length CudaBackend::dim used for questions */
        cpu_vec;
    std::vector<Index>
    /** 2d index map */
        cpu_imap;
};


} // namespace RSOM

#endif // CPUBACKEND_H
