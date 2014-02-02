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

#ifndef BACKEND_H
#define BACKEND_H

#include "som_types.h"

namespace RSOM {

/** abstract backend prototype for SOM class. */
class Backend
{
public:
    Backend() { }
    virtual ~Backend() { }

    /** sets parameters and inits additional memory.
        @return success. */
    virtual bool setMemory(Index sizex, Index sizey, Index dim) = 0;

    // --- upload data ---

    /** transfer map data */
    virtual bool uploadMap(const Float * map) = 0;

    /** transfer question vector */
    virtual bool uploadVec(Float * vec) = 0;

    // --- download data ---

    /** get current map [sizey][sizex][dim] */
    virtual bool downloadMap(Float * map) = 0;

    /** get current difference map */
    virtual bool downloadDMap(Float * dmap) = 0;

    // --- functions ---

    /** adjust the neighbourhood around x,y, with radius rx,ry, to uploaded vector. */
    virtual bool set(Index x, Index y, Index rx, Index ry, Float amp) = 0;

    /** Calculates the distance of each cell to the
        previously uploaded vector.
        Result can be requested via downloadDMap(). */
    virtual bool calcDMap() = 0;

    /** return smallest dmap value in @p index. */
    virtual bool getMinDMap(Index& index) = 0;

};

} // namespace RSOM

#endif // BACKEND_H
