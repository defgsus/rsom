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

#include <string>

#include "som_types.h"

namespace RSOM {

/** abstract backend prototype for SOM class. */
class Backend
{
public:
    Backend() { }
    virtual ~Backend() { }

    /** descriptive runtime name */
    virtual std::string name() const = 0;

    /** sets parameters and inits additional memory.
        @return success. */
    virtual bool setMemory(Index sizex, Index sizey, Index dim) = 0;

    /** free allocated memory, invalidate backend */
    virtual bool free() = 0;

    // --- upload data ---

    /** transfer 3d som data [dim][sizey][sizex] */
    virtual bool uploadMap(const Float * map) = 0;

    /** transfer index map data [sizey][sizex] */
    virtual bool uploadIMap(const Index * imap) = 0;

    /** transfer question vector */
    virtual bool uploadVec(const Float * vec) = 0;

    // --- download data ---

    /** get current map [dim][sizey][sizex].
        If @p z and @p depth == 0, then the whole map is downloaded.
        To download one or several slices of the map use @p z as
        the index to the dimension and @p depth as the number of
        slices.
        The @p depth is alway set to the maximum when left zero (eg. dim - z). */
    virtual bool downloadMap(Float * map, Index z = 0, Index depth = 0) = 0;

    /** get current index map */
    virtual bool downloadIMap(Index * imap) = 0;

    /** get current difference map */
    virtual bool downloadDMap(Float * dmap) = 0;

    // --- functions ---

    /** adjust the neighbourhood around x,y, with radius rx,ry, to uploaded vector. */
    virtual bool set(Index x, Index y, Index rx, Index ry, Float amp) = 0;

    /** change the @p x th value in imap to @p value. */
    virtual bool setIMapValue(Index x, Index value) = 0;

    /** Calculates the distance of each cell to the
        previously uploaded vector.
        Result can be requested via downloadDMap(),
        best match can be found via getMinDMap(). */
    virtual bool calcDMap() = 0;

    /** Returns index to smallest dmap value in @p index.
        If @p only_vacant is true, only empty cells are
        selected. If there is none, -1 is returned in @p index. */
    virtual bool getMinDMap(Index& index, bool only_vacant = false) = 0;

    // ---- debug ----

    virtual bool debugFunc() = 0;
};

} // namespace RSOM

#endif // BACKEND_H
