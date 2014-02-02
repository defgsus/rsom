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

#include "cpubackend.h"

#include "log.h"


namespace RSOM {

CpuBackend::CpuBackend()
    : Backend(),
      size(0),
      sizex(0),
      sizey(0),
      dim(0)
{
}

CpuBackend::~CpuBackend()
{
    free();
}

bool CpuBackend::free()
{
    sizex =
    sizey =
    size =
    dim = 0;

    cpu_map.resize(0);
    cpu_dmap.resize(0);
    cpu_vec.resize(0);

    return true;
}

bool CpuBackend::setMemory(Index sizex_, Index sizey_, Index dim_)
{
    SOM_DEBUG("CpuBackend::setMemory("<<sizex_<<", "<<sizey_<<", "<<dim_<<")");

    // re-initialize
    free();

    // copy data size
    sizex = sizex_;
    sizey = sizey_;
    size  = sizex_ * sizey_;
    dim   = dim_;

    // --- get memory ----

    cpu_vec.resize(dim);
    cpu_dmap.resize(size);
    cpu_map.resize(size*dim);

    return true;
}


bool CpuBackend::uploadMap(const Float * map)
{
    for (auto i=cpu_map.begin(); i!=cpu_map.end(); ++i, ++map)
        *i = *map;
    return true;
}

bool CpuBackend::uploadVec(Float * vec)
{
    for (auto i=cpu_vec.begin(); i!=cpu_vec.end(); ++i, ++vec)
        *i = *vec;
    return true;
}

bool CpuBackend::downloadMap(Float * map)
{
    for (auto i=cpu_map.begin(); i!=cpu_map.end(); ++i, ++map)
        *map = *i;
    return true;
}

bool CpuBackend::downloadDMap(Float * dmap)
{
    for (auto i=cpu_dmap.begin(); i!=cpu_dmap.end(); ++i, ++dmap)
        *dmap = *i;
    return true;
}

bool CpuBackend::set(Index x, Index y, Index rx, Index ry, Float amp)
{
    return true;
}

bool CpuBackend::calcDMap()
{
    return true;
}

bool CpuBackend::getMinDMap(Index& index)
{
    return true;
}

} // namespace RSOM
