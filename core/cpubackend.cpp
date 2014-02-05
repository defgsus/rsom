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

#include <cmath>

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

    { std::vector<Float> tmp; tmp.swap(cpu_map); }
    { std::vector<Float> tmp; tmp.swap(cpu_dmap); }
    { std::vector<Float> tmp; tmp.swap(cpu_vec); }
    { std::vector<Index> tmp; tmp.swap(cpu_imap); }

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

    try
    {
        cpu_vec.resize(dim);
        cpu_dmap.resize(size);
        cpu_imap.resize(size);
        cpu_map.resize(size*dim);
    }
    catch (std::exception& e)
    {
        //std::cerr << "exception: " << e.what() << "\n";
        return false;
    }

    return true;
}


bool CpuBackend::uploadMap(const Float * map)
{
    for (auto i=cpu_map.begin(); i!=cpu_map.end(); ++i, ++map)
        *i = *map;
    return true;
}

bool CpuBackend::uploadIMap(const Index * imap)
{
    for (auto i=cpu_imap.begin(); i!=cpu_imap.end(); ++i, ++imap)
        *i = *imap;
    return true;
}

bool CpuBackend::uploadVec(const Float * vec)
{
    for (auto i=cpu_vec.begin(); i!=cpu_vec.end(); ++i, ++vec)
        *i = *vec;
    return true;
}

bool CpuBackend::downloadMap(Float * map, Index z, Index depth)
{
    if (depth == 0) depth = dim;
    depth = std::min(dim - z, depth);
    if (depth < 0) return false;

    Float * dst = map,
          * src = &cpu_map[z*size];
    for (int i=0; i<size*depth; ++i)
        *dst++ = *src++;

    return true;
}

bool CpuBackend::downloadIMap(Index * imap)
{
    for (auto i=cpu_imap.begin(); i!=cpu_imap.end(); ++i, ++imap)
        *imap = *i;
    return true;
}

bool CpuBackend::downloadDMap(Float * dmap)
{
    for (auto i=cpu_dmap.begin(); i!=cpu_dmap.end(); ++i, ++dmap)
        *dmap = *i;
    return true;
}

bool CpuBackend::setIMapValue(Index x, Index value)
{
    if (x<0 || x>=size) return false;
    cpu_imap[x] = value;
    return true;
}

bool CpuBackend::set(Index x, Index y, Index rx, Index ry, Float amp)
{
    //std::cout << x << ", " << y << ", " << rx << ", " << ry << "\n";

    // for each line
    for (Index j=y-ry; j<=y+ry; ++j)
    if (j>=0 && j<sizey)
    {
        Float dy = (Float)(j-y) / ry;

        // for each column
        for (Index i=x-rx; i<=x+rx; ++i)
        if (i>=0 && i<sizex)
        {
            Float   dx = (Float)(i-x) / rx,
                    d = sqrtf(dx*dx + dy*dy),
            // amplitude from radius
                    a = amp * ((Float)1 - d);
            // skip outside radius
            if (a<=0) continue;

            // adjust whole vector at this cell
            Float * p = &cpu_map[j*sizex+i];
            for (Index k=0; k<dim; ++k, p += size)
                *p += a * (cpu_vec[k] - *p);
        }
    }
    return true;
}

bool CpuBackend::calcDMap(bool only_vacant, Float fixed_value)
{
    return calcDMap(0,0,sizex,sizey, only_vacant, fixed_value);
}

bool CpuBackend::calcDMap(Index x, Index y, Index w, Index h,
                          bool only_vacant, Float fixed_value)
{
    const Index
            x0 = std::max(0,std::min(sizex-1, x )),
            x1 = std::max(0,std::min(sizex-1, x + w )),
            y0 = std::max(0,std::min(sizey-1, y )),
            y1 = std::max(0,std::min(sizey-1, y + h ));

    // for each cell
    for (Index j=y0; j<y1; ++j)
    for (Index i=x0; i<x1; ++i)
    {
        const Index idx = j * sizex + i;

        Float * p = &cpu_map[idx];

        if (only_vacant && cpu_imap[idx]>=0)
        {
            cpu_dmap[idx] = fixed_value;
        }
        else
        {
            // get difference between cpu_vec and cell
            Float d = 0;
            for (Index k=0; k<dim; ++k, p += size)
                d += fabsf(cpu_vec[k] - *p);

            cpu_dmap[i] = d / dim;
        }
    }
    return true;
}

bool CpuBackend::getMinDMap(Index& index, Float& value, Index count)
{
    count = count? std::min(size, count) : size;

    index = -1;
    value = 0;
    for (Index i=0; i<count; ++i)
    {
        Float d = cpu_dmap[i];
        if (index<0 || d < value)
        {
            index = i;
            value = d;
        }
    }
    return true;
}

bool CpuBackend::debugFunc()
{
    Float   * dst = &cpu_dmap[0],
            * src1 = &cpu_map[0],
            * src2 = &cpu_map[0];
    for (Index i=0; i<size; ++i)
        *dst ++ = *src1++ * *src2++;
    return true;
}

} // namespace RSOM
