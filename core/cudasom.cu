/*  This is free software; you can redistribute it and/or
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
*/
/** @file
    @brief low-level cuda operations for RSOM::CudaBackend

    @version 2014/02/01 started

    copyright 2014 stefan.berke @ modular-audio-graphics.com
*/


#include <cuda.h>

#include "cuda_util.h"
#include "som_types.h"

namespace RSOM
{

// ------------------------ SET --------------------------------

__global__ void kernel_set(Float * map, Float * vec, Index w, Index h, Index dim,
                           Index bw, Index bh,// Index bxo, Index byo,
                           Index xo, Index yo,
                           Float b_amp)
{
    const Index
            x = blockDim.x * blockIdx.x + threadIdx.x + xo,
            y = blockDim.y * blockIdx.y + threadIdx.y + yo;

    if (x<w && y<h)
    {
        // calc radius dependent amplitude
        Float   dx = (Float)(x - xo) / bw * 2.f,
                dy = (Float)(y - yo) / bh * 2.f,
                d = sqrtf(dx*dx+dy*dy);
        Float amp = b_amp * max(0.f, 1.f - d);

        Float * p = &map[(y*w+x)*dim];
        for (Index i=0; i<dim; ++i, ++p)
            *p += amp * (*vec - *p);
    }
}

bool cudaSom_set(Float * map, Float * vec, Index mapw, Index maph, Index dim,
                 Index bw, Index bh, Index xpos, Index ypos, Float amp)
{
    Index
    // actual brush size
        bxs = bw,
        bys = bh,
    // brush corner position
        bx = xpos - bw/2,
        by = ypos - bw/2;
    // brush offset (for edge clamping)
        //bxo = 0,
        //byo = 0

    // out of map?
    if (bx < 0) { bxs += bx; /*bxo = -bx;*/ bx = 0; }
    if (by < 0) { bys += by; /*byo = -by;*/ by = 0; }
    // limit width/height
    if (bx+bxs >= mapw) bxs = mapw - bxs - 1;
    if (by+bys >= maph) bys = maph - bys - 1;

    // set blocks/threads
    const dim3
            threads(31, 31),
            blocks((bxs+threads.x-1)/threads.x, (bys+threads.y-1)/threads.y);

    CHECK_CUDA_KERNEL(( kernel_set<<<blocks, threads>>>(map, vec, mapw, maph, dim,
                                                        bxs,bys,bx,by,amp) ), return false );

    return true;
}



// ---------------------  COMPARE  -----------------------------

__global__ void kernel_compare(Float * map, Float * dmap, Float * vec, Index size, Index dim)
{
    // cell for this thread
    const Index i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i<size)
    {
        Float * cell = &map[i * dim];

        // step through dimensions of cell
        Float d = 0;
        for (Index j=0; j<dim; ++j)
        {
            d += fabsf(vec[j] - cell[j]);
        }

        // store result
        dmap[0] = d / dim;
    }
}

bool cudaSom_compare(Float * map, Index w, Index h, Index dim, Float * dmap, Float * vec)
{
    int threads = 512;
    int blocks = (w*h+threads-1)/threads;

    CHECK_CUDA_KERNEL(( kernel_compare<<<blocks, threads>>>(map, dmap, vec, w*h, dim) ),
                      DEBUG_CUDA("blocks="<<blocks<<", threads="<<threads); return false );

    return true;
}


// ----------------------- GET MAX -------------------------------

bool cudaSom_getMax(Float * map, Index size, Index& output)
{
    return true;
}

} // namespace RSOM
