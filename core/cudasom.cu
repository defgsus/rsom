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
                           Index bw, Index bh,
                           Index bworg, Index bhorg, Index bxo, Index byo,
                           Index xo, Index yo,
                           Float b_amp)
{
    const Index
            x = blockDim.x * blockIdx.x + threadIdx.x,
            y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x<bw && y<bh)
    {
        // calc radius dependent amplitude
        Float   dx = (Float)(x - bworg/2 + bxo) / bworg * 2,
                dy = (Float)(y - bhorg/2 + byo) / bhorg * 2,
                d = sqrtf(dx*dx+dy*dy);
        Float amp = b_amp * max(0.f, 1.f - d);

        Float * p = &map[((y+yo)*w+x+xo)*dim];
        for (Index i=0; i<dim; ++i, ++p)
            *p += amp * (*vec - *p);
    }
}

bool cudaSom_set(Float * map, Float * vec, Index mapw, Index maph, Index dim,
                 Index brx, Index bry, Index xpos, Index ypos, Float amp)
{
    Index
    // actual brush size
        bxs = brx*2+1,
        bys = bry*2+1,
    // brush corner position
        bx = xpos - brx,
        by = ypos - bry,
    // brush offset (for edge clamping)
        bxo = 0,
        byo = 0;

    // out of map?
    if (bx < 0) { bxs += bx; bxo = -bx; bx = 0; }
    if (by < 0) { bys += by; byo = -by; by = 0; }
    // limit width/height
    if (bx+bxs >= mapw) bxs = mapw - bxs - 1;
    if (by+bys >= maph) bys = maph - bys - 1;

    // set blocks/threads
    const dim3
            threads(std::min(32,bxs), std::min(32,bys)),
            blocks((bxs+threads.x-1)/threads.x, (bys+threads.y-1)/threads.y);

    CHECK_CUDA_KERNEL(( kernel_set<<<blocks, threads>>>(map, vec, mapw, maph, dim,
                                                        bxs,bys,
                                                        brx*2+1, bry*2+1, bxo,byo,
                                                        bx,by,amp) ), return false );

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
        dmap[i] = d / dim;
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

__global__ void kernel_reduce_min(Index * minindex, Float *dmap, Index size, Index stride)
{
    const Index tid = threadIdx.x;
    Index i = (blockDim.x * blockIdx.x + tid) * stride;

    for (Index j = 0; j<stride && i<size; ++j, ++i)
    {
        Float d = dmap[i];
        if (d < dmap[minindex[tid]])
            minindex[tid] = i;
    }
}

__global__ void kernel_get_min(Index * minindex, Float * dmap, Index size)
{
    Index m = 0;
    Float d = dmap[minindex[0]];
    for (Index i = 1; i<size; ++i)
    {
        Index midx = minindex[i];
        Float d1 = dmap[midx];
        if (d1 < d)
        {
            m = midx;
            d = d1;
        }
    }
    minindex[0] = m;
}

/** Searches for the minimum value in dmap.
    @p size is the size of @p dmap, e.g. width * height.
    @p idxmap is a scratch area that needs to be allocated to size / threads ints.
    @p stride must be size / threads
    */
bool cudaSom_getMin(Float * dmap, Index size, Index& output,
                    Index * idxmap, Index threads, Index stride)
{
    // clear idxmap
    CHECK_CUDA( cudaMemset(idxmap, 0, threads * sizeof(Index)), return false );

    // reduce
    CHECK_CUDA_KERNEL((
        kernel_reduce_min<<<stride, threads>>>(idxmap, dmap, size, stride)
            ), return false );

    // get min in reduced map
    CHECK_CUDA_KERNEL((
        kernel_get_min<<<1, 1>>>(idxmap, dmap, stride)
            ), return false );

    // get the first entry
    CHECK_CUDA( cudaMemcpy(&output, idxmap, sizeof(Index), cudaMemcpyDeviceToHost), return false );

    return true;
}

} // namespace RSOM