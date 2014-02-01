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

    CHECK_CUDA_KERNEL((kernel_compare<<<blocks, threads>>>(map, dmap, vec, w*h, dim)),
                      DEBUG_CUDA("blocks="<<blocks<<",threads="<<threads); return false );

    return true;
}


} // namespace RSOM
