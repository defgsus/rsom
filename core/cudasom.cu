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
//#include <thrust/device_vector.h>
#include <cublas.h>

#include "cuda_util.h"
#include "cublas_util.h"
#include "som_types.h"

namespace RSOM
{

cublasHandle_t cublas_handle()
{
    static cublasHandle_t handle_ = 0;

    if (!handle_)
        CHECK_CUBLAS( cublasCreate_v2(&handle_), );
    return handle_;
}

// ------------------------ SET IMAP --------------------------------

__global__ void kernel_set_imap(Index * imap, Index x, Index value)
{
    imap[x] = value;
}

bool cudaSom_setImap(Index * imap, Index x, Index value)
{
    CHECK_CUDA_KERNEL((
        kernel_set_imap<<<1,1>>>(imap, x, value)
        ), return false );
    return true;
}

// ------------------------ SET MAP --------------------------------

__global__ void kernel_set(Float * map, Float * vec, Index w, Index h, Index dim,
                           Index bw, Index bh,
                           Index bworg, Index bhorg, Index bxo, Index byo,
                           Index xo, Index yo,
                           Float b_amp)
{
    const Index
            x = blockDim.x * blockIdx.x + threadIdx.x,
            y = blockDim.y * blockIdx.y + threadIdx.y,
            size = w*h;

    if (x<bw && y<bh)
    {
        // calc radius dependent amplitude
        Float   dx = (Float)(x - bworg/2 + bxo) / bworg * 2,
                dy = (Float)(y - bhorg/2 + byo) / bhorg * 2,
                d = sqrtf(dx*dx+dy*dy);
        Float amp = b_amp * max(0.f, 1.f - d);

        Float * p = &map[((y+yo)*w+x+xo)];
        for (Index i=0; i<dim; ++i, p += size, ++vec)
            *p += amp * (*vec - *p);
    }
}

/** Inserts sample into @p map.
 *  @p map is defined by [@p maph][@p mapw][@p dim].
 *  Radius of adjustment is @p brx, @p bry.
 *  Sample is inserted at @p xpos, @p ypos with amplitude @p amp.
 *  @p threads_sqrt is the square root of maximum number of threads to use.
 **/
bool cudaSom_set(Float * map, Float * vec, Index mapw, Index maph, Index dim,
                 Index brx, Index bry, Index xpos, Index ypos, Float amp,
                 Index threads_sqrt)
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

//    std::cout << "bx="<<bx<<" by="<<by<<"  bxs="<<bxs<<" bys="<<bys<<"  bxo="<<bxo<<" byo="<<byo<<"\n";

    // out of map?
    if (bx < 0) { bxs += bx; bxo = -bx; bx = 0; }
    if (by < 0) { bys += by; byo = -by; by = 0; }
    // limit width/height
    bxs = std::min(bxs, mapw - bx);
    bys = std::min(bys, mapw - by);

//    std::cout << "bx="<<bx<<" by="<<by<<"  bxs="<<bxs<<" bys="<<bys<<"  bxo="<<bxo<<" byo="<<byo<<"\n";

    // set blocks/threads
    const dim3
            threads(bxs < threads_sqrt? nextPow2(bxs) : threads_sqrt,
                    bys < threads_sqrt? nextPow2(bys) : threads_sqrt),
            blocks((bxs+threads.x-1)/threads.x, (bys+threads.y-1)/threads.y);

    CHECK_CUDA_KERNEL((
            kernel_set<<<blocks, threads>>>(map, vec, mapw, maph, dim,
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
        // step through dimensions of cell
        Float d = 0;
        for (Index j=0; j<dim; ++j)
        {
            d += fabsf(vec[j] - map[j * size + i]);
        }

        // store result
        dmap[i] = d / dim;
    }
}

/** only for testing. working depends on number of threads!! */
__global__ void kernel_compare_shared(Float * map, Float * dmap, Float * vec, Index size, Index dim)
{
    // cell for this thread
    const Index i = blockDim.x * blockIdx.x + threadIdx.x;


    __shared__ Float svec[1024];
    if (i<dim)
        svec[i] = vec[i];
    __syncthreads();

    if (i<size)
    {
        // step through dimensions of cell
        Float d = 0;
        for (Index j=0; j<dim; ++j)
        {
            d += fabsf(svec[j] - map[j * size + i]);
        }

        // store result
        dmap[i] = d / dim;
    }
}

/** compare each cell in map with vector @p vec.
    Store difference of each cell in @p dmap. */
bool cudaSom_compare(Float * map, Index w, Index h, Index dim, Float * dmap, Float * vec,
                     Index threads)
{
    int blocks = (w*h+threads-1)/threads;

    CHECK_CUDA_KERNEL(( kernel_compare<<<blocks, threads>>>(map, dmap, vec, w*h, dim) ),
                      DEBUG_CUDA("blocks="<<blocks<<", threads="<<threads); return false );

    return true;
}

#ifdef WINDOW_FUNCTION_TODO
__global__ void kernel_compare_window(Float * map, Float * dmap, Float * vec, Index size, Index dim)
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

/** compare each cell in the map window with vector @p vec.
    Store difference of each cell in @p dmap. */
bool cudaSom_compare_window(Float * map, Index w, Index h, Index dim, Float * dmap, Float * vec,
                     Index threads)
{
    int blocks = (w*h+threads-1)/threads;

    CHECK_CUDA_KERNEL(( kernel_compare<<<blocks, threads>>>(map, dmap, vec, w*h, dim) ),
                      DEBUG_CUDA("blocks="<<blocks<<", threads="<<threads); return false );

    return true;
}
#endif

// ----------------------- GET MIN -------------------------------

/** Returns in @p output, the index to the smallest value in @p dmap. */
bool cudaSom_getMin(Float * dmap, Index size, Index& output)
{
    int midx;

    CHECK_CUBLAS( cublasIsamin_v2(cublas_handle(), size, dmap, 1, &midx), );
    // cublas is 1-based!!
    output = midx - 1;

    return true;
}

#ifdef OLD_EXPERIMENT_NOT_NEEDED_RIGHT_NOW
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

/** determine least index in minindex */
__global__ void kernel_get_min_index(Index * minindex, Float * dmap, Index size)
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
#endif

/** Determines index to least value in dmap, stores in minindex.
    Ignores cells that are not vacant, according to @p imap. */
__global__ void kernel_get_min(Index * minindex, Float * dmap, Index * imap, Index size)
{
    Index index = -1;
    Float md = 0;
    for (Index i=0; i<size; ++i)
    {
        if (imap[i]>=0)
            continue;

        Float d = dmap[i];
        if (index<0 || d < md)
        {
            index = i;
            md = d;
        }
    }
    *minindex = index;
}

/** Searches for the minimum value in dmap.
    @p size is the size of @p dmap, e.g. width * height.
    @p imap is the index map
    */
bool cudaSom_getMinVacant(Float * dmap, Index * imap, Index size, Index& output, Index * scratch)
{
    CHECK_CUDA_KERNEL((
        kernel_get_min<<<1, 1>>>(scratch, dmap, imap, size)
            ), return false );

    // get the first entry of index map
    CHECK_CUDA( cudaMemcpy(&output, scratch, sizeof(Index), cudaMemcpyDeviceToHost), return false );


    return true;
}



// ------------------------------- MULT ------------------------------

__global__ void kernel_mult(Float * dst, Float * src1, Float * src2, Index size)
{
    // cell for this thread
    const Index i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i<size)
    {
        dst[i] = src1[i] * src2[i];
    }
}

bool cudaSom_mult(Float * dst, Float * src1, Float * src2, Index size, Index threads)
{
    int blocks = (size+threads-1)/threads;

    CHECK_CUDA_KERNEL(( kernel_mult<<<blocks, threads>>>(dst, src1, src2, size) ),
                      DEBUG_CUDA("blocks="<<blocks<<", threads="<<threads); return false );
    return true;
}





} // namespace RSOM
