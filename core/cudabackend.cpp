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

#ifndef RSOM_NO_CUDA

#include "cudabackend.h"

#include <sstream>

#ifndef RSOM_NO_CUDA

    #include "log.h"
    #include "cuda_util.h"
#endif


namespace RSOM {


// forwards of kernel calls
bool cudaSom_set(Float * map, Float * vec, Index mapw, Index maph, Index dim,
                 Index bw, Index bh, Index xpos, Index ypos, Float amp,
                 Index threads_sqrt);
bool cudaSom_compare(Float * map, Index w, Index h, Index d, Float * dmap, Float * vec,
                     Index threads, bool only_vacant=false, Float fixed_value=0, Index * imap=0);
bool cudaSom_compare(Float * map, Index w, Index h, Index d, Float * dmap, Float * vec,
                     Index wx, Index wy, Index ww, Index wh,
                     Index threads, bool only_vacant=false, Float fixed_value=0, Index * imap=0);
bool cudaSom_getMin(Float * dmap, Index size, Index& output, Float& distance);
bool cudaSom_getMin(Float * dmap, Index sizex, Index x, Index y, Index w, Index h, Index& output, Float& distance);
bool cudaSom_calcUMap(Float * map, Index w, Index h, Index dim, Float * umap, Index threads);
//bool cudaSom_getMinVacant(Float * dmap, Index * imap, Index size, Index& output, Index * scratch);
bool cudaSom_mult(Float * dst, Float * src1, Float * src2, Index size, Index threads);
bool cudaSom_setImap(Index * imap, Index x, Index value);
bool cudaSom_normalize(Float * data, Index size, Float value);

CudaBackend::CudaBackend(Index max_threads)
    : Backend(),
      size(0),
      sizex(0),
      sizey(0),
      dim(0),
      dev_map(0),
      dev_dmap(0),
      dev_umap(0),
      dev_vec(0),
      dev_scratch(0),
      max_threads(max_threads)
{
}

CudaBackend::~CudaBackend()
{
    free();
}

std::string CudaBackend::name() const
{
    std::stringstream s;
    s << "cuda" << max_threads;
    return s.str();
}

bool CudaBackend::free()
{
    sizex =
    sizey =
    size =
    dim = 0;

    bool res = true;

    if (dev_map)
    {
        CHECK_CUDA( cudaFree(dev_map), res = false; );
        dev_map = 0;
    }

    if (dev_dmap)
    {
        CHECK_CUDA( cudaFree(dev_dmap), res = false; );
        dev_dmap = 0;
    }

    if (dev_dmap)
    {
        CHECK_CUDA( cudaFree(dev_umap), res = false; );
        dev_umap = 0;
    }

    if (dev_imap)
    {
        CHECK_CUDA( cudaFree(dev_imap), res = false; );
        dev_imap = 0;
    }

    if (dev_vec)
    {
        CHECK_CUDA( cudaFree(dev_vec), res = false; );
        dev_vec = 0;
    }

    if (dev_scratch)
    {
        CHECK_CUDA( cudaFree(dev_scratch), res = false; );
        dev_scratch = 0;
    }

    return res;
}

bool CudaBackend::setMemory(Index sizex_, Index sizey_, Index dim_)
{
    SOM_DEBUG("CudaBackend::setMemory("<<sizex_<<", "<<sizey_<<", "<<dim_<<")");

    // re-initialize
    free();

    // copy data size
    sizex = sizex_;
    sizey = sizey_;
    size  = sizex_ * sizey_;
    dim   = dim_;

    // --- get device memory ----

    // question vector
    CHECK_CUDA( cudaMalloc((void**)&dev_vec,  dim * sizeof(Float)), return false );

    // difference map
    CHECK_CUDA( cudaMalloc((void**)&dev_dmap, size * sizeof(Float)), return false );

    // neighbour difference map
    CHECK_CUDA( cudaMalloc((void**)&dev_umap, size * sizeof(Float)), return false );
    CHECK_CUDA( cudaMemset((void*)dev_umap, 0, size * sizeof(Float)), return false );

    // index map
    CHECK_CUDA( cudaMalloc((void**)&dev_imap, size * sizeof(Index)), return false );

    // som map
    CHECK_CUDA( cudaMalloc((void**)&dev_map,  size * dim * sizeof(Float)), return false );

    // scratch space
    CHECK_CUDA( cudaMalloc((void**)&dev_scratch, sizeof(Index)), return false );


    CHECK_CUDA( cudaThreadSynchronize(), return false );
    return true;
}


bool CudaBackend::uploadMap(const Float * map)
{
    CHECK_CUDA( cudaMemcpy(dev_map, map, size * dim * sizeof(Float), cudaMemcpyHostToDevice), return false );
//    CHECK_CUDA( cudaMemcpy3D(p_upload_), return false );
    CHECK_CUDA( cudaThreadSynchronize(), return false );
    return true;
}

bool CudaBackend::uploadIMap(const Index * map)
{
    CHECK_CUDA( cudaMemcpy(dev_imap, map, size * sizeof(Index), cudaMemcpyHostToDevice), return false );
    CHECK_CUDA( cudaThreadSynchronize(), return false );
    return true;
}

bool CudaBackend::uploadVec(const Float * vec)
{
    CHECK_CUDA( cudaMemcpy(dev_vec, vec, dim * sizeof(Float), cudaMemcpyHostToDevice), return false );
    CHECK_CUDA( cudaThreadSynchronize(), return false );
    return true;
}

bool CudaBackend::downloadMap(Float * map, Index z, Index depth)
{
    CHECK_CUDA( cudaThreadSynchronize(), return false );

    if (depth == 0) depth = dim;
    depth = std::min(dim - z, depth);
    if (depth < 0) return false;

    CHECK_CUDA( cudaMemcpy(map, &dev_map[z*size], size * depth * sizeof(Float),
            cudaMemcpyDeviceToHost), return false );

    return true;
}

bool CudaBackend::downloadIMap(Index * map)
{
    CHECK_CUDA( cudaThreadSynchronize(), return false );
    CHECK_CUDA( cudaMemcpy(map, dev_imap, size * sizeof(Index), cudaMemcpyDeviceToHost), return false );
    return true;
}

bool CudaBackend::downloadDMap(Float * dmap)
{
    CHECK_CUDA( cudaThreadSynchronize(), return false );
    CHECK_CUDA( cudaMemcpy(dmap, dev_dmap, size * sizeof(Float), cudaMemcpyDeviceToHost), return false );
    return true;
}

bool CudaBackend::downloadUMap(Float * umap)
{
    CHECK_CUDA( cudaThreadSynchronize(), return false );
    CHECK_CUDA( cudaMemcpy(umap, dev_umap, size * sizeof(Float), cudaMemcpyDeviceToHost), return false );
    return true;
}

bool CudaBackend::setIMapValue(Index x, Index value)
{
    if (x<0 || x>=size) return false;
    return cudaSom_setImap(dev_imap, x, value);
}

bool CudaBackend::set(Index x, Index y, Index rx, Index ry, Float amp)
{
    if (! cudaSom_set(dev_map, dev_vec, sizex, sizey, dim,
                       rx, ry, x, y, amp, 32)
        ) return false;

    CHECK_CUDA( cudaThreadSynchronize(), return false );
    return true;
}

bool CudaBackend::calcUMap()
{
    if (! cudaSom_calcUMap(dev_map, sizex, sizey, dim, dev_umap, max_threads)
        ) return false;

    CHECK_CUDA( cudaThreadSynchronize(), return false );
    return true;
}

bool CudaBackend::calcDMap(bool only_vacant, Float fixed_value)
{
    if (! cudaSom_compare(
                dev_map, sizex, sizey, dim, dev_dmap, dev_vec, max_threads,
                only_vacant, fixed_value, dev_imap)
        ) return false;

    CHECK_CUDA( cudaThreadSynchronize(), return false );
    return true;
}

bool CudaBackend::calcDMap(Index x, Index y, Index w, Index h,
                           bool only_vacant, Float fixed_value)
{
    if (! cudaSom_compare(
                dev_map, sizex, sizey, dim, dev_dmap, dev_vec,
                x, y, w, h,
                max_threads, only_vacant, fixed_value, dev_imap)
        ) return false;

    CHECK_CUDA( cudaThreadSynchronize(), return false );
    return true;
}

bool CudaBackend::getMinDMap(Index& index, Float& distance)
{
    if (! cudaSom_getMin(dev_dmap, size, index, distance)
        ) return false;

    CHECK_CUDA( cudaThreadSynchronize(), return false );
    return true;
}

bool CudaBackend::getMinDMap(Index& index, Float& distance,
                             Index x, Index y, Index w, Index h)
{
    if (! cudaSom_getMin(dev_dmap, sizex, x, y, w, h, index, distance)
        ) return false;

    CHECK_CUDA( cudaThreadSynchronize(), return false );
    return true;
}

bool CudaBackend::normalizeUMap(Float factor)
{
    return cudaSom_normalize(dev_umap, size, factor);
}

bool CudaBackend::normalize(Float * map, Index size, Float factor)
{
    std::cerr << "not implemented\n";
    exit(-1);
    return true;
}

bool CudaBackend::debugFunc()
{
    if (! cudaSom_mult(dev_dmap, dev_map, dev_map, size, max_threads)
        ) return false;

    CHECK_CUDA( cudaThreadSynchronize(), return false );
    return true;

}


} // namespace RSOM


#endif //#ifndef RSOM_NO_CUDA
