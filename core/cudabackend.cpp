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

#include "cudabackend.h"

#include <sstream>

#include "log.h"
#include "cuda_util.h"



namespace RSOM {

// forwards of kernel calls
bool cudaSom_set(Float * map, Float * vec, Index mapw, Index maph, Index dim,
                 Index bw, Index bh, Index xpos, Index ypos, Float amp,
                 Index threads_sqrt);
bool cudaSom_compare(Float * map, Index w, Index h, Index d, Float * dmap, Float * vec,
                     Index threads);
bool cudaSom_getMin(Float * map, Index size, Index& output,
                    Index * idxmap, Index threads, Index stride);
bool cudaSom_mult(Float * dst, Float * src1, Float * src2, Index size, Index threads);


CudaBackend::CudaBackend(Index max_threads)
    : Backend(),
      size(0),
      sizex(0),
      sizey(0),
      dim(0),
      dev_map(0),
      dev_dmap(0),
      dev_vec(0),
      dev_idx(0),
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

    if (dev_vec)
    {
        CHECK_CUDA( cudaFree(dev_vec), res = false; );
        dev_vec = 0;
    }

    if (dev_idx)
    {
        CHECK_CUDA( cudaFree(dev_idx), res = false; );
        dev_idx = 0;
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

    // som map
    CHECK_CUDA( cudaMalloc((void**)&dev_map,  size * dim * sizeof(Float)), return false );

    threads_idx = std::min(max_threads, size);
    stride_idx = size / threads_idx;
    CHECK_CUDA( cudaMalloc((void**)&dev_idx,  threads_idx * sizeof(Index)), return false );

    //thrust_alloc(&thrust_interface, sizex, sizey, dim);

//    DEBUG_CUDA( size << " " << dev_vec << " " << dev_dmap << " " << dev_map );

    /*cudaExtent ext = make_cudaExtent(sizex, sizey, dim);
    cudaPitchedPtr p;

    CHECK_CUDA( cudaMalloc3D(&p, ext), return false );

    SOM_DEBUG("CudaBackend::setMemory:: cudaMalloc3d: pitch="
              <<p.pitch<<" ptr="<<p.ptr);

    dev_map = p.ptr;


    // setup memcpy parameters
    p_upload_ = new cudaMemcpy3DParms;
    p_download_ = new cudaMemcpy3DParms;

    p_upload_->srcArray = p_upload_->dstArray = 0;
    p_upload_->srcPos = p_upload_->dstPos = make_cudaPos(0,0,0);
    p_upload_->extent = ext;
    *p_download_ = *p_upload_;

    p_upload_->kind = cudaMemcpyHostToDevice;
    p_download_->kind = cudaMemcpyDeviceToHost;
    p_upload_->dstPtr = p_download_->srcPtr = p;
    p_upload_->srcPtr = p_download_->dstPtr
            = make_cudaPitchedPtr(&map[0], sizex, sizex, sizey);
    */

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

bool CudaBackend::uploadVec(const Float * vec)
{
    CHECK_CUDA( cudaMemcpy(dev_vec, vec, dim * sizeof(Float), cudaMemcpyHostToDevice), return false );
    CHECK_CUDA( cudaThreadSynchronize(), return false );
    return true;
}

bool CudaBackend::downloadMap(Float * map)
{
    CHECK_CUDA( cudaThreadSynchronize(), return false );
    CHECK_CUDA( cudaMemcpy(map, dev_map, size * dim * sizeof(Float), cudaMemcpyDeviceToHost), return false );
    return true;
}

bool CudaBackend::downloadDMap(Float * dmap)
{
    CHECK_CUDA( cudaThreadSynchronize(), return false );
    CHECK_CUDA( cudaMemcpy(dmap, dev_dmap, size * sizeof(Float), cudaMemcpyDeviceToHost), return false );
    return true;
}

bool CudaBackend::set(Index x, Index y, Index rx, Index ry, Float amp)
{
    if (! cudaSom_set(dev_map, dev_vec, sizex, sizey, dim,
                       rx, ry, x, y, amp, 32)
        ) return false;

    CHECK_CUDA( cudaThreadSynchronize(), return false );
    return true;
}

bool CudaBackend::calcDMap()
{
    if (! cudaSom_compare(dev_map, sizex, sizey, dim, dev_dmap, dev_vec, max_threads)
        ) return false;
    //return thrust_dmap(thrust_interface);

    CHECK_CUDA( cudaThreadSynchronize(), return false );
    return true;
}

bool CudaBackend::getMinDMap(Index& index)
{
    if (! cudaSom_getMin(dev_dmap, size, index,
                          dev_idx, threads_idx, stride_idx)
        ) return false;

    CHECK_CUDA( cudaThreadSynchronize(), return false );
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
