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

/** @file
    XXX this is completely experimental pre-beta
*/
#include "cublasbackend.h"

#include <sstream>

#include "cuda_util.h"

// XXX log.h requires c++0x
//#include "log.h"
#ifndef NDEBUG
#   define SOM_DEBUG(stream_arg__) \
        std::cerr << stream_arg__ << "\n";
#else
#   define SOM_DEBUG(unused__) { }
#endif

/** Macro for checking for cuda errors.
    Define CHECK_CUDA before including this header to change behaviour */
#define CHECK_CUBLAS( command__, code_on_error__ ) \
{ \
    SOM_DEBUG( ":" << #command__ ); \
    cublasStatus_t err = command__; \
    if (err != CUBLAS_STATUS_SUCCESS) \
    { \
        std::cerr << "Cublas Error: " << err \
                          << "\nfor command '" #command__ "'\n"; \
        code_on_error__; \
    } \
}

namespace RSOM {

cublasHandle_t CublasBackend::handle = 0;

CublasBackend::CublasBackend(Index max_threads)
    : Backend(),
      size(0),
      sizex(0),
      sizey(0),
      dim(0),
      dev_map(0),
      dev_dmap(0),
      dev_vec(0),
      dev_idx(0)

{
    if (!handle)
        CHECK_CUBLAS( cublasCreate_v2(&handle), );
}

CublasBackend::~CublasBackend()
{
    free();
}

std::string CublasBackend::name() const
{
    return "cublas";/*
    std::stringstream s;
    s << "cuda" << max_threads;
    return s.str();*/
}

bool CublasBackend::free()
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

bool CublasBackend::setMemory(Index sizex_, Index sizey_, Index dim_)
{
    SOM_DEBUG("CublasBackend::setMemory("<<sizex_<<", "<<sizey_<<", "<<dim_<<")");

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

//    CHECK_CUDA( cudaMalloc((void**)&dev_idx,  threads_idx * sizeof(Index)), return false );

    //thrust_alloc(&thrust_interface, sizex, sizey, dim);

//    DEBUG_CUDA( size << " " << dev_vec << " " << dev_dmap << " " << dev_map );

    /*cudaExtent ext = make_cudaExtent(sizex, sizey, dim);
    cudaPitchedPtr p;

    CHECK_CUDA( cudaMalloc3D(&p, ext), return false );

    SOM_DEBUG("CublasBackend::setMemory:: cudaMalloc3d: pitch="
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
    return true;
}


bool CublasBackend::uploadMap(const Float * map)
{
    CHECK_CUDA( cudaMemcpy(dev_map, map, size * dim * sizeof(Float), cudaMemcpyHostToDevice), return false );
//    CHECK_CUDA( cudaMemcpy3D(p_upload_), return false );
    return true;
}

bool CublasBackend::uploadVec(const Float * vec)
{
    CHECK_CUDA( cudaMemcpy(dev_vec, vec, dim * sizeof(Float), cudaMemcpyHostToDevice), return false );
    return true;
}

bool CublasBackend::downloadMap(Float * map)
{
    CHECK_CUDA( cudaMemcpy(map, dev_map, size * dim * sizeof(Float), cudaMemcpyDeviceToHost), return false );
    return true;
}

bool CublasBackend::downloadDMap(Float * dmap)
{
    CHECK_CUDA( cudaMemcpy(dmap, dev_dmap, size * sizeof(Float), cudaMemcpyDeviceToHost), return false );
    return true;
}

bool CublasBackend::set(Index x, Index y, Index rx, Index ry, Float amp)
{
//    return cudaSom_set(dev_map, dev_vec, sizex, sizey, dim,
//                       rx, ry, x, y, amp, 32);
}

bool CublasBackend::calcDMap()
{
//    for (Index i=0; i<size; ++i)
//        CHECK_CUBLAS( cublasSdot_v2(handle, dim, dev_vec, 1, &dev_map[i*dim], 1, &dev_dmap[i]), );
      Float r;
        CHECK_CUBLAS( cublasSdot(handle, dim, dev_vec, 1, dev_map, 1, &r), );
//    Float alpha = 1.f;
//    CHECK_CUBLAS( cublasSaxpy(handle, dim, &alpha, dev_map, 1, dev_vec, 1), );

    int midx;
    CHECK_CUBLAS( cublasIsamin_v2(handle, size, dev_dmap, 1, &midx), );

//    return cudaSom_compare(dev_map, sizex, sizey, dim, dev_dmap, dev_vec, max_threads);
    return true;
}

bool CublasBackend::getMinDMap(Index& index)
{
//    return cudaSom_getMin(dev_dmap, size, index,
//                          dev_idx, threads_idx, stride_idx);
}

bool CublasBackend::debugFunc()
{
//    cublasSscal_v2(
//    return cudaSom_mult(dev_dmap, dev_map, dev_map, size, max_threads);
}

} // namespace RSOM
