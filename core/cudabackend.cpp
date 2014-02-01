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

#include "log.h"

#include <cuda_runtime.h>

namespace RSOM {


CudaBackend::CudaBackend()
    :   size(0),
      sizex(0),
      sizey(0),
      dim(0),
      map(0),
      dev_map(0),
      p_upload_(0),
      p_download_(0)
{
}

CudaBackend::~CudaBackend()
{
    free();
}

bool CudaBackend::free()
{
    map = 0;
    sizex =
    sizey =
    size =
    dim = 0;

    if (dev_map)
    {
        cudaError_t err = cudaFree(dev_map);
        if (err != cudaSuccess)
        {
            SOM_ERROR("cudaFree() failed on dev_map " << dev_map);
            return false;
        }
        dev_map = 0;
    }

    if (p_upload_) { delete p_upload_; p_upload_ = 0; }
    if (p_download_) { delete p_download_; p_download_ = 0; }

    return true;
}

bool CudaBackend::setMemory(Float * map, Index sizex, Index sizey, Index dim)
{
    SOM_DEBUG("CudaBackend::setMemory("<<map<<", "<<sizex<<", "<<sizey<<", "<<dim<<")");

    if (!map)
    {
        free();
        return false;
    }

    // re-initialize
    if (this->map || dev_map)
        free();

    // copy data reference
    this->map   = map;
    this->sizex = sizex;
    this->sizex = sizex;
    this->size  = sizex*sizey;
    this->dim   = dim;

    // --- get device memory ----

    // map
    cudaExtent ext = make_cudaExtent(sizex, sizey, dim);
    cudaPitchedPtr p;

    cudaError_t err = cudaMalloc3D(&p, ext);
    if (err != cudaSuccess)
    {
        SOM_ERROR("cudaMalloc3D() failed.");
        return false;
    }

    SOM_DEBUG("CudaBackend::setMemory:: cudaMalloc3d: pitch="
              <<p.pitch<<" ptr="<<p.ptr);

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

    return true;
}


bool CudaBackend::downloadMap()
{
    cudaError_t err = cudaMemcpy3D(p_download_);
    if (err != cudaSuccess)
    {
        SOM_ERROR("cudaMemcpy3D() download failed.");
        return false;
    }
    return true;
}

bool CudaBackend::uploadMap()
{
    cudaError_t err = cudaMemcpy3D(p_upload_);
    if (err != cudaSuccess)
    {
        SOM_ERROR("cudaMemcpy3D() upload failed.");
        return false;
    }
    return true;
}

} // namespace RSOM
