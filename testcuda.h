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

#ifndef TESTCUDA_H
#define TESTCUDA_H

#include "core/cudabackend.h"

int testCuda()
{
    using namespace RSOM;

    CudaBackend cuda;

    const Index
        w = 512,
        h = 512,
        dim = 64;

    std::vector<Float> map(w*h*dim);

    cuda.setMemory(&map[0], w,h,dim);
    cuda.uploadMap();
    cuda.downloadMap();

    return 0;
}


#endif // TESTCUDA_H
