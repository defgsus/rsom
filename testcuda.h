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

#include <vector>
#include "core/cudabackend.h"

/** make an artificial object from index parameter */
void makeVector(std::vector<RSOM::Float>& vec, RSOM::Index index)
{
    for (size_t i=0; i<vec.size(); ++i)
    {
        RSOM::Float t = (RSOM::Float)i/vec.size();
        vec[i] = 0.5 + 0.5 * cos(t * 6.28 * (1.0+0.1*index));
    }
}

int testCuda()
{
    using namespace RSOM;

    CudaBackend cuda;

    const Index
        w = 512,
        h = 512,
        dim = 64;

    std::vector<Float>
            map(w*h*dim),
            dmap(w*h),
            vec(dim);

    cuda.setMemory(w,h,dim);
    cuda.uploadMap(&map[0]);
    cuda.downloadMap(&map[0]);

    makeVector(vec, 1);
    cuda.uploadVec(&vec[0]);
    cuda.set(10,10, 10,10, 1.0);

    cuda.calcDMap();
    cuda.downloadDMap(&dmap[0]);

    return 0;
}


#endif // TESTCUDA_H
