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
#include <iomanip>

#include "core/cuda_util.h"

#include "core/cudabackend.h"
#include "core/cpubackend.h"
#include "core/som.h"

/** make an artificial object from index parameter */
void makeVector(std::vector<RSOM::Float>& vec, RSOM::Index index)
{
    for (size_t i=0; i<vec.size(); ++i)
    {
        if (index == 0)
            vec[i] = 1.0;
        else
        {
            RSOM::Float t = (RSOM::Float)i/vec.size();
            vec[i] = 0.5 + 0.5 * cos(t * 6.28 * (1.0+0.1*index) + 0.1*i);
        }
    }
}


bool insertSome(RSOM::Backend * cuda, RSOM::Index w, RSOM::Index /*h*/, RSOM::Index dim, RSOM::Index numIt)
{
    using namespace RSOM;

    std::vector<Float> vec(dim);

    Messure m;
    for (Index i=0; i<numIt; ++i)
    {
        // create and upload vector
        makeVector(vec, i+1);
        cuda->uploadVec(&vec[0]);

        // get best match
        if (!cuda->calcDMap())
            return false;

        Index idx=0;
        if (!cuda->getMinDMap(idx))
            return false;

        // insert
        //Index x = rand()%(w/2), y = rand()%(h/2);
        Index x = idx%w, y = idx/w;
        if (!cuda->set(x,y, 10,10, 1.0))
            return false;

        if ((i+1)%50==0)
        {
            double fps = 50.0 / m.elapsed();
            std::cout << (i+1) << "/" << numIt
                      << " : " << fps << "fps\n";
            m.start();
        }
    }
    return true;
}

template <class B>
void testBackend()
{
    using namespace RSOM;

    const Index
        w = 512,
        h = 512,
        dim = 64;

    std::vector<Float>
            map(w*h*dim),
            dmap(w*h),
            vec(dim);

    B som;

    som.setMemory(w,h,dim);
    som.uploadMap(&map[0]);

    insertSome(&som, w, h, dim, 2000);

    som.downloadMap(&map[0]);

    Som::printMap(&map[0], w, h, dim, 0.05);

    som.calcDMap();
    som.downloadDMap(&dmap[0]);
    Som::printDMap(&dmap[0], w, h);

    Index idx;
    som.getMinDMap(idx);
    std::cout << "best match: " << idx << " = " << (idx%w) << ", " << (idx/w) << "\n";
}

void compareDMap()
{
    using namespace RSOM;

    std::vector<Backend*>
            backends = { new CpuBackend,
                         new CudaBackend(64),
                         new CudaBackend(128),
                         new CudaBackend(256),
                         new CudaBackend(512),
                         new CudaBackend(1024)
                       };

    std::vector<Index>
        sizes = { 32, 64, 128, 256, 374, 512, 768, 1024
                  , 2048, 4096
                },
        dims = { 8, 16, 128, 256, 512 };
        //dims = { 1 };


    // header line

    std::cout << "          size ";
    for (size_t i=0; i<backends.size(); ++i)
    {
        std::cout << "|" << std::setw(17) << backends[i]->name() + " mops" << " ";
    }
    std::cout << "| best\n";

    Messure time;
    double cpu_time = 1;
    for (auto s=sizes.begin(); s!=sizes.end(); ++s)
    {
        for (auto d=dims.begin(); d!=dims.end(); ++d)
        {
            // print size
            std::cout
                    << std::setw(4) << *s << "x" << std::setw(4) << *s
                    << "x" << std::setw(4) << *d
                    << " ";
            std::cout.flush();

            // create some data
            int cells = *s * *s * *d;

            std::vector<Float> map(cells);
            for (int i=0; i<cells; ++i)
                map[i] = (Float)rand()/RAND_MAX;

            std::vector<Float> vec(*d);
            for (int i=0; i<*d; ++i)
                vec[i] = (Float)rand()/RAND_MAX;

            // number of operations
            int numOps = *s * *s * *d;

            double elapsed, ops, best_elapsed=0;
            int iters = 0;

            // test each backend
            for (auto be=backends.begin(); be!=backends.end(); ++be)
            {

            #define BENCH_CHECK_CUDA(command__) \
                if (!command__) { \
                    std::cout << "|" << std::setw(18) << "fail "; \
                    b->free(); \
                    goto skip_; \
                }

                // allocate memory in backend
                Backend * b = *be;
                BENCH_CHECK_CUDA( b->setMemory(*s, *s, *d) );

                // init with some data
                BENCH_CHECK_CUDA( b->uploadMap(&map[0]) );
                BENCH_CHECK_CUDA( b->uploadVec(&vec[0]) );

                iters = std::max(std::max(1,32 / *d), 100000000 / numOps);

                time.start();
                for (int i=0; i<iters; ++i)
                {
                    BENCH_CHECK_CUDA( b->calcDMap() );
                    //BENCH_CHECK_CUDA( b->debugFunc() );
                }
                if (b->name() != "cpu") CHECK_CUDA( cudaThreadSynchronize(), );

                elapsed = time.elapsed();

                // free the backend resource
                BENCH_CHECK_CUDA( b->free() );

            #undef BENCH_CHECK_CUDA

                // print statistics
                ops = (numOps * iters) / elapsed;

                std::cout
                        << "|"
                        //<< std::setw(12) << (int)(iters / elapsed);
                        << std::setw(10) << (int)(ops / 1000)
                        << " ";
                if (be==backends.begin())
                {
                    std::cout << "    -- ";
                    cpu_time = elapsed;
                }
                else
                {
                    std::cout << std::setw(5) << std::setprecision(3) << (cpu_time / elapsed) << "x ";
                    if (!best_elapsed)
                        best_elapsed = elapsed;
                    else
                        best_elapsed = std::min(best_elapsed, elapsed);
                }
            skip_:

                std::cout.flush();
            }
            // best in row
            if (cpu_time && best_elapsed)
                std::cout << "| " << std::setprecision(3) << (cpu_time / best_elapsed) << "x";

            std::cout << "\n";
        }
    }
}


int testCuda()
{
    using namespace RSOM;

    //testBackend<CudaBackend>(); return 0;
    compareDMap(); return 0;

    CpuBackend cuda;

    const Index
        w = 32,
        h = 32,
        dim = 64;

    std::vector<Float>
            map(w*h*dim),
            dmap(w*h),
            vec(dim);

    cuda.setMemory(w,h,dim);
    cuda.uploadMap(&map[0]);
    cuda.downloadMap(&map[0]);

    makeVector(vec, 0);

    cuda.uploadVec(&vec[0]);
    cuda.set(6,6, 5,5, 1.0);

    //insertSome(cuda, w, h, dim, 2000);

    cuda.downloadMap(&map[0]);

    Som::printMap(&map[0], w, h, dim, 0.05);

    cuda.calcDMap();
    cuda.downloadDMap(&dmap[0]);
    Som::printDMap(&dmap[0], w, h);

    Index idx;
    cuda.getMinDMap(idx);
    std::cout << "best match: " << idx << " = " << (idx%w) << ", " << (idx/w) << "\n";

    return 0;
}


#endif // TESTCUDA_H
