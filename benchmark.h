#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "core/cpubackend.h"
#include "core/cudabackend.h"

namespace RSOM
{

template <class Work>
void benchmark(const std::vector<RSOM::Backend*>& backends,
               const std::vector<RSOM::Index>& sizes,
               const std::vector<RSOM::Index>& dims)
{
    Work work;

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

            if (!work.initData(*s, *s, *d))
            {
                std::cout << "failed to init data\n";
                continue;
            }

            double elapsed, ops, best_elapsed=0;

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
                BENCH_CHECK_CUDA( work.initBackend(b) );

                time.start();

                BENCH_CHECK_CUDA( work.work() );

                if (b->name() != "cpu") CHECK_CUDA( cudaThreadSynchronize(), );

                elapsed = time.elapsed();

                // free the backend resource
                BENCH_CHECK_CUDA( work.freeBackend() );

            #undef BENCH_CHECK_CUDA


                // print statistics

                ops = (double)work.numOperations() / elapsed;

                std::cout
                        << "|"
                        //<< std::setw(12) << (int)(iters / elapsed);
                        << std::setw(10) << (int)(ops / 1000)
                        << " ";

                // print speed-up

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

                // when something failed
            skip_:

                std::cout.flush();
            }

            // best in row
            std::cout << "|";
            if (cpu_time && best_elapsed)
                std::cout << " " << std::setprecision(3) << (cpu_time / best_elapsed) << "x";

            std::cout << "\n";
        }
    }

    for (auto i=backends.begin(); i!=backends.end(); ++i)
        delete *i;
}



struct WorkPrototype
{
    std::vector<Float> map, vec;
    Index w,h,dim;
    Backend * b;

    virtual int numOperations() const = 0;
    virtual bool work() = 0;

    bool initData(Index w_, Index h_, Index dim_)
    {
        w = w_;
        h = h_;
        dim = dim_;

        // create some data

        try
        {
            map.resize(w*h*dim);
            for (int i=0; i<w*h*dim; ++i)
                map[i] = (Float)rand()/RAND_MAX;

            vec.resize(dim);
            for (int i=0; i<dim; ++i)
                vec[i] = (Float)rand()/RAND_MAX;
        }
        catch (...) { return false; }

        return true;
    }

    bool initBackend(Backend * b_)
    {
        b = b_;

        if (! b->setMemory(w, h, dim) ) return false;

        // init with some data
        if (! b->uploadMap(&map[0]) ) return false;
        if (! b->uploadVec(&vec[0]) ) return false;

        return true;
    }

    bool freeBackend()
    {
        return b->free();
    }

};




void benchmarkDmap()
{
    std::cout << "\nbenchmark: difference map\n";

    struct WorkDMap : public WorkPrototype
    {
        int numOperations() const { return w * h * dim * numIterations(); }

        int numIterations() const { return std::max(std::max(1,32 / dim), 100000000 / (w*h*dim)); }

        bool work()
        {
            int iter = numIterations();

            for (int i=0; i<iter; ++i)
                if (!b->calcDMap()) return false;

            return true;
        }
    };

    benchmark<WorkDMap>(
                        { new CpuBackend,
                          new CudaBackend(64),
                          new CudaBackend(128),
                          new CudaBackend(256),
                          new CudaBackend(512),
                          new CudaBackend(1024)
                          //,new CublasBackend()
                        },
                        // sizes
                        { 32, 64, 128, 256, 374, 512, 768, 1024, 2048, 4096 },
                        // dimensions
                        { 8, 16, 128, 256, 512 } );
}

void benchmarkInsert()
{
    std::cout << "\nbenchmark: insert into map (radius = half sidelength)\n";

    struct WorkInsert : public WorkPrototype
    {
        int numOperations() const { return w * h * dim * numIterations(); }

        int numIterations() const { return std::max(2, 100000000 / (w*h*dim)); }

        bool work()
        {
            int iter = numIterations();
            int rx = w / 2, ry = h / 2;
            for (int i=0; i<iter; ++i)
                if (!b->set(rand()%w, rand()%h, rx, ry, 0.1)) return false;
                //if (!b->set(w/2, h/2, rx, 10, 0.1)) return false;
            return true;
        }
    };

    benchmark<WorkInsert>(
                        { new CpuBackend,
                          new CudaBackend(64),
                          new CudaBackend(128),
                          new CudaBackend(256),
                          new CudaBackend(512),
                          new CudaBackend(1024)
                          //,new CublasBackend()
                        },
                        // sizes
                        { 32, 64, 128, 256, 374, 512, 768, 1024 },
                        // dimensions
                        { 8, 16, 128, 256, 512, 1024, 2048 } );
}



void benchmarkBestmatch()
{
    std::cout << "\nbenchmark: bestmatch in dmap\n";

    struct WorkBestmatch : public WorkPrototype
    {
        int numOperations() const { return w * h * numIterations(); }

        int numIterations() const { return std::max(2, 10000000 / (w*h)); }

        bool work()
        {
            int iter = numIterations();
            Index index;

            for (int i=0; i<iter; ++i)
                if (!b->getMinDMap(index))
                    return false;

            return true;
        }
    };

    benchmark<WorkBestmatch>(
                        { new CpuBackend,
                          new CudaBackend(64),
                          new CudaBackend(128),
                          new CudaBackend(256),
                          new CudaBackend(512),
                          new CudaBackend(1024)
                          //,new CublasBackend()
                        },
                        // sizes
                        { 32, 64, 128, 256, 374, 512, 768, 1024, 2048 },
                        // dimensions
                        { 1 } );
}

void benchmarkAll()
{
    benchmarkInsert();
    benchmarkDmap();
    benchmarkBestmatch();
}


} // namespace RSOM

#endif // BENCHMARK_H
