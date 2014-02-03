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

#ifndef CUDA_UTIL_H
#define CUDA_UTIL_H

/**	@file

    @brief Cuda utility functions

    @author def.gsus-
    @version 2014/01/28 started

    <p>This header can be included by host and device code.
    It basically helps with error-checking.</p>
*/

#include <cstdlib>
#include <iostream>
#include <iomanip>

#include <cuda_runtime.h>

#ifndef NDEBUG
#define DEBUG_CUDA(arg__) { std::cerr << arg__ << "\n"; }
#else
#define DEBUG_CUDA(unused__) { }
#endif

#ifndef NDEBUG
#define DEBUG_CUDA_ERROR(arg__) { std::cerr << arg__ << "\n"; }
#else
#define DEBUG_CUDA_ERROR(unused__) { }
#endif

#ifndef CHECK_CUDA
    /** Macro for checking for cuda errors.
        Define CHECK_CUDA before including this header to change behaviour */
    #define CHECK_CUDA( command__, code_on_error__ ) \
    { \
        DEBUG_CUDA( ":" << #command__ ); \
        cudaError_t err = command__; \
        if (err != cudaSuccess) \
        { \
            DEBUG_CUDA_ERROR("Cuda Error: " << cudaGetErrorString(err) \
                              << "\nfor command '" #command__ "'"); \
            code_on_error__; \
        } \
    }
#endif

#ifndef CHECK_CUDA_KERNEL
    /** Macro for checking for cuda errors after kernel calls.
        Define CHECK_CUDA before including this header to change behaviour */
    #define CHECK_CUDA_KERNEL( kcommand__, code_on_error__ ) \
    { \
        DEBUG_CUDA( ":" << #kcommand__ ); \
        kcommand__; \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) \
        { \
            DEBUG_CUDA_ERROR("Cuda Error: " << cudaGetErrorString(err) \
                             << "\nfor kernel call '" #kcommand__ "'"); \
            code_on_error__; \
        } \
    }
#endif

/** return the next power of two for x */
template <typename I>
I nextPow2( I x )
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}


#endif // CUDA_UTIL_H
