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
#ifndef CUBLAS_UTIL_H
#define CUBLAS_UTIL_H

#include <iostream>

#if (0)
    #define CUBLAS_PRINT(arg__) \
        { std::cerr << arg__ << "\n"; }
#else
    #define CUBLAS_PRINT(unused__)
#endif

/** Macro for checking for cuda errors.
    Define CHECK_CUDA before including this header to change behaviour */
#define CHECK_CUBLAS( command__, code_on_error__ ) \
{ \
    CUBLAS_PRINT( ":" << #command__ ); \
    cublasStatus_t err = command__; \
    if (err != CUBLAS_STATUS_SUCCESS) \
    { \
        std::cerr << "Cublas Error: " << err \
                          << "\nfor command '" #command__ "'\n"; \
        code_on_error__; \
    } \
}

#endif // CUBLAS_UTIL_H
