/*  This is free software; you can redistribute it and/or
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
*/
/** @file
    @brief Native Table Format writer for use with reaktor_som

    @version 2012/07/11 started

    copyright 2013 stefan.berke @ modular-audio-graphics.com
*/
#ifndef WRITE_NTF_H_INCLUDED
#define WRITE_NTF_H_INCLUDED

#include "som.h"

// store the som data as 'native table format'
// actually, only the 'umap' data is saved which at this point
// should contain the grain-positions for each som-node
bool save_ntf(const std::string& filename, const Som& som);


#endif // WRITE_NTF_H_INCLUDED
