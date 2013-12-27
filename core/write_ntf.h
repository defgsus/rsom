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
    @brief Native Table Format writer (for use with reaktor_som)

    @version 2012/07/11 started
    @version 2012/12/26 untied from som.h

    copyright 2013 stefan.berke @ modular-audio-graphics.com
*/
#ifndef WRITE_NTF_H_INCLUDED
#define WRITE_NTF_H_INCLUDED

#include <string>

/** Stores the data as 'native table format'.
    min_value & max_value are used by Reaktor for displaying.
    data[] is expected as row-major [y*sizex+x]. */
bool save_ntf(const std::string& filename,
              float min_value, float max_value,
              int sizex, int sizey, const float * data);


#endif // WRITE_NTF_H_INCLUDED
