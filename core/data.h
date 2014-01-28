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

#ifndef DATA_H
#define DATA_H

#include <string>
#include <vector>

/** General purpose data container.
    Contains a number of samples/objects, each with a number of data points.
    */
class Data
{
public:
    /** floating point format used by this class. */
    typedef float Float;

    Data();

    /** return number of objects */
    size_t numObjects() const { return data_.size(); }

    /** return number of data points in each object. */
    size_t numDataPoints() const { return data_.size(); }

    /** retrieve a pointer to the indexth object's data.
        This pointer will point to numDataPoints() number of consecutive float values. */
    const Float * data(size_t index) const
        { if (index>=data_.size()) return 0; return &data_[index][0]; }

    // debug

    void createRandomData(size_t numObjects, size_t numPoints);

private:
    /** contains all the data */
    std::vector<std::vector<Float>> data_;
};

#endif // DATA_H
