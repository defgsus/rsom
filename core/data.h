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

#include "som_types.h"

namespace RSOM
{



/** General purpose data container.
    Contains a number of samples/objects, each with a number of data points.
    */
class Data
{
public:

    Data();

    // ----- config --------

    /** Sets the maximum number of objects in this class.
        This will wipe out current objects if there are more,
        and will affect adding data in the future.
        Set to zero for unlimited objects (default). */
    void maxObjects(size_t num);

    /** Returns the maximum allowed objects in this container.
        Zero for unlimited. */
    size_t maxObjects() const { return max_objects_; }

    // ------ getter -------

    /** return number of objects */
    size_t numObjects() const { return data_.size(); }

    /** return number of data points in each object. */
    size_t numDataPoints() const { return num_points_; }

    /** retrieve a pointer to the indexth object's data.
        This pointer will point to numDataPoints() number of consecutive float values. */
    const Float * getObjectData(size_t index) const
        { if (index>=data_.size()) return 0; return &data_[index][0]; }

    // ----- IO ----

    const std::string& filepath() const { return filepath_; }

    bool addAsciiFile(const std::string& filename);

    bool loadAsciiDir(const std::string& pathname);

    bool addCsvFile(const std::string& filename);

    // ----- manipulation ------

    /** return the maximum value in all samples */
    Float maxValue();

    /** divide all by maxValue() */
    void normalize();

    /** clamp all values to this range. */
    void clamp(Float minval, Float maxval);

    // ------ debug ------

    void createRandomData(size_t numObjects, size_t numPoints);

private:

    /** called to signal data has changed. */
    void changed_();

    /** contains all the data */
    std::vector<std::vector<Float>> data_;
    /** last used filepath */
    std::string filepath_;
    /** number of data points. if 0, then undecided yet. */
    size_t num_points_,
    /** if this is != 0, it limits the maximum number of samples. */
        max_objects_;

    /** maximum value in all data files */
    Float max_value_;
    bool do_calc_maxval_;
};

} // namespace RSOM

#endif // DATA_H
