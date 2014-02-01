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

#include "data.h"

#include <cstdlib>
#include <fstream>
#include <array>

#include "scandir.h"
#include "log.h"

Data::Data()
    :   num_points_ (0),
        max_value_  (0)
{
}



void Data::createRandomData(size_t numObjects, size_t numPoints)
{
    num_points_ = numPoints;

    data_.resize(numObjects);
    for (auto o=data_.begin(); o!=data_.end(); ++o)
    {
        o->resize(numPoints);
        for (auto p=o->begin(); p!=o->end(); ++p)
        {
            *p = (Float)rand() / RAND_MAX;
        }
    }
}


bool Data::addAsciiFile(const std::string& filename)
{
    SOM_DEBUG("Data::addAsciFile(" << filename << ")");

    std::ifstream f;
    f.open(filename, std::ios_base::in);
    if (!f.is_open())
    {
        SOM_ERROR("could not open ascii file " << filename);
        return false;
    }

    // create new data entry
    data_.push_back(std::vector<Float>());
    std::vector<Float> * vec = &data_.back();

    Float local_max = 0.0;

    Float num;
    while (f.good())
    {
        try { f >> num; }
        catch (...) { SOM_DEBUG("std::ifstream exception"); break; }

        // break if more data than needed
        if (num_points_ && vec->size() >= num_points_)
        {
            SOM_ERROR("more data in file than appreciated.");
            break;
        }

        // add to vector
        vec->push_back(num);
        max_value_ = std::max(max_value_, num);
        local_max = std::max(local_max, num);
    };

    // store initial data length
    if (!num_points_)
        num_points_ = vec->size();
    // check if too less data
    else
    while (vec->size() < num_points_)
    {
        vec->push_back((Float)0);
    }

    if (local_max)
    {
        for (auto i=vec->begin(); i!=vec->end(); ++i)
            *i /= local_max;
    }

    SOM_DEBUG("Data::addAsciiFile:: added object with " << num_points_ << " data points.");

    return true;
}


bool Data::loadAsciiDir(const std::string& pathname)
{
    filepath_ = pathname;
    //createRandomData(1000, 200);

    ScanDir scan;
    if (scan.scandir(pathname, "", false) != 0) return false;

    SOM_DEBUG("found " << scan.files.size() << " files.");

    bool res = false;
    for (auto i=scan.files.begin(); i!=scan.files.end(); ++i)
    {
        res |= addAsciiFile(*i);
    }
    return res;
}

void Data::normalize()
{
    if (!max_value_) return;

    for (auto o=data_.begin(); o!=data_.end(); ++o)
    {
        for (auto p=o->begin(); p!=o->end(); ++p)
        {
            *p /= max_value_;
        }
    }
}




bool Data::addCsvFile(const std::string& filename)
{
    SOM_DEBUG("Data::addCsvFile(" << filename << ")");

    std::ifstream f;
    f.open(filename, std::ios_base::in);
    if (!f.is_open())
    {
        SOM_ERROR("could not open csv file " << filename);
        return false;
    }

    /*std::vector<char> line(1024);
    f.getline(&line[0], line.size());
    std::cout << "[" << &line[0] << "]";
    return true;*/

#define CSV_LINE_BREAK while (f.good() && f.get() != '\r');

    CSV_LINE_BREAK

    std::string str;
    while (f.good())
    {
        f >> str;
        //std::cout << "\n[" << str << "] ";

        // create new data entry
        std::vector<Float> vec;

        Float local_max = 0.0;

        Float num;
        while (f.good() && vec.size() < 12)
        {
            try { f >> num; }
            catch (...) { SOM_DEBUG("std::ifstream exception"); break; }
            //std::cout << num << "\n";

            // break if more data than needed
            if (num_points_ && vec.size() >= num_points_)
            {
                SOM_ERROR("more data in file than appreciated.");
                break;
            }

            // add to vector
            vec.push_back(num);
            max_value_ = std::max(max_value_, num);
            local_max = std::max(local_max, num);
        };

        // store initial data length (when this is first item)
        if (!num_points_)
            num_points_ = vec.size();

        // check if too less data
        else
        while (vec.size() < num_points_)
        {
            vec.push_back((Float)0);
        }

        // normalize
        if (false && local_max)
        {
            for (auto i=vec.begin(); i!=vec.end(); ++i)
                *i /= local_max;
        }

        data_.push_back(vec);
    }

    SOM_DEBUG("Data::addAsciiFile:: added " << data_.size()
              << " objects with " << num_points_ << " data points.");

    return true;
}
