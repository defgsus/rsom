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
#include "som.h"

#include "data.h"
#include "log.h"
#include "cpubackend.h"
#include "cudabackend.h"

#include <sstream>
#include <cmath>
#include <fstream>

/** linear filter coefficient for following with runtime statistics */
#define SOM_FOLLOW_SPEED 1.f/300.f

namespace RSOM
{


Som::Som(BackendType backend_type)
    :   sizex_               (0),
        sizey_               (0),
        size_                (0),
        dim_                 (0),
        rand_seed_           (0),

        stat_av_best_match_  (0),
        generation_          (0),
        num_failed_inserts_  (0),

        radius_              (0.2),
        alpha_               (0.05),
        local_search_radius_ (2),

        do_wrap_             (true),
        do_non_duplicate_    (true),

        backend_type_        (backend_type),
        backend_             (0)
{
    SOM_DEBUG("Som::Som()");
    switch (backend_type_)
    {
        case CPU:  backend_ = new CpuBackend(); break;
        case CUDA: backend_ = new CudaBackend(512); break;
    }
}

Som::~Som()
{
    delete backend_;
}

bool Som::saveMap(const std::string& filename)
{
    SOM_DEBUG("Som::saveMap(" << filename << ")");

    std::ofstream f;
    f.open(filename.c_str(), std::ios_base::out);
    if (!f.is_open()) return false;

    getMap();

    f << sizex_ << " " << sizey_ << " " << dim_;
    for (auto i = map_.begin(); i!=map_.end(); ++i)
        f << " " << *i;

    return true;
}

bool Som::loadMap(const std::string& filename)
{
    SOM_DEBUG("Som::loadMap(" << filename << ")");

    std::ifstream f;
    f.open(filename.c_str(), std::ios_base::in);
    if (!f.is_open()) return false;

    try
    {
        f >> sizex_ >> sizey_ >> dim_;

        create(sizex_, sizey_, dim_, 0);
        initMap();

        for (auto i = map_.begin(); i!=map_.end(); ++i)
            f >> *i;

        backend_->uploadMap(&map_[0]);
        backend_->uploadIMap(&imap_[0]);
    }
    catch (...)
    {
        SOM_ERROR("error while reading " << filename);
        return false;
    }

    return true;
}








std::string Som::info_str() const
{
    std::stringstream s;
    s << sizex_ << "x" << sizey_ << "x" << dim_ << " (" << sizex_*sizey_ << "x" << dim_ << ")"
      << "\ngeneration     " << generation_ << " (" << (generation_/1000000) << "M)"
      << "\nepoch          " << generation_ / std::max((size_t)1, samples_.size())
      << "\nbest match     " << stat_av_best_match_
      << "\nfailed inserts " << num_failed_inserts_
    ;
    return s.str();
}


// ------------------ map ------------------

void Som::create(Index sizex, Index sizey, Index dim, int rand_seed)
{
    SOM_DEBUG("Som::create(" << sizex << ", " << sizey << ", " << dim << ", " << rand_seed << ")");

    sizex_      = sizex;
    sizey_      = sizey;
    size_       = sizex * sizey;
    size_diag_  = sqrt(sizex*sizex + sizey*sizey);
    dim_        = dim;
    rand_seed_  = rand_seed;
    generation_ = 0;

    // setup data
    map_.resize(size_ * dim_);
    umap_.resize(size_);
    imap_.resize(size_);

    // setup backend
    backend_->setMemory(sizex_, sizey_, dim_);
}

void Som::initMap()
{
    SOM_DEBUG("Som::initMap()");

    // clear stats
    stat_av_best_match_ = 0;
    num_failed_inserts_ = 0;
    generation_ = 0;

    // clear data -> map specifics
    for (auto i=samples_.begin(); i!=samples_.end(); ++i)
        i->index = -1;

    // clear umap
    for (auto i=umap_.begin(); i!=umap_.end(); ++i)
        *i = 0;

    // clear imap
    for (auto i=imap_.begin(); i!=imap_.end(); ++i)
        *i = -1;

    // rather zero-out the map?
    if (rand_seed_ == -1)
    {
        for (Index i=0; i<size_ * dim_; ++i)
            map_[i] = 0.f;
    }
    // randomly select and take apart the sample data
    else
    {
        srand(rand_seed_);

        for (Index i=0; i<size_; ++i)
        {
            // circular amplitude
            const float x = (float)(i%sizex_)/sizex_ - 0.5f,
                        y = (float)(i/sizex_)/sizey_ - 0.5f,
                        amp = 1.f / (1.f + 5.f * sqrtf(x*x+y*y));

            // one random sample
            const Index dat = rand()%samples_.size();
            for (Index j=0; j<dim_; ++j)
            {
                // look around a bit for each band
                const Index index = std::max(0, std::min((Index)samples_.size()-1,
                        dat + (rand()%20) - 10
                    ));
                map_[j*size_+i] = samples_[index].data[j] * amp;
            }
        }
    }

    backend_->uploadMap(&map_[0]);
    backend_->uploadIMap(&imap_[0]);
}




// ------------- data handling ------------

Som::DataIndex * Som::createDataIndex(const Float * dat, int user_id)
{
    SOM_DEBUGN(1, "Som::createDataIndex(" << dat << ", " << user_id << ")");

    DataIndex d;

    d.data = dat;
    d.user_id = user_id;
    d.index = -1;
    d.count = samples_.size();

    samples_.push_back(d);
    return &samples_.back();
}


void Som::setData(const Data * data)
{
    SOM_DEBUG("Som::setData(" << data << ")");

    data_container_ = data;

    // clear previous indices
    samples_.clear();
    dim_ = 0;

    if (data == 0) return;

    dim_ = data->numDataPoints();

    for (size_t i=0; i<data->numObjects(); ++i)
        createDataIndex( data->getObjectData(i), i );
}

Float * Som::getMap()
{
    backend_->downloadMap(&map_[0]);
    return &map_[0];
}

const Float * Som::getMap() const
{
    backend_->downloadMap(&map_[0]);
    return &map_[0];
}

Index * Som::getIMap()
{
    backend_->downloadIMap(&imap_[0]);
    return &imap_[0];
}

const Index * Som::getIMap() const
{
    backend_->downloadIMap(&imap_[0]);
    return &imap_[0];
}






// ---------- som algorithms --------------

void Som::insert(Index sample_index)
{
    SOM_DEBUGN(3, "Som::insert()");

    if (samples_.empty())
    {
        SOM_ERROR("Som::insert() called without data");
        return;
    }

    // select grain to train :)
    Index nr = sample_index<0?
                rand() % samples_.size()
              : std::min(sample_index, (Index)samples_.size());

    // find best match
    Float diff;
    int index;

    // cell was not indexed yet?
    if (samples_[nr].index<0 || local_search_radius_ >= size_diag_)
        index = best_match_(&samples_[nr], do_non_duplicate_, &diff);
    // search only in neighbourhood
    else
    {
        Index x = std::max(0, (int)(samples_[nr].index%sizex_ - local_search_radius_)),
              y = std::max(0, (int)(samples_[nr].index/sizex_ - local_search_radius_)),
              w = std::min(sizex_ - x, (int)local_search_radius_),
              h = std::min(sizey_ - y, (int)local_search_radius_);
        index = best_match_window_(&samples_[nr], do_non_duplicate_, &diff,
                                   x,y,w,h);
    }

    // failed??
    if (index<0)
    {
        ++num_failed_inserts_;
        return;
    }

    // insert sample
    //backend_->uploadVec(samples_[nr].data); // vector is already there
    const int rad = ceil(radius_);
    backend_->set(index % sizex_, index / sizex_, rad, rad, alpha_);

    // keep track of indices
    moveData(&samples_[nr], index);

    // keep track of best match
    stat_av_best_match_ += SOM_FOLLOW_SPEED * (diff - stat_av_best_match_);

    ++generation_;
}


void Som::moveData(DataIndex * data, Index index)
{
    // this sample is on the map already?
    if (data->index>=0)
    {
        // remove old imap point
        if (imap_[data->index] == data->count)
        {
            imap_[data->index] = -1;
            backend_->setIMapValue(data->index, -1);
        }
    }

    imap_[index] = data->count;
    backend_->setIMapValue(index, data->count);

    // keep info in data
    data->index = index;
}

// ------------------ matching ----------------------

Index Som::best_match_(DataIndex * data, bool only_vacant, Float * pvalue)
{
    const Float HIGH_VALUE = 1000000;
    Index index = 0;
    Float value = 0;

    backend_->uploadVec(data->data);

    // take this sample out of the map while searching
    if (data->index>=0)
        backend_->setIMapValue(data->index, -1);

    // find best match
    backend_->calcDMap(only_vacant, HIGH_VALUE);
    backend_->getMinDMap(index, value);

    // all cells where occupied??
    if (only_vacant && value >= HIGH_VALUE)
        return -1;

    // copy diff value
    if (pvalue) *pvalue = value;
    return index;
}

Index Som::best_match_window_(DataIndex * data, bool only_vacant, Float * pvalue,
                              Index x, Index y, Index w, Index h)
{
    const Float HIGH_VALUE = 1000000;
    Index index = 0;
    Float value = 0;
    backend_->uploadVec(data->data);

    // take this sample out of the map while searching
    if (data->index>=0)
        backend_->setIMapValue(data->index, -1);

    // find best match
    backend_->calcDMap(x, y, w, h, only_vacant, HIGH_VALUE);
    backend_->getMinDMap(index, value, x, y, w, h);

    // all cells where occupied??
    if (only_vacant && value >= HIGH_VALUE)
        return -1;

    // copy diff value
    if (pvalue) *pvalue = value;
    return index;
}







// ---------------------- debug --------------------------

void Som::printMap(const Float * map, Index w, Index h, Index dim_offset,
                         Float threshold, Index screen_w, Index screen_h)
{
    screen_w = std::min(screen_w, w);
    screen_h = std::min(screen_h, h);

    for (Index y=0; y<screen_h; ++y)
    {
        for (Index x=0; x<screen_w; ++x)
            std::cout << (map[dim_offset * w * h + y*w+x]>=threshold? "*" : ".");
        std::cout << "\n";
    }
}


} // namespace RSOM
