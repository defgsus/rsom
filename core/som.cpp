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

/** linear filter coefficient for following with runtime statistics */
#define SOM_FOLLOW_SPEED 1.f/5000.f

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

        radius_              (1),
        alpha_               (1),
        local_search_radius_ (1),

        do_wrap_             (true),
        do_non_duplicate_    (true),

        backend_type_        (backend_type),
        backend_             (0)
{
    SOM_DEBUG("Som::Som()");
    switch (backend_type_)
    {
        case CPU:  backend_ = new CpuBackend(); break;
        case CUDA: backend_ = new CudaBackend; break;
    }
}

Som::~Som()
{
    delete backend_;
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
                map_[i*dim_+j] = samples_[index].data[j] * amp;
            }
        }
    }

    backend_->uploadMap(&map_[0]);
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
    return &imap_[0];
}

const Index * Som::getIMap() const
{
    return &imap_[0];
}






// ---------- som algorithms --------------

void Som::insert()
{
    SOM_DEBUGN(3, "Som::insert()");

    if (samples_.empty())
    {
        SOM_ERROR("Som::insert() called without data");
        return;
    }

    // select grain to train :)
    Index nr = rand() % samples_.size();
    int index = do_non_duplicate_?
                best_match_avoid_(&samples_[nr])
              : best_match_(&samples_[nr]);

    if (index<0)
    {
        ++num_failed_inserts_;
        return;
    }

    // insert sample
    const int rad = ceil(radius_);
    backend_->uploadVec(samples_[nr].data);
    backend_->set(index % sizex_, index / sizex_,
                 rad, rad, alpha_);
    /*
    // adjust cell and neighbourhood
    const float radius_sq = radius*radius;
    const int rad = ceil(radius);
    for (int j=-rad; j<=rad; ++j)
    for (int i=-rad; i<=rad; ++i)
    {
        int x = i + (int)(index % sizex);
        int y = j + (int)(index / sizex);

        // skip if not wrapping and out-of-range
        if (!do_wrap && (x<0 || y<0 || x>=(int)sizex || y>=(int)sizey)) continue;

        if (x<0) x += sizex; else if (x>=(int)sizex) x -= sizex;
        if (y<0) y += sizey; else if (y>=(int)sizey) y -= sizey;

        float r = i*i + j*j;
        if (r >= radius_sq) continue;

        r = sqrtf(r);

        // amount to adjust
        float amt = alpha * (1.f - r/radius);

        // pointer to cell
        float * pd = &map[y * sizex + x][0];
        // pointer to input data
        const float * ps = &data[nr].data[0];

        // blend data
        for (int k=0; k<dim; ++k, ++ps, ++pd)
            *pd += amt * (*ps - *pd);
    }
    */
    ++generation_;
}


void Som::moveData(DataIndex * data, Index index)
{
    if (data->index>=0)
    {
        // remove old imap point
        if (imap_[data->index] == data->count)
            imap_[data->index] = -1;
    }

    imap_[index] = data->count;

    // keep info in data
    data->index = index;
}

// ------------------ matching ----------------------

Index Som::best_match_(DataIndex * data)
{
    Index index = 0;
    backend_->uploadVec(data->data);
    backend_->calcDMap();
    backend_->getMinDMap(index);

    moveData(data, index);

    return index;

    /*
    // search everywhere
    if ( data->index < 0
        || local_search_radius >= size_diag)
    {
        Index i = best_match(data->data);
        /// @todo need distance
        moveData(data, i);
        return i;
    }

    // search locally

    float md = 1000000.0;
    Index index = data->index;

    const float radius_sq = local_search_radius * local_search_radius;
    const int rad = ceil(local_search_radius);
    for (int j=-rad; j<=rad; ++j)
    for (int i=-rad; i<=rad; ++i)
    {
        int x = i + (int)(data->index % sizex);
        int y = j + (int)(data->index / sizex);

        // skip if not wrapping and out-of-range
        if (!do_wrap && (x<0 || y<0 || x>=(int)sizex || y>=(int)sizey)) continue;
        // wrap
        if (x<0) x += sizex; else if (x>=(int)sizex) x -= sizex;
        if (y<0) y += sizey; else if (y>=(int)sizey) y -= sizey;

        // within radius?
        float r = i*i + j*j;
        if (r >= radius_sq) continue;

        const size_t ind = y * sizex + x;

        // calc difference to cell
        float d = get_distance(data, ind);
        // best match?
        if (d<md) { md = d; index = ind; }
    }

    stat_av_best_match += SOM_FOLLOW_SPEED * (md - stat_av_best_match);

    // update index map
    moveData(data, index);

    return index;
    */
}


Index Som::best_match_avoid_(DataIndex * data)
{
    return best_match_(data);
    /*
    // search everywhere
    if ( data->index < 0
      || local_search_radius >= size_diag)
    {
        Index i = best_match_avoid(data->data);
        /// @todo need distance
        moveData(data, i);
        return i;
    }

    // search locally

    bool found = false;
    float md = 1000000.0;
    Index index = data->index;

    const float radius_sq = local_search_radius * local_search_radius;
    const int rad = ceil(local_search_radius);
    for (int j=-rad; j<=rad; ++j)
    for (int i=-rad; i<=rad; ++i)
    {
        int x = i + (int)(data->index % sizex);
        int y = j + (int)(data->index / sizex);

        // skip if not wrapping and out-of-range
        if (!do_wrap && (x<0 || y<0 || x>=(int)sizex || y>=(int)sizey)) continue;
        // wrap
        if (x<0) x += sizex; else if (x>=(int)sizex) x -= sizex;
        if (y<0) y += sizey; else if (y>=(int)sizey) y -= sizey;

        const Index ind = y * sizex + x;

        // ignore vacant cells
        if (imap[ind] >= 0
            //&& imap[ind] != data->count
            ) continue;

        // within radius?
        float r = i*i + j*j;
        if (r >= radius_sq) continue;

        // calc difference to cell
        float d = get_distance(data, ind);
        // best match?
        if (d<md) { md = d; index = ind; found = true; }
    }

    if (!found) return -1;

    stat_av_best_match += SOM_FOLLOW_SPEED * (md - stat_av_best_match);

    // update index map
    moveData(data, index);

    return index;
    */
}




Index Som::best_match(const float* dat)
{
    Index index = 0;
    backend_->uploadVec(dat);
    backend_->calcDMap();
    backend_->getMinDMap(index);
    return index;
/*
    float md = 10000000.0;
    int index = 0;
    for (Index i=0; i<size; ++i)
    {
        // difference to each cell
        float d = 0.0;
        for (Index j=0; j<dim; ++j)
            d += fabs(map[i][j] - dat[j]);

        if (d<md) { md = d; index = i; }
    }

    stat_av_best_match += SOM_FOLLOW_SPEED * (md - stat_av_best_match);

    return index;
*/
}


Index Som::best_match_avoid(const float* dat)
{
    return best_match(dat);
    /*
    float md = 1000000.0;
    int index = -1;

    for (Index i=0; i<size; ++i)
    if (imap[i] < 0)
    {
        // difference to each cell
        float d = 0.0;
        for (Index j=0; j<dim; ++j)
            d += fabs(map[i][j] - dat[j]);

        if (d<md) { md = d; index = i; }
    }

    if (index>=0)
        stat_av_best_match += SOM_FOLLOW_SPEED * (md - stat_av_best_match);

    return index;
    */
}




#if (0)

// return distance between cell i1 and i2
Float Som::get_distance(Index i1, Index i2) const
{
    Float d = 0.0;
    const
    Float * p1 = &map_[i1][0],
          * p2 = &map_[i2][0];
    for (Index i=0; i<dim; ++i, ++p1, ++p2)
        d += fabs(*p1 - *p2);

    return d / dim;
}

Float Som::get_distance(const DataIndex * data, Index ci) const
{
    Float d = 0.0;
    const
    Float * p1 = &data->data[0],
          * p2 = &map[ci][0];
    for (Index i=0; i<dim; ++i, ++p1, ++p2)
        d += fabs(*p1 - *p2);

    return d / dim;
}

void Som::set_umap(Float value)
{
    SOM_DEBUGN(0, "Som::set_umap(" << value << ")");

    for (auto i=umap.begin(); i!=umap.end(); ++i)
        *i = value;
}

void Som::set_imap(Index value)
{
    SOM_DEBUGN(0, "Som::set_imap(" << value << ")");

    for (auto i=imap.begin(); i!=imap.end(); ++i)
        *i = value;
}


// calculates the distance to neighbours for each cell
void Som::calc_umap()
{
    SOM_DEBUGN(0, "Som::calc_umap()");

    set_umap();

    Float ma = 0.00001;

    if (!do_wrap)
    for (Index j=1; j<sizey-1; ++j)
    for (Index i=1; i<sizex-1; ++i)
    {
        int k = j*sizex + i;
        Float d =
              get_distance(k, (j-1)*sizex + i - 1) * 0.75
            + get_distance(k, (j-1)*sizex + i)
            + get_distance(k, (j-1)*sizex + i + 1) * 0.75
            + get_distance(k,     j*sizex + i - 1)
            + get_distance(k,     j*sizex + i + 1)
            + get_distance(k, (j+1)*sizex + i - 1) * 0.75
            + get_distance(k, (j+1)*sizex + i)
            + get_distance(k, (j+1)*sizex + i + 1) * 0.75;

        umap[k] = d;
        ma = std::max(ma, d);
    }
    else /* on do_wrap */
    for (Index j=0; j<sizey; ++j)
    for (Index i=0; i<sizex; ++i)
    {
        Index
            k = j*sizex + i,
            j0 = (j==0)? sizey-1 : j-1,
            j1 = (j==sizey-1)? 0 : j+1,
            i0 = (i==0)? sizex-1 : i-1,
            i1 = (i==sizex-1)? 0 : i+1;

        float d =
              get_distance(k, j0*sizex + i0) * 0.75
            + get_distance(k, j0*sizex + i)
            + get_distance(k, j0*sizex + i1) * 0.75
            + get_distance(k, j *sizex + i0)
            + get_distance(k, j *sizex + i1)
            + get_distance(k, j1*sizex + i0) * 0.75
            + get_distance(k, j1*sizex + i)
            + get_distance(k, j1*sizex + i1) * 0.75;

        umap[k] = d;
        ma = std::max(ma, d);
    }

    // normalize
    for (Index i=0; i<size; ++i)
        umap[i] /= ma;
}

// find the best match for each grain and store the grain's position (in seconds)
void Som::calc_imap()
{
    SOM_DEBUGN(0, "Som::calc_imap()");
/*
    // index-all mode. EVERY cell will get a grain index!
    if (do_index_all)
    {
        // each cell will be indexed with the best-matching band
        for (size_t i=0; i<size; ++i)
        {
            int index = wave->get_best_match(&map[i][0]);
            imap[i] = (float)index * wave->grain_size / wave->info.samplerate;
        }
        return;
    }
*/
    // or other way around: each band will index the best-matching cell

    // first clear the map
    // we are using best_match_avoid(), which will only consider
    // cells who's imap[] value is negative
    set_imap(-1.f);
    for (size_t i=0; i<data.size(); ++i)
    {
        int index = best_match_avoid(&data[i]);

        if (index >= 0)
            imap[index] = data[i].user_id;
                //(float)(i * wave->grain_size)/wave->info.samplerate;
    }

    // finally, we set the non-indexed cells to 0
    //for (size_t i=0; i<size; ++i)
        //if (imap[i]<0.f) umap[i] = 0.f;
}


#endif


// ---------------------- debug --------------------------

void Som::printMap(const Float * map, Index w, Index h, Index dim,
                         Float threshold, Index screen_w, Index screen_h)
{
    screen_w = std::min(screen_w, w);
    screen_h = std::min(screen_h, h);

    for (Index y=0; y<screen_h; ++y)
    {
        for (Index x=0; x<screen_w; ++x)
            std::cout << (map[(y*w+x)*dim]>=threshold? "*" : ".");
        std::cout << "\n";
    }
}

void Som::printDMap(const Float * map, Index w, Index h,
                         Float threshold, Index screen_w, Index screen_h)
{
    screen_w = std::min(screen_w, w);
    screen_h = std::min(screen_h, h);

    for (Index y=0; y<screen_h; ++y)
    {
        for (Index x=0; x<screen_w; ++x)
            std::cout << (map[y*w+x]>=threshold? "*" : ".");
        std::cout << "\n";
    }
}

} // namespace RSOM
