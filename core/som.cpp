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

#include <sstream>
#include <cmath>

/** linear filter coefficient for following with runtime statistics */
#define SOM_FOLLOW_SPEED 1.f/5000.f

namespace RSOM
{


Som::Som()
    :   sizex               (0),
        sizey               (0),
        size                (0),
        dim                 (0),
        rand_seed           (0),

        stat_av_best_match  (0),

        radius              (1),
        alpha               (1),
        local_search_radius (1),

        generation          (0),

        do_wrap             (true),
        do_non_duplicate    (true),
        do_index_all        (false)
{
    SOM_DEBUG("Som::Som()");
}

std::string Som::info_str() const
{
    std::stringstream s;
    s << sizex << "x" << sizey << "x" << dim << " (" << sizex*sizey << "x" << dim << ")"
      << "\ngeneration " << generation << " (" << (generation/1000000) << "M)"
      << "\nepoch      " << generation / std::max((size_t)1, data.size())
      << "\nbest match " << stat_av_best_match
    ;
    return s.str();
}


// ------------------ map ------------------

void Som::create(Index sizex, Index sizey, Index dim, int rand_seed)
{
    SOM_DEBUG("Som::create(" << sizex << ", " << sizey << ", " << dim << ", " << rand_seed << ")");

    this->sizex = sizex;
    this->sizey = sizey;
    this->size = sizex * sizey;
    this->size_diag = sqrt(sizex*sizex + sizey*sizey);
    this->dim = dim;
    this->rand_seed = rand_seed;

    // setup data
    map.resize(size);
    for (Index i=0; i<size; ++i)
        map[i].resize(dim);

    umap.resize(size);
    imap.resize(size);

    generation = 0;
}

void Som::initMap()
{
    SOM_DEBUG("Som::initMap()");

    // clear stats
    stat_av_best_match = 0;

    // clear data -> map specifics
    for (auto i=data.begin(); i!=data.end(); ++i)
        i->index = -1;

    // clear umap
    for (auto i=umap.begin(); i!=umap.end(); ++i)
        *i = 0;

    // clear imap
    for (auto i=imap.begin(); i!=imap.end(); ++i)
        *i = -1;

    // rather zero-out the map?
    if (rand_seed == -1)
    {
        for (Index i=0; i<size; ++i)
            for (Index j=0; j<dim; ++j)
                map[i][j] = 0.f;
    }
    // randomly select and take apart the sample data
    else
    {
        srand(rand_seed);

        for (Index i=0; i<size; ++i)
        {
            // circular amplitude
            const float x = (float)(i%sizex)/sizex - 0.5f,
                        y = (float)(i/sizex)/sizey - 0.5f,
                        amp = 1.f / (1.f + 5.f * sqrtf(x*x+y*y));

            // one random sample
            const Index dat = rand()%data.size();
            for (Index j=0; j<dim; ++j)
            {
                // look around a bit for each band
                const Index index = std::max(0, std::min((Index)data.size()-1,
                        dat + (rand()%20) - 10
                    ));
                map[i][j] = data[index].data[j] * amp;
            }
        }
    }

    generation = 0;
}




// ------------- data handling ------------

Som::DataIndex * Som::createDataIndex(const Float * dat, int user_id)
{
    SOM_DEBUGN(1, "Som::createDataIndex(" << dat << ", " << user_id << ")");

    DataIndex d;

    d.data = dat;
    d.user_id = user_id;
    d.index = -1;
    d.count = data.size();

    data.push_back(d);
    return &data.back();
}


void Som::setData(const Data * data)
{
    SOM_DEBUG("Som::setData(" << data << ")");

    dataContainer = data;

    // clear previous indices
    this->data.clear();
    dim = 0;

    if (data == 0) return;

    dim = data->numDataPoints();

    for (size_t i=0; i<data->numObjects(); ++i)
        createDataIndex( data->getObjectData(i), i );
}










// ---------- som algorithms --------------

void Som::insert()
{
    SOM_DEBUGN(3, "Som::insert()");

    if (!data.size())
    {
        SOM_ERROR("Som::insert() called without data");
        return;
    }

    // select grain to train :)
    Index nr = rand() % data.size();
    int index = do_non_duplicate?
                best_match_avoid(&data[nr])
              : best_match(&data[nr]);

    if (index<0) return;

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

    ++generation;
}


void Som::moveData(DataIndex * data, Index index)
{
    if (data->index>=0)
    {
        // remove old imap point
        if (imap[data->index] == data->count)
            imap[data->index] = -1;
    }

    imap[index] = data->count;

    // keep info in data
    data->index = index;
}

// ------------------ matching ----------------------

Index Som::best_match(DataIndex * data)
{
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
}


Index Som::best_match_avoid(DataIndex * data)
{
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
}




Index Som::best_match(const float* dat)
{
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
}


Index Som::best_match_avoid(const float* dat)
{
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
}






// return distance between cell i1 and i2
Float Som::get_distance(Index i1, Index i2) const
{
    Float d = 0.0;
    const
    Float * p1 = &map[i1][0],
          * p2 = &map[i2][0];
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


} // namespace RSOM
