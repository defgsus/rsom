#include "som.h"

#include "log.h"

#include <sstream>

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
        do_non_duplicate    (false),
        do_index_all        (false)
{
    SOM_DEBUG("Som::Som()");
}

std::string Som::info_str() const
{
    std::stringstream s;
    s << sizex << "x" << sizey << "x" << dim << " (" << sizex*sizey*dim << ")"
      << "\ngeneration " << generation
      << "\nbest match " << stat_av_best_match
    ;
    return s.str();
}


// ------------------ map ------------------

void Som::create(int sizex, int sizey, int dim, int rand_seed)
{
    SOM_DEBUG("Som::create(" << sizex << ", " << sizey << ", " << dim << ", " << rand_seed << ")");

    this->sizex = sizex;
    this->sizey = sizey;
    this->size = sizex * sizey;
    this->size_diag = sqrt(sizex*sizex + sizey*sizey);
    this->dim = dim;

    // setup data
    map.resize(size);
    for (size_t i=0; i<size; ++i)
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
        for (size_t i=0; i<size; ++i)
            for (size_t j=0; j<dim; ++j)
                map[i][j] = 0.f;
    }
    // randomly select and take apart the sample data
    else
    {
        srand(rand_seed);

        for (size_t i=0; i<size; ++i)
            for (size_t j=0; j<dim; ++j)
                map[i][j] = 0.7*data[rand()%data.size()].data[j];
    }

    generation = 0;
}




// ------------- data handling ------------

void Som::clearData()
{
    SOM_DEBUG("Som::clearData()");

    data.clear();
}

Som::Data * Som::insertData(const float * dat, int user_id)
{
    SOM_DEBUGN(1, "Som::insertData(" << dat << ", " << user_id << ")");

    Data d;

    d.data = dat;
    d.user_id = user_id;
    d.index = -1;
    d.count = data.size();

    data.push_back(d);
    return &data.back();
}


void Som::insertWave(Wave& wave)
{
    SOM_DEBUG("Som::insertWave(" << &wave << ")");

    this->wave = &wave;

    clearData();

    for (size_t i=0; i<wave.nr_grains; ++i)
        insertData( &wave.band[i][0] );
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
    size_t nr = rand() % data.size();
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
        for (size_t k=0; k<dim; ++k, ++ps, ++pd)
            *pd += amt * (*ps - *pd);
    }

    ++generation;
}


void Som::moveData(Data * data, size_t index)
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

size_t Som::best_match(Data * data)
{
    // search everywhere
    if ( data->index < 0
        || local_search_radius >= size_diag)
    {
        size_t i = best_match(data->data);
        /// @todo need distance
        moveData(data, i);
        return i;
    }

    // search locally

    float md = 1000000.0;
    size_t index = data->index;

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

    // update index map
    moveData(data, index);

    return index;
}


size_t Som::best_match_avoid(Data * data)
{
    // search everywhere
    if ( data->index < 0
      || local_search_radius >= size_diag)
    {
        size_t i = best_match_avoid(data->data);
        /// @todo need distance
        moveData(data, i);
        return i;
    }

    // search locally

    bool found = false;
    float md = 1000000.0;
    size_t index = data->index;

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

        const size_t ind = y * sizex + x;

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

    // update index map
    moveData(data, index);

    return index;
}




size_t Som::best_match(const float* dat)
{
    float md = 10000000.0;
    int index = 0;
    for (size_t i=0; i<size; ++i)
    {
        // difference to each cell
        float d = 0.0;
        for (size_t j=0; j<dim; ++j)
            d += fabs(map[i][j] - dat[j]);

        if (d<md) { md = d; index = i; }
    }

    stat_av_best_match += 0.1 * (md - stat_av_best_match);

    return index;
}


size_t Som::best_match_avoid(const float* dat)
{
    float md = 1000000.0;
    int index = -1;

    for (size_t i=0; i<size; ++i)
    if (imap[i] < 0)
    {
        // difference to each cell
        float d = 0.0;
        for (size_t j=0; j<dim; ++j)
            d += fabs(map[i][j] - dat[j]);

        if (d<md) { md = d; index = i; }
    }

    return index;
}






// return distance between cell i1 and i2
float Som::get_distance(size_t i1, size_t i2) const
{
    float d = 0.0;
    const
    float * p1 = &map[i1][0],
          * p2 = &map[i2][0];
    for (size_t i=0; i<dim; ++i, ++p1, ++p2)
        d += fabs(*p1 - *p2);

    return d / dim;
}

float Som::get_distance(const Data * data, size_t ci) const
{
    float d = 0.0;
    const
    float * p1 = &data->data[0],
          * p2 = &map[ci][0];
    for (size_t i=0; i<dim; ++i, ++p1, ++p2)
        d += fabs(*p1 - *p2);

    return d / dim;
}

void Som::set_umap(float value)
{
    SOM_DEBUGN(0, "Som::set_umap(" << value << ")");

    for (auto i=umap.begin(); i!=umap.end(); ++i)
        *i = value;
}

void Som::set_imap(int value)
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

    float ma = 0.00001;

    if (!do_wrap)
    for (size_t j=1; j<sizey-1; ++j)
    for (size_t i=1; i<sizex-1; ++i)
    {
        int k = j*sizex + i;
        float d =
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
    for (size_t j=0; j<sizey; ++j)
    for (size_t i=0; i<sizex; ++i)
    {
        size_t
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
    for (size_t i=0; i<size; ++i)
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
            imap[index] = (float)(i * wave->grain_size)/wave->info.samplerate;
    }

    // finally, we set the non-indexed cells to 0
    //for (size_t i=0; i<size; ++i)
        //if (imap[i]<0.f) umap[i] = 0.f;
}


