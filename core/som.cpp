#include "som.h"

#include "log.h"

#include <sstream>

Som::Som()
    :   sizex       (0),
        sizey       (0),
        size        (0),
        dim         (0),
        rand_seed   (0),

        stat_av_best_match  (0),

        radius      (0.1),
        alpha       (0.1),

        draw_mode   (0),
        nth_band    (0),
        generation  (0),
        do_wrap     (false),
        do_index_all(false)
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


void Som::create(int sizex, int sizey, int dim, int rand_seed)
{
    SOM_DEBUG("Som::create(" << sizex << ", " << sizey << ", " << dim << ", " << rand_seed << ")");

    this->sizex = sizex;
    this->sizey = sizey;
    this->size = sizex * sizey;
    this->dim = dim;

    // setup data
    data.resize(size);
    for (size_t i=0; i<size; ++i)
        data[i].resize(dim);

    umap.resize(size);

}

// fill randomly
// this will use the band data in 'wave' to initialize the map
// to avoid a strong bias, the band data is taken apart and recombined randomly
void Som::init(Wave& wave)
{
    SOM_DEBUG("Som::init(" << &wave << ")");

    this->wave = &wave;

    // clear stats
    stat_av_best_match = 0;

    // rather zero-out the map?
    if (rand_seed == -1)
    {
        for (size_t i=0; i<size; ++i)
            for (size_t j=0; j<dim; ++j)
                data[i][j] = 0.f;
    }
    else
    {
        if (rand_seed == 0)
            srand(time(NULL));
        else
            srand(rand_seed);

        for (size_t i=0; i<size; ++i)
            for (size_t j=0; j<dim; ++j)
                data[i][j] = 0.7*wave.band[rand()%wave.nr_grains][j];
    }

    generation = 0;
}


// ---------- som algorithms --------------

// find best matching entry for data
// dat must point to 'dim' floats
size_t Som::best_match(const float* dat)
{
    float md = 10000000000.0;
    int index = 0;
    for (size_t i=0; i<size; ++i)
    {
        // difference to each cell
        float d = 0.0;
        for (size_t j=0; j<dim; ++j)
            d += fabs(data[i][j] - dat[j]);

        if (d<md) { md = d; index = i; }
    }

    stat_av_best_match += 0.1 * (md - stat_av_best_match);

    return index;
}

// find best matching entry for data
// avoids fields who's 'umap' value is equal to or above 'thresh'
// 'dat' must point to 'dim' floats
// (this is used to assign each cell a paritcular grain)
size_t Som::best_match_avoid(const float* dat, float thresh)
{
    float md = 10000000000.0;
    int index = 0;
    for (size_t i=0; i<size; ++i)
    if (umap[i]<thresh)
    {
        // difference to each cell
        float d = 0.0;
        for (size_t j=0; j<dim; ++j)
            d += fabs(data[i][j] - dat[j]);

        if (d<md) { md = d; index = i; }
    }

    return index;
}

void Som::insert(const float* dat)
{
    size_t index = best_match(dat);

    // adjust cell and neighbourhood
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

        float r = sqrtf(i*i+j*j);
        if (r>=radius) continue;

        // pointer to cell
        float * pd = &data[y * sizex + x][0];
        // pointer to input data
        const float * ps = dat;
        // amount to adjust
        float amt = alpha * (1.f - r/radius);

        for (size_t k=0; k<dim; ++k, ++ps, ++pd)
            *pd += amt * (*ps - *pd);
    }

    ++generation;
}


// return distance between cell i1 and i2
float Som::get_distance(size_t i1, size_t i2)
{
    float d = 0.0;
    float * p1 = &data[i1][0],
          * p2 = &data[i2][0];
    for (size_t i=0; i<dim; ++i, ++p1, ++p2)
        d += fabs(*p1 - *p2);

    return d / dim;
}

void Som::set_umap(float value)
{
    SOM_DEBUG("Som::set_umap(" << value << ")");

    for (size_t i=0; i<size; ++i)
        umap[i] = value;
}

// calculates the distance to neighbours for each cell
void Som::calc_umap()
{
    SOM_DEBUG("Som::calc_umap()");

    set_umap();

    float ma = 0.00001;

    if (!do_wrap)
    for (size_t j=1; j<sizey-1; ++j)
    for (size_t i=1; i<sizex-1; ++i)
    {
        int k = j*size + i;
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
            k = j*size + i,
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
    SOM_DEBUG("Som::calc_imap()");

    // index-all mode. EVERY cell will get a grain index!
    if (do_index_all)
    {
        // each cell will be indexed with the best-matching band
        for (size_t i=0; i<size; ++i)
        {
            int index = wave->get_best_match(&data[i][0]);
            umap[i] = (float)index * wave->grain_size / wave->info.samplerate;
        }
        return;
    }

    // or other way around: each band will index the best-matching cell

    // first clear the map
    // we are using best_match_avoid() below, which will only consider
    // cells who's umap[] value is below zero!
    set_umap(-1.f);
    for (size_t i=0; i<wave->nr_grains; ++i)
    {
        int index = best_match_avoid(&wave->band[i][0]);

        if (umap[index]<0.f)
            umap[index] = (float)(i*wave->grain_size)/wave->info.samplerate;
    }

    // finally, we set the non-indexed cells to 0
    for (size_t i=0; i<size; ++i)
        if (umap[i]<0.f) umap[i] = 0.f;
}


/*
// draw the map via opengl imidiate calls
void draw(int xoff, int yoff, int width, int height)
{
    // size of one data-point
    float   xs = (float)width / size,
            ys = (float)height / size;

    if (draw_mode == 1) make_umap(); else
    if (draw_mode == 2) make_imap();

    for (int j=0; j<size; ++j)
    for (int i=0; i<size; ++i)
    {
        // determine color
        float f;
        switch (draw_mode)
        {
            default:    f = data[j*size+i][nth_band]; break;
            case 1:     f = umap[j*size+i]; break;
            case 2:     f = umap[j*size+i] / wave->length_in_secs; break;
        }
        color.set_color(f);

        float   x = (float)i * xs + xoff,
                y = (float)j * ys + yoff;

        // filled rectangle
        glBegin(GL_QUADS);
            glVertex2f(x,y);
            glVertex2f(x+xs,y);
            glVertex2f(x+xs,y+ys);
            glVertex2f(x,y+ys);
        glEnd();
    }
}
*/

