/** @file
    @brief self-organizing map class for use with reaktor_som

    @author def.gsus- (berke@cymatrix.org)
    @version 2012/07/11
    @version 2013/12/18 removed opengl stuff

    copyright 2012, 2013 stefan.berke @ modular-audio-graphics.com

    This program is coverd by the GNU General Public License
*/
#ifndef SOM_H_INCLUDED
#define SOM_H_INCLUDED

#include <cstdlib>
#include <ctime>
#include "wavefile.h"

/** kohonen or self-organizing map */
class SOM
{
    public:

    SOM()
        :   radius      (5),
            alpha       (0.01),
            draw_mode   (0),
            nth_band    (0),
            generation  (0),
            do_wrap     (false),
            do_index_all(false)
    { }

    void create(int size, int dim, int rand_seed)
    {
        this->size = size;
        this->dim = dim;
        this->radius = size / 4;
        this->rand_seed = rand_seed;

        // setup data
        data.resize(size * size);
        for (int i=0; i<size*size; ++i)
            data[i].resize(dim);

        umap.resize(size*size);

    }

    // fill randomly
    // this will use the band data in 'wave' to initialize the map
    // to avoid a strong bias, the band data is taken apart and recombined randomly
    void init(Wave& wave)
    {
        this->wave = &wave;

        // rather zero-out the map?
        if (rand_seed == -1)
        {
            for (int i=0; i<size*size; ++i)
                for (int j=0; j<dim; ++j)
                    data[i][j] = 0.f;
        }
        else
        {
            if (rand_seed == 0)
                srand(time(NULL));
            else
                srand(rand_seed);

            for (int i=0; i<size*size; ++i)
                for (int j=0; j<dim; ++j)
                    data[i][j] = 0.7*wave.band[rand()%wave.nr_grains][j];
        }

        generation = 0;
    }


    // ---------- som algorithms --------------

    // find best matching entry for data
    // dat must point to 'dim' floats
    int best_match(const float* dat)
    {
        float md = 10000000000.0;
        int index = 0;
        for (int i=0; i<size*size; ++i)
        {
            // difference to each cell
            float d = 0.0;
            for (int j=0; j<dim; ++j)
                d += fabs(data[i][j] - dat[j]);

            if (d<md) { md = d; index = i; }
        }

        return index;
    }

    // find best matching entry for data
    // avoids fields who's 'umap' value is equal to or above 'thresh'
    // 'dat' must point to 'dim' floats
    // (this is used to assign each cell a paritcular grain)
    int best_match_avoid(const float* dat, float thresh = 0.0)
    {
        float md = 10000000000.0;
        int index = 0;
        for (int i=0; i<size*size; ++i)
        if (umap[i]<thresh)
        {
            // difference to each cell
            float d = 0.0;
            for (int j=0; j<dim; ++j)
                d += fabs(data[i][j] - dat[j]);

            if (d<md) { md = d; index = i; }
        }

        return index;
    }

    // insert the data into the map
    // dat must point to 'dim' floats
    // 'radius' is in pixels/cells, 'alpha' = 0 (transparent) to 1 (fully opaque)
    void insert(const float* dat)
    {
        int index = best_match(dat);

        // adjust cell and neighbourhood
        for (int j=-radius; j<=radius; ++j)
        for (int i=-radius; i<=radius; ++i)
        {
            int x = i + (index % size);
            int y = j + (index / size);
            if (!do_wrap && (x<0 || y<0 || x>=size || y>=size)) continue;

            if (x<0) x += size; else if (x>=size) x -= size;
            if (y<0) y += size; else if (y>=size) y -= size;

            float r = sqrtf(i*i+j*j);
            if (r>=radius) continue;

            // pointer to cell
            float * pd = &data[y * size + x][0];
            // pointer to input data
            const float * ps = dat;
            // amount to adjust
            float amt = alpha * (1.f - r/radius);

            for (int k=0; k<dim; ++k, ++ps, ++pd)
                *pd += amt * (*ps - *pd);
        }

        ++generation;
    }


    // return distance between cell i1 and i2
    float get_distance(int i1, int i2)
    {
        float d = 0.0;
        float * p1 = &data[i1][0],
              * p2 = &data[i2][0];
        for (int i=0; i<dim; ++i, ++p1, ++p2)
            d += fabs(*p1 - *p2);

        return d / dim;
    }

    void clear_umap(float value = 0.0)
    {
        for (int i=0; i<size*size; ++i)
            umap[i] = value;
    }

    // calculates the distance to neighbours for each cell
    void make_umap()
    {
        clear_umap();

        float ma = 0.00001;

        if (!do_wrap)
        for (int j=1; j<size-1; ++j)
        for (int i=1; i<size-1; ++i)
        {
            int k = j*size + i;
            float d =
                  get_distance(k, (j-1)*size + i - 1) * 0.75
                + get_distance(k, (j-1)*size + i)
                + get_distance(k, (j-1)*size + i + 1) * 0.75
                + get_distance(k,     j*size + i - 1)
                + get_distance(k,     j*size + i + 1)
                + get_distance(k, (j+1)*size + i - 1) * 0.75
                + get_distance(k, (j+1)*size + i)
                + get_distance(k, (j+1)*size + i + 1) * 0.75;

            umap[k] = d;
            ma = std::max(ma, d);
        }
        else /* on do_wrap */
        for (int j=0; j<size; ++j)
        for (int i=0; i<size; ++i)
        {
            int k = j*size + i,
                j0 = (j==0)? size-1 : j-1,
                j1 = (j==size-1)? 0 : j+1,
                i0 = (i==0)? size-1 : i-1,
                i1 = (i==size-1)? 0 : i+1;

            float d =
                  get_distance(k, j0*size + i0) * 0.75
                + get_distance(k, j0*size + i)
                + get_distance(k, j0*size + i1) * 0.75
                + get_distance(k, j *size + i0)
                + get_distance(k, j *size + i1)
                + get_distance(k, j1*size + i0) * 0.75
                + get_distance(k, j1*size + i)
                + get_distance(k, j1*size + i1) * 0.75;

            umap[k] = d;
            ma = std::max(ma, d);
        }

        // normalize
        for (int i=0; i<size*size; ++i)
            umap[i] /= ma;
    }

    // find the best match for each grain and store the grain's position (in seconds)
    void make_imap()
    {
        // index-all mode. EVERY cell will get a grain index!
        if (do_index_all)
        {
            // each cell will be indexed with the best-matching band
            for (int i=0; i<size*size; ++i)
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
        clear_umap(-1.f);
        for (int i=0; i<wave->nr_grains; ++i)
        {
            int index = best_match_avoid(&wave->band[i][0]);

            if (umap[index]<0.f)
                umap[index] = (float)(i*wave->grain_size)/wave->info.samplerate;
        }
        // finally, we set the non-indexed cells to 0
        for (int i=0; i<size*size; ++i)
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
    // _______ PUBLIC MEMBER _________

    int size, dim, rand_seed;

    // configurables
    int radius;
    float alpha;
    int draw_mode, nth_band, generation;
    bool
        do_wrap,
        do_index_all;

    // the som
    std::vector<std::vector<float> > data;
    // multi-purpose space
    std::vector<float> umap;

    // reference to the processed wavefile
    Wave *wave;

    ColorScale color;
};

#endif // SOM_H_INCLUDED
