/** @file
    @brief sound file loader and spectral analyzer

    @author def.gsus- (berke@cymatrix.org)
    @version 2012/07/11

    copyright 2012 Stefan Berke,
    this program is coverd by the GNU General Public License
*/

#ifndef WAVEFILE_H_INCLUDED
#define WAVEFILE_H_INCLUDED

#include <cmath>
#include <vector>
#include <string>
#include <iostream>
//#include <SFML/OpenGL.hpp>
#include <sndfile.h>

#define TAU 6.283185307179586476925286766559

/** some arbitrary but unique color scale for values of 0.0 to 1.0 */
class ColorScale
{
    public:

    ColorScale()
    {
        // create color-scale
        color_map.resize(256*3);
        for (int i=0; i<256; ++i)
        {
            float t = (float)i/255;
            t *= 7.0;
            color_map[i*3  ] = 0.55-0.45*cosf(t);
            color_map[i*3+1] = 0.55-0.45*cosf(t*1.3);
            color_map[i*3+2] = 0.55-0.45*cosf(t*1.7);
        }
    }

    // set the colorscale for value 'f' (0-1)
    /*void set_color(const float f)
    {
        if (f<0) { glColor3f(1,0,0); return; }
        if (f>1) { glColor3f(1,1,1); return; }

        int i = (int)(f * 255.f + 0.5f);
        glColor3f(color_map[i*3], color_map[i*3+1], color_map[i*3+2]);
    }*/

    // some precalculated color scale (256*3)
    std::vector<float> color_map;
};




/** load a file and get the band data */
class Wave
{
    public:


    // load wave, return success
    bool open(const std::string& filename)
    {
        this->filename = filename;

        info.format = 0;
        sfile = sf_open( filename.c_str(), SFM_READ, &info);

        if (!sfile) return false;

        // get sample data
        std::vector<float> temp(info.channels * info.frames);
        sf_count_t r = sf_readf_float(sfile, &temp[0], info.frames);

        // check result
        if (r != info.frames)
        {
            sf_close(sfile);
            return false;
        }

        float ma = 0.0001;
        // convert to mono whatever it is
        wave.resize(info.frames);
        for (int i=0; i<info.frames; ++i)
        {
            float sum = 0.0;
            for (int j=0; j<info.channels; ++j)
                sum += temp[i*info.channels + j];
            wave[i] = sum / info.channels;
            ma = std::max(ma, wave[i]);
        }

        // and normalize
        for (int i=0; i<info.frames; ++i)
            wave[i] /= ma;

        // got data, can throw away file
        sf_close(sfile);

        length_in_secs = (float)info.frames / info.samplerate;
        return true;
    }

    // set some spectral parameters
    // call this AFTER opening and BEFORE get_bands()
    void set(int nr_bands, float min_freq, float max_freq, int grain_size, int window_width)
    {
        this->nr_bands = nr_bands;
        this->min_freq = min_freq;
        this->max_freq = max_freq;
        this->grain_size = grain_size;
        this->window_width = window_width;
        nr_grains = info.frames / grain_size;
    }


    void get_bands()
    {
        band.resize(nr_grains);
        for (int i=0; i<nr_grains; ++i)
            band[i].resize(nr_bands);

        float ma = 0.0001;

        // --- the infamous triple-loop of a discrete fourier transform ---

        // for each grain
        for (int i=0; i<nr_grains; ++i)
        {
            // for each band
            for (int b=0; b!=nr_bands; ++b)
            {
                // frequency of each band
                float f = (float)b/(nr_bands-1);
                f = TAU * 0.5f * info.samplerate * ((1.f-f) * min_freq + f * max_freq );

                float sa=0.0, ca=0.0;
                // for each sample within the scanning window
                for (int j=0; j<window_width; j++)
                {
                    float t = (float)j / info.samplerate;
                    int si = i*grain_size + j;
                    float sam = (si<info.frames)? wave[si] : 0.f;

                    sa += sinf(t*f) * sam;
                    ca += cosf(t*f) * sam;
                }

                // the last bit raises the level of high bands
                band[i][b] = sqrtf(sa*sa + ca*ca);// * (1.f + 20.0f * (float)b/nr_bands);
                ma = std::max(ma, band[i][b]);
            }

            std::cout << i << "/" << nr_grains << "          \r";
            std::cout.flush();
        }

        // normalize as a whole and shape logarithmically
        for (int i=0; i<nr_grains; ++i)
            for (int j=0; j<nr_bands; ++j)
                band[i][j] = pow(band[i][j]/ma, 0.5);

        std::cout << "                    \r";
    }

    // returns the index (grain) that best fit's the given band-data 'dat'
    // 'dat' needs to point at 'nr_bands' floats
    int get_best_match(const float* dat)
    {
        float mi = 10000000.0;
        int index = 0;
        for (int i=0; i<nr_grains; ++i)
        {
            float   d = 0.0,
                    *pb = &band[i][0];
            const float *pd = dat;

            for (int j=0; j<nr_bands; ++j, ++pb, ++pd)
                d += fabs(*pd - *pb);

            if (d<mi) { mi = d; index = i; }
        }
        return index;
    }

/*
    // draws the band data via opengl imidiate calls
    void draw_band_data(int xoff, int yoff, int width, int height)
    {
        // size of one data-point
        float   xs = (float)width / nr_grains,
                ys = (float)(height-10) / nr_bands;

        // draw small stripe of color scale below
        for (int i=0; i<width; ++i)
        {
            color.set_color((float)i/(width-1));
            glBegin(GL_LINES);
                glVertex2i(i,height-10);
                glVertex2i(i,height);
            glEnd();
        }

        for (int j=0; j<nr_bands; ++j)
        for (int i=0; i<nr_grains; ++i)
        {
            float f = band[i][nr_bands-1-j];
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
    // _________ PUBLIC MEMBER __________

    // settings
    std::string filename;
    int nr_bands, grain_size, nr_grains, window_width;
    float min_freq, max_freq;

    // wave data
    std::vector<float> wave;
    // band data
    std::vector<std::vector<float> > band;

    SNDFILE *sfile;
    // libsndfile info struct
    SF_INFO info;

    // length in seconds
    float length_in_secs;

    ColorScale color;
};

#endif // WAVEFILE_H_INCLUDED
