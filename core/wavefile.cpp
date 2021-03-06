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
#include "wavefile.h"

#include "log.h"

Wave::Wave()
    :   ok_(false)
{
    SOM_DEBUG("Wave::Wave()");
}



bool Wave::open(const std::string& filename)
{
    SOM_DEBUG("Wave::open(" << filename << ")");

    ok_ = false;
    this->filename = filename;

    info.format = 0;
    sfile = sf_open( filename.c_str(), SFM_READ, &info);

    if (!sfile)
    {
        SOM_ERROR("Wave::open::sf_open failed");
        return false;
    }

    // get sample data
    std::vector<float> temp(info.channels * info.frames);
    sf_count_t r = sf_readf_float(sfile, &temp[0], info.frames);

    // check result
    if (r != info.frames)
    {
        SOM_ERROR("Wave::open::sf_open could not read completely");
        sf_close(sfile);
        return false;
    }

    SOM_DEBUG("Wave::open:: converting to mono");

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

    SOM_DEBUG("Wave::open:: normalizing");

    // and normalize
    for (int i=0; i<info.frames; ++i)
        wave[i] /= ma;

    // got data, can throw away file
    sf_close(sfile);
    sfile = 0;

    length_in_secs = (float)info.frames / info.samplerate;

    SOM_LOG("opened " << filename
            << "\nlength        " << length_in_secs << " sec, " << wave.size() << " samples"
            << "\nmax amplitude " << ma
            );

    return ok_ = true;
}

void Wave::update()
{
    SOM_DEBUG("Wave::update()");

    set(nr_bands, min_freq, max_freq, grain_size, window_width);
}

void Wave::set(size_t nr_bands, float min_freq, float max_freq, size_t grain_size, size_t window_width)
{
    SOM_DEBUG("Wave::set(" << nr_bands << ", " << min_freq << ", " << max_freq << ", " << grain_size << ", " << window_width << ")" );

    this->nr_bands = nr_bands;
    this->min_freq = min_freq;
    this->max_freq = max_freq;
    this->grain_size = grain_size;
    this->window_width = window_width;
    if (!ok_)
    {
        nr_grains = 0;
        return;
    }
    nr_grains = info.frames / grain_size;

    // get memory

    SOM_DEBUG("Wave::set:: resizing arrays");

    band.resize(nr_grains);
    for (size_t i=0; i<nr_grains; ++i)
    {
        band[i].resize(nr_bands);
        for (auto j=band[i].begin(); j!=band[i].end(); ++j)
            *j = 0.f;
    }

    // window function
    table_window.resize(window_width);
    for (size_t i=0; i<window_width; ++i)
    {
        const float t = (float)i / window_width;
        table_window[i] = sinf(t * 3.141592654);
    }
    // get sin/cos table

    table_sin.resize(window_width * nr_bands);
    table_cos.resize(window_width * nr_bands);
    auto psin = table_sin.begin(),
         pcos = table_cos.begin();
    for (size_t b=0; b!=nr_bands; ++b)
    {
        // frequency of each band
        float f = (float)b/std::max(1, (int)nr_bands-1);
        f = TAU * 0.5f * (min_freq + f * (max_freq - min_freq));

        // for each sample within the scanning window
        for (size_t j=0; j<window_width; j++)
        {
            const float t = (float)j / info.samplerate;
            *psin++ = sinf(t*f);
            *pcos++ = cosf(t*f);
        }
    }

}


float Wave::get_bands(size_t xstart_, size_t xlen_, float amp)
{
    float ma = 0.0001;

    // safety checks

    if (!nr_grains) return ma;

    if (table_sin.size() != nr_bands * window_width
     || table_cos.size() != nr_bands * window_width )
    {
        SOM_ERROR("Wave::get_bands:: sin/cos table not initialized");
        return ma;
    }

    // get window of samples to process
    size_t xstart = 0, xend = nr_grains;
    if (xstart_ || xlen_)
    {
        xstart = xstart_;
        xend = std::min(xend, xstart_ + xlen_);
    }


    // --- the infamous triple-loop of a discrete fourier transform ---

    /* No, this is not very fast. But accurate and completely user adjustable.
     * E.g. you can choose the window width, number of bands and frequency range
     * individually.
     */

    const int half_window = window_width / 2;

    // for each grain
    for (size_t i=xstart; i<xend; ++i)
    {
        auto psin = table_sin.cbegin(),
             pcos = table_cos.cbegin();

        // for each band
        for (size_t b=0; b!=nr_bands; ++b)
        {
            // frequency of each band
//            float f = (float)b/std::max(1, (int)nr_bands-1);
//            f = TAU * 0.5f * (min_freq + f * (max_freq - min_freq));

            // for each sample within the scanning window
            float sa=0.0, ca=0.0;
            for (size_t j=0; j<window_width; j++)
            {
//              const float t = (float)j / info.samplerate;
                const int   si = (int)(i*grain_size + j) - half_window;
                const float sam = (si>=0 && si<info.frames)?
                                wave[si] * table_window[j] : 0.f;

                sa += *psin++ * sam;
                ca += *pcos++ * sam;
            }

            // the last bit raises the level of high bands
            band[i][b] = amp * sqrtf(sa*sa + ca*ca);// * (1.f + 20.0f * (float)b/nr_bands);
            ma = std::max(ma, band[i][b]);
        }
    }

    return ma;
}

void Wave::normalize(float ma, float exp)
{
    SOM_DEBUG("Wave::normalize(" << ma << ", " << exp << ")");

    // normalize as a whole and shape
    for (size_t i=0; i<nr_grains; ++i)
        for (size_t j=0; j<nr_bands; ++j)
            band[i][j] = powf(band[i][j]/ma, exp);
}

void Wave::shape(float amp, float exp)
{
    SOM_DEBUG("Wave::shape(" << amp << ", " << exp << ")");

    for (size_t i=0; i<nr_grains; ++i)
        for (size_t j=0; j<nr_bands; ++j)
            band[i][j] = powf(std::max(0.f,std::min(1.f,
                band[i][j] * amp )), exp);
}


int Wave::get_best_match(const float* dat)
{
    float mi = 10000000.0;
    int index = 0;
    for (size_t i=0; i<nr_grains; ++i)
    {
        float   d = 0.0,
                *pb = &band[i][0];
        const float *pd = dat;

        for (size_t j=0; j<nr_bands; ++j, ++pb, ++pd)
            d += fabs(*pd - *pb);

        if (d<mi) { mi = d; index = i; }
    }
    return index;
}
