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
    nr_grains = info.frames / grain_size;

    // get memory

    SOM_DEBUG("Wave::set:: resizing array");

    band.resize(nr_grains);
    for (size_t i=0; i<nr_grains; ++i)
        band[i].resize(nr_bands);

}


float Wave::get_bands(size_t xstart_, size_t xlen_, float amp)
{
    if (!nr_grains) return 0.000001;

    // get window
    size_t xstart = 0, xend = nr_grains;
    if (xstart_ || xlen_)
    {
        xstart = xstart_;
        xend = std::min(xend, xstart_ + xlen_);
    }


    float ma = 0.0001;

    // --- the infamous triple-loop of a discrete fourier transform ---

    // for each grain
    for (size_t i=xstart; i<xend; ++i)
    {
        // for each band
        for (size_t b=0; b!=nr_bands; ++b)
        {
            // frequency of each band
            float f = (float)b/std::max(1, (int)nr_bands-1);
            f = TAU * 0.5f * (min_freq + f * (max_freq - min_freq));
                    /// info.samplerate;

            // for each sample within the scanning window
            float sa=0.0, ca=0.0;
            for (size_t j=0; j<window_width; j++)
            {
                const float t = (float)j / info.samplerate;
                const int   si = i*grain_size + j;
                const float sam = (si<info.frames)? wave[si] : 0.f;

                sa += sinf(t*f) * sam;
                ca += cosf(t*f) * sam;
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

// returns the index (grain) that best fit's the given band-data 'dat'
// 'dat' needs to point at 'nr_bands' floats
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
