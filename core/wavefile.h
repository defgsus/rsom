/** @file
    @brief sound file loader and spectral analyzer

    @version 2012/07/11 init
    @version 2013/12/18 tidied up

    copyright 2013 stefan.berke @ modular-audio-graphics.com

    This program is coverd by the GNU General Public License
*/

#ifndef WAVEFILE_H_INCLUDED
#define WAVEFILE_H_INCLUDED

#include <cmath>
#include <vector>
#include <string>
#include <sndfile.h>

#ifndef TAU
#   define TAU 6.283185307179586476925286766559
#endif


/** Load a file and get the band data.
    This is a low-level container, with linear data in public members.

    Use open(), set(), get_bands() and normalize() to calc the spectrum.

    uses libsndfile :)
*/
class Wave
{
    public:

    Wave();

    /** load wave, return success */
    bool open(const std::string& filename);

    /** open? */
    bool ok() const { return ok_; }

    /** set some spectral parameters
        call this AFTER opening and BEFORE get_bands() */
    void set(size_t nr_bands, float min_freq, float max_freq, size_t grain_size, size_t window_width);

    /** after loading a new file, call set() with same parameters as last call */
    void update();

    /** Calculate the whole spectrum, as set previously with set().
        Specify 'x' and 'num' to only calculated portions of the spectrum.
        The end (x+num) will be clamped to the end of the data.
        If both are 0, all is calculated.
        @returns
        Returns maximum found value needed for normalize.
    */
    float get_bands(size_t x = 0, size_t num = 0, float amp = 1.f);

    /** finalize data, previously gathered with get_bands() */
    void normalize(float max_value);

    /** shape and clamp band data. alternative to normalize() */
    void shape(float amplitude, float exponent);

    /** Returns the index (grain) that best fit's the given band-data 'dat'.
        'dat' needs to point at 'nr_bands' floats */
    int get_best_match(const float* dat);

    // -- callbacks ---

    // _________ PUBLIC MEMBER __________

    // settings
    std::string filename;
    size_t nr_bands, grain_size, nr_grains, window_width;
    float min_freq, max_freq;

    // wave data
    std::vector<float> wave;
    // band data
    std::vector<std::vector<float>> band;

    SNDFILE *sfile;
    // libsndfile info struct
    SF_INFO info;

    /** length in seconds */
    float length_in_secs;

private:
    bool ok_;
};

#endif // WAVEFILE_H_INCLUDED

