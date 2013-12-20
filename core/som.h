/** @file
    @brief self-organizing map class for use with reaktor_som

    @author def.gsus- (berke@cymatrix.org)
    @version 2012/07/11
    @version 2013/12/18 tidied up and removed opengl stuff, made x/y separate

    copyright 2012, 2013 stefan.berke @ modular-audio-graphics.com

    This program is coverd by the GNU General Public License
*/
#ifndef SOM_H_INCLUDED
#define SOM_H_INCLUDED

#include <cstdlib>
#include <ctime>
#include <string>

#include "wavefile.h"

/** Kohonen map or self-organizing map.
    low-level class, public member acess

    use:
    create(), init(Wave&), insert() ..., calc_umap(), calc_imap(), ...
*/
class Som
{
    public:

    Som();

    /** create or reset the map */
    void create(int sizex, int sizey, int dim, int rand_seed);

    /** Sets Wave and fills map randomly.
        This will use the band data in 'wave' to initialize the map.
        To avoid a strong bias, the band data is taken apart and recombined randomly. */
    void init(Wave& wave);

    std::string info_str() const;

    // ---------- som algorithms --------------

    /** Finds best matching entry for data.
        'dat' must point to 'dim' floats */
    size_t best_match(const float* dat);

    /** Finds best matching entry for data.
        Avoids fields who's 'umap' value is equal to or above 'thresh'.
        'dat' must point to 'dim' floats.
        This function is used to assign each cell a paritcular grain. */
    size_t best_match_avoid(const float* dat, float thresh = 0.0);

    /** Inserts the data into the map,
        'dat' must point to 'dim' floats.
        'radius' is the ratio of half the largest sidelength [0,1],
        'alpha' = 0 (transparent) to 1 (fully opaque). */
    void insert(const float* dat);

    /** return distance between cell i1 and i2 */
    float get_distance(size_t i1, size_t i2);

    /** Sets umap to 'value' */
    void set_umap(float value = 0.0);

    /** Calculates the distance to neighbours for each cell */
    void calc_umap();

    /** find the best match for each grain and store the grain's position (in seconds) */
    void calc_imap();

    // _______ PUBLIC MEMBER _________

    size_t sizex, sizey,
    /** sizex * sizey */
        size,
    /** number of floats per cell */
        dim;
    /** initial random seed */
    int rand_seed;

    // --- stats ---

    float
    /** running average closest distance set by insert() */
        stat_av_best_match;

    // --- configurables ---
    float  radius;
    float  alpha;
    size_t draw_mode,
           nth_band,
           generation;
    bool
        do_wrap,
        do_index_all;

    /** the self-organizing map [sizey*sizex][dim] */
    std::vector<std::vector<float> > data;
    /** neighbour relations, multi-purpose space */
    std::vector<float> umap;

    /** reference to the processed wavefile */
    Wave *wave;

};

#endif // SOM_H_INCLUDED
