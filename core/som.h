/** @file
    @brief self-organizing map class for use with reaktor_som

    @author def.gsus- (berke@cymatrix.org)
    @version 2012/07/11
    @version 2013/12/18 tidied up and removed opengl stuff, made x/y separate
    @version 2012/12/21 starting local search

    copyright 2012, 2013 stefan.berke @ modular-audio-graphics.com

    This program is coverd by the GNU General Public License
*/
#ifndef SOM_H_INCLUDED
#define SOM_H_INCLUDED

#include <cstdlib>
#include <ctime>
#include <string>

#include "wavefile.h"

/** Kohonen network or self-organizing map.
    low-level class, public member acess

    use:
    create(),
    insertData() ...,
    initMap(),
    insert() ...,
    calc_umap(), calc_imap(), ...
*/
class Som
{
    public:

    // ------------- types ---------------

    /** One sample.
        This represents the input data as well as
        running statistics. */
    struct Data
    {
        /** pointer to 'dim' floats */
        const float * data;

        /** index in map (-1 if not indexed yet) */
        int index;

        /** simply the index in data vector */
        int count;

        /** used freely */
        int user_id;
    };


    // ------------ ctor -----------------

    Som();

    std::string info_str() const;

    // --------- map handling ------------

    /** Creates or resets the map dimension.
        @note rand_seed is here only for convenience and can be set later as well. */
    void create(int sizex, int sizey, int dim, int rand_seed);

    /** Initializes the map according to current settings.
        For certain init strategies, make sure data was inserted before. */
    void initMap();

    // ---------- data handling ---------------

    /** Removes all inserted data. */
    void clearData();

    /** Creates a data entry for training.
        'dat' is expected to point at 'dim' consecutive floats
        which must not change or deallocate until clearData() is
        called or the Som class is destroyed.
        @note The validity of the returned pointer to the Data structure
        may only be defined until insertData() is called again. */
    Data * insertData(const float * dat, int user_id = 0);

    /** Sets Wave and fills map randomly.
        * Convenience function for interoperating with Wave class *
        This will use the band data in 'wave' to initialize the map.
        To avoid a strong bias, the band data is taken apart and recombined randomly.
        */
    void insertWave(Wave& wave);

    // ---------- som algorithms --------------

    /** Inserts a data sample into the map according to current settings. */
    void insert();

    /** move the Data to map[index]. Affects Data::index and imap[] */
    void moveData(Data * data, size_t index);

    // ---------- data matching ---------------

    /** Returns the index of the best matching cell for the Data,
        according to current strategy. */
    size_t best_match(Data * data);

    /** Returns the distance/difference between 'data' and the map cell */
    float get_distance(const Data * data, size_t cell_index) const;

    /** Returns the distance/difference between cell i1 and i2 */
    float get_distance(size_t i1, size_t i2) const;

    // ---------- info maps -------------------

    /** Sets umap to 'value' */
    void set_umap(float value = 0.0);

    /** Calculates the distance to neighbours for each cell */
    void calc_umap();

    /** find the best match for each grain and store the grain's position (in seconds) */
    void calc_imap();

    // ---- matching with arbitrary samples ---

    /** Finds best matching entry for data.
        'dat' must point to 'dim' floats */
    size_t best_match(const float* dat);

    /** Finds best matching entry for data.
        Avoids fields who's 'umap' value is equal to or above 'thresh'.
        'dat' must point to 'dim' floats.
        This function is used to assign each cell a particular grain. */
    size_t best_match_avoid(const float* dat, float thresh = 0.0);

    // _______ PUBLIC MEMBER _________

    size_t sizex, sizey,
    /** sizex * sizey */
        size,
    /** diagonal length, calculated by create() */
        size_diag,
    /** number of floats per cell */
        dim;
    /** initial random seed */
    int rand_seed;

    // --- stats ---

    float
    /** running average closest distance set by insert() */
        stat_av_best_match;

    // --- configurables ---
    /** Data insert radius in cells, */
    float  radius;
    /** Data insert tranparency (transparent) to 1 (fully opaque). */
    float  alpha;
    /** Search radius in cells. */
    float  local_search_radius;
    /** Number of inserted samples. */
    size_t generation;

    bool
        do_wrap,
        /// @todo not fully clear
        do_index_all;

    /** representation of input samples */
    std::vector<Data> data;
    /** the self-organizing map [sizey*sizex][dim] */
    std::vector<std::vector<float> > map;
    /** neighbour relations, multi-purpose space */
    std::vector<float> umap;
    /** data indices for each cell */
    std::vector<int> imap;

    /** reference to the processed wavefile */
    Wave *wave;

};

#endif // SOM_H_INCLUDED
