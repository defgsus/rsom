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
/** @file
    @brief self-organizing map class for use with reaktor_som

    @author def.gsus- (berke@cymatrix.org)
    @version 2012/07/11
    @version 2013/12/18 tidied up and removed opengl stuff, made x/y separate
    @version 2012/12/21 started local search

    copyright 2012, 2013 stefan.berke @ modular-audio-graphics.com
*/
#ifndef SOM_H_INCLUDED
#define SOM_H_INCLUDED

#include <cstdlib>
#include <ctime>
#include <string>
#include <vector>

class Data;

/** Kohonen network or self-organizing map.
    low-level class, public member acess

    use:
    create(),
    setData() ...,
    initMap(),
    insert() ...,
    calc_umap(), calc_imap(), ...
*/
class Som
{
    public:

    // ------------- types ---------------

    typedef float Float;

    /** One sample.
        This represents the input data as well as
        running statistics. */
    struct DataIndex
    {
        /** pointer to 'dim' Floats */
        const Float * data;

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

    /** Creates a data entry for training.
        'dat' is expected to point at 'dim' consecutive Floats
        which must not change or deallocate until clearData() is
        called or the Som class is destroyed.
        @note The validity of the returned pointer to the DataIndex structure
        may only be valid until createDataIndex() is called again. */
    DataIndex * createDataIndex(const Float * dat, int user_id);

    /** Sets Data container.
        Previous indices will be cleared and createDataIndex() will
        be called for each object in @p data.
        Set to NULL, to disconnect from Data container and clear the indices.
        @note The handed-over container must stay valid during the
        lifetime of the map or until disconnected. */
    void setData(const Data * data);

    // ---------- som algorithms --------------

    /** Inserts a data sample into the map according to current settings. */
    void insert();

    /** move the Data to map[index]. Affects Data::index and imap[] */
    void moveData(DataIndex * data, size_t index);

    // ---------- data matching ---------------

    /** Returns the index of the best matching cell for the Data,
        according to current strategy. */
    size_t best_match(DataIndex * data);

    /** Returns the index of the best matching cell for the Data,
        according to current strategy. Avoids cells that already
        contain a data index (in imap).
        The function returns -1, if no entry could be found. */
    size_t best_match_avoid(DataIndex * data);

    /** Returns the distance/difference between 'data' and the map cell */
    Float get_distance(const DataIndex * data, size_t cell_index) const;

    /** Returns the distance/difference between cell i1 and i2 */
    Float get_distance(size_t i1, size_t i2) const;

    // ---------- info maps -------------------

    /** Sets whole umap to 'value' */
    void set_umap(Float value = 0.0);
    /** Sets whole imap to 'value' */
    void set_imap(int value = 0.0);

    /** Calculates the distance to neighbours for each cell */
    void calc_umap();

    /** find the best match for each grain and store the grain's position (in seconds) */
    void calc_imap();

    // ---- matching with arbitrary samples ---

    /** Finds best matching entry for data.
        'dat' must point to 'dim' Floats */
    size_t best_match(const Float* dat);

    /** Finds best matching entry for data.
        Avoids fields who's 'imap' value is equal to or above zero.
        'dat' must point to 'dim' Floats.
        This function is used to assign each cell a particular grain. */
    size_t best_match_avoid(const Float* dat);

    // _______ PUBLIC MEMBER _________

    size_t sizex, sizey,
    /** sizex * sizey */
        size,
    /** diagonal length, calculated by create() */
        size_diag,
    /** number of Floats per cell */
        dim;
    /** initial random seed */
    int rand_seed;

    // --- stats ---

    Float
    /** running average closest distance set by insert() */
        stat_av_best_match;

    // --- configurables ---
    /** Data insert radius in cells, */
    Float  radius;
    /** Data insert tranparency (transparent) to 1 (fully opaque). */
    Float  alpha;
    /** Search radius in cells. */
    Float  local_search_radius;
    /** Number of inserted samples. */
    size_t generation;

    bool
        do_wrap,
    /** no data sample can be on top of a previous other match */
        do_non_duplicate,
        /// @todo not fully clear
        do_index_all;

    /** representation of input samples */
    std::vector<DataIndex> data;
    /** the self-organizing map [sizey*sizex][dim] */
    std::vector<std::vector<Float> > map;
    /** neighbour relations, multi-purpose space */
    std::vector<Float> umap;
    /** data indices for each cell */
    std::vector<int> imap;

    /** reference to the processed data */
    const Data * dataContainer;

};

#endif // SOM_H_INCLUDED
