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
    @brief self-organizing map class

    @version 2012/07/11 started as part of reaktor_som
    @version 2013/12/18 tidied up and removed opengl stuff, made x/y separate
    @version 2013/12/21 started local search
    @version 2014/01/28 threw out Wave class and worked with Data instead
    @version 2014/02/02 started Backend integration

    copyright 2012, 2013, 2014 stefan.berke @ modular-audio-graphics.com
*/
#ifndef SOM_H_INCLUDED
#define SOM_H_INCLUDED

#include <cstdlib>
#include <ctime>
#include <string>
#include <vector>

#include "som_types.h"


namespace RSOM
{

class Backend;
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

    enum BackendType
    {
        CPU,
        CUDA
    };

    /** One sample.
        This represents the input data as well as running statistics.
        There will be a structure created for each input sample. */
    struct DataIndex
    {
        /** pointer to 'dim' Floats */
        const Float * data;

        /** index in map (-1 if not indexed yet) */
        Index index;

        /** simply the index in data vector */
        Index count;

        /** used freely */
        int user_id;
    };


    // ------------ ctor -----------------

    Som(BackendType backend_type = CPU);
    ~Som();

    // ------------- IO ------------------

    bool saveMap(const std::string& filename);
    bool loadMap(const std::string& filename);

    // --------- getter ------------------

    Index size() const { return size_; }
    Index sizex() const { return sizex_; }
    Index sizey() const { return sizey_; }
    Index dim() const { return dim_; }
    Index numSamples() const { return samples_.size(); }

    int rand_seed() const { return rand_seed_; }

    size_t generation() const { return generation_; }
    size_t num_failed_inserts() const { return num_failed_inserts_; }

    Float alpha() const { return alpha_; }
    Float radius() const { return radius_; }
    Float local_search_radius() const { return local_search_radius_; }

    bool do_non_duplicate() const { return do_non_duplicate_; }

    /** return a few lines of information */
    std::string info_str() const;

    // ----------- setter ------------

    void do_non_duplicate(bool vacant_only) { do_non_duplicate_ = vacant_only; }

    // ------------ debug ------------

    static void printMap(const Float * map, Index w, Index h, Index dim_offset = 0,
                         Float threshold=0.5,
                         Index screen_w = 80, Index screen_h = 20);

    // --------- map handling ------------

    /** Creates or resets the map dimension.
        @note rand_seed is here only for convenience and can be set later as well. */
    void create(Index sizex, Index sizey, Index dim, int rand_seed);

    /** Initializes the map according to current settings.
        For certain init strategies, make sure data was inserted before. */
    void initMap();

    // ---------- data handling ---------------

    /** Sets Data container.
        Previous indices will be cleared and createDataIndex() will
        be called for each object in @p data.
        Set to NULL, to disconnect from Data container and clear the indices.
        @note The handed-over container must stay valid during the
        lifetime of the map or until disconnected. */
    void setData(const Data * data);

    /** Creates a data entry for training.
        This function can be used to insert arbitrary data objects.
        @p dat is expected to point at Som::dim consecutive Floats
        which must not change or deallocate until clearData() is
        called or the Som class is destroyed.
        @note The validity of the returned pointer to the DataIndex structure
        may only be valid until createDataIndex() is called again. */
    DataIndex * createDataIndex(const Float * dat, int user_id);

    // ---------- som algorithms --------------

    /** Inserts a data sample into the map according to current settings.
        If @p sample_index == -1, the sample will be selected at random. */
    void insert(Index sample_index = -1);

    /** move the DataIndex to map[index]. Affects Data::index and imap[] */
    void moveData(DataIndex * data, Index index);

    // ---------- training parameters ---------

    void alpha(Float value) { alpha_ = value; }
    void radius(Float value) { radius_ = value; }
    void local_search_radius(Float value) { local_search_radius_ = value; }

    // ---- matching with arbitrary samples ---

    // --------- map access -------------------

    /** Copys the current map to threadsafe memory, accessed by getMap(). */
    bool          updateMap();
    /** Returns pointer to sizey * sizex * dim floats. */
          Float * getMap() { return &map_[0]; }
    const Float * getMap() const { return &map_[0]; }

    /** Copys the current imap to threadsafe memory, accessed by getIMap(). */
    bool          updateIMap();
    /** Returns pointer to sizey * sizex ints. */
          Index * getIMap() { return &imap_[0]; }
    const Index * getIMap() const { return &imap_[0]; }

    /** Calculates and copys the current neighbour distance map to
        threadsafe memory, accessed by getUMap(). */
    bool          updateUMap(bool do_normalize = false, Float factor = 1.0);
    /** Returns pointer to sizey * sizex floats. */
          Float * getUMap() { return &umap_[0]; }
    const Float * getUMap() const { return &umap_[0]; }

    // _________ PRIVATE AREA _________________
private:

    // ---------- data matching ---------------

    /** Returns the index of the best matching cell for the Data.
        If @p only_vacant is true, this will avoids cells that are
        already occupied by a sample index (in imap).
        Then, the function returns -1, if no entry could be found.
        If @p value is not NULL, it will contain the difference between
        the data sample and the map and the found index. */
    Index best_match_(DataIndex * data, bool only_vacant, Float * value);

    /** Same as best_match_() but only looks in the given window. */
    Index best_match_window_(DataIndex * data, bool only_vacant, Float * value,
                             Index x, Index y, Index w, Index h);

    // _______ PRIVATE MEMBER _________

    // ---- config ----

    Index
        sizex_, sizey_,
    /** sizex * sizey */
        size_,
    /** diagonal length, calculated by create() */
        size_diag_,
    /** number of Floats per cell */
        dim_;
    /** initial random seed */
    int rand_seed_;

    // --- stats ---

    Float
    /** running average closest distance set by insert() */
        stat_av_best_match_;

    /** Number of inserted samples. */
    size_t generation_,
    /** Number of calls to insert() that did not lead
        to an actual insert. */
        num_failed_inserts_;

    // --- configurables ---
    /** Data insert radius in cells, */
    Float  radius_;
    /** Data insert tranparency (transparent) to 1 (fully opaque). */
    Float  alpha_;
    /** Search radius in cells. */
    Float  local_search_radius_;

    bool
        do_wrap_,
    /** no data sample can be on top of a previous other match */
        do_non_duplicate_;

    /** type of computation backend */
    BackendType backend_type_;
    Backend * backend_;

    /** representation of input samples */
    std::vector<DataIndex> samples_;
    /** the self-organizing map [sizey*sizex][dim] */
    std::vector<Float> map_;
    /** neighbour differences */
    std::vector<Float> umap_;
    /** data indices for each cell */
    std::vector<Index> imap_;

    /** reference to the processed data */
    const Data * data_container_;

};

} // namespace RSOM

#endif // SOM_H_INCLUDED
