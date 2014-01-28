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
    @brief settings and task manager

    @version 2013/12/18 init

    copyright 2013 stefan.berke @ modular-audio-graphics.com
*/
#ifndef PROJECT_H
#define PROJECT_H

#include <string>
#include <thread>
#include <future>

#include "wavefile.h"
#include "som.h"

class Data;

/** Settings and task manager.
    A Project contains all settings and data
    and runs the analysis and training in separate threads.

    To work with this class, you need to respond to some callbacks.
*/
class Project
{
public:

    // ---- ctor ------------

    Project();
    ~Project();

    // --- get properties ---

    const std::string  info_str()       const;

    // --- data access ---

    const Data&        data()           const { return *data_; }
          Data&        data()                 { return *data_; }

    // --- som access ---

    const Som&         som()            const { return *som_; }
          Som&         som()                  { return *som_; }

    bool               running()        const { return run_som_; }

    size_t             som_sizex()      const { return som_sizex_; }
    size_t             som_sizey()      const { return som_sizey_; }
    size_t             som_seed()       const { return som_seed_; }

    float              som_alpha()      const { return som_alpha_; }
    float              som_radius()     const { return som_radius_; }
    float              som_search_radius()
                                        const { return som_search_radius_; }
    /** only true when the som is allocated. */
    bool               som_ready()      const { return som_ready_; }

    // --- access ---------

    /* all of these are locked and sanity checked as individually required. */

    /** set name of project */
    void name(const std::string& project_name);


    /** set som parameters, restarts the SOM.
        On return of the call, the som will be initialized and the cb_som_ready callback
        will be send.
        Subsequently, startSomThread() and stopSomThread() can be used to start/stop training.
        @note Make sure you have the Data ready beforehand. */
    void set_som(size_t sizex, size_t sizey, int rand_seed);

    // -------- live training parameters ----------

    void set_som_alpha(float alpha) { som_alpha_ = alpha; }
    void set_som_radius(float radius) { som_radius_ = radius; }
    void set_som_search_radius(float radius) { som_search_radius_ = radius; }

    // ------- IO ---------

    // ----- callbacks ----

    /** install callback for when wavefile is loaded */
    void cb_wave_loaded(std::function<void()> func) { cb_wave_loaded_ = func; }

    /** install callback for when bands are partially or fully calculated.
        The last call of cb_bands will happen after all bands are calculated. */
    void cb_bands(std::function<void()> func) { cb_bands_ = func; }

    /** install callback for when bands are finished calculating */
    void cb_bands_finished(std::function<void()> func) { cb_bands_finished_ = func; }

    /** install callback for when the SOM has been (re-)allocated */
    void cb_som_ready(std::function<void()> func) { cb_som_ready_ = func; }

    /** install callback for when the SOM has made progress. */
    void cb_som(std::function<void()> func) { cb_som_ = func; }

    // ------ threading --------

    /** start the thread, if not already running. return if running. */
    bool startSomThread();

    /** stop worker thread, if running */
    void stopSomThread();


    // _____ PRIV _______
private:

    // see comment in .cpp file
    void work_loop_();

    // -- properties --
    std::string
    // name of project. not really used right now
        name_;

    // SOM parameters
    size_t som_sizex_, som_sizey_, som_seed_;
    float som_alpha_, som_radius_, som_search_radius_;

    // -- data --

    Data * data_;
    Som * som_;

    // -- runtime --

    std::thread
        * som_thread_;
    //std::mutex mutex_;

    volatile bool
    /** flag for stopping the som worker thread */
        run_som_,

        som_ready_;

    // runtime statistics

    int inserts_per_second_;
    size_t last_generation_;

    // -- callbacks --
    std::function<void()>
        cb_wave_loaded_,
        cb_bands_,
        cb_bands_finished_,
        cb_som_ready_,
        cb_som_;
};

#endif // PROJECT_H
