/** @file
    @brief settings and task manager

    @version 2013/12/18 init

    copyright 2013 stefan.berke @ modular-audio-graphics.com

    This program is coverd by the GNU General Public License
*/
#ifndef PROJECT_H
#define PROJECT_H

#include <string>
#include <thread>

#include "wavefile.h"
#include "som.h"


/** Settings and task manager.
    Each Project contains all settings and data
    and runs the analyzis in a separate task. */
class Project
{
public:

    Project();
    ~Project();

    // --- get properties ---

    const std::string& filename()       const { return filename_; }
    const std::string& name()           const { return name_; }
    const std::string& wavename()       const { return wave_->filename; }

    const Wave&        wave()           const { return *wave_; }
    const Som&         som()            const { return *som_; }

    size_t             num_bands()      const { return wave_->nr_bands; }
    size_t             num_grains()     const { return wave_->nr_grains; }
    size_t             grain_size()     const { return wave_->grain_size; }
    size_t             window_width()   const { return wave_->window_width; }
    float              band_amp()       const { return band_amp_; }
    float              band_exponent()  const { return band_exp_; }

    size_t             som_sizex()      const { return som_sizex_; }
    size_t             som_sizey()      const { return som_sizey_; }
    size_t             som_seed()       const { return som_seed_; }

    float              som_alpha()      const { return som_alpha_; }
    float              som_radius()     const { return som_radius_; }

    // --- access ---------

    /* all of these are locked and sanity checked as individually required. */

    /** set name of project */
    void name(const std::string& project_name);

    /** set spectrum parameters, restarts ALL */
    void set(size_t nr_bands, float min_freq, float max_freq, size_t grain_size, size_t window_width,
             float band_amp_, float band_exp_);

    /** set som parameters, restarts the SOM */
    void set_som(size_t sizex, size_t sizey, int rand_seed);

    void set_som_alpha(float alpha) { som_alpha_ = alpha; }
    void set_som_radius(float radius) { som_radius_ = radius; }

    // ------- IO ---------

    /** Try to load a sample. Discards all current calculations.
        After succesful return of this function, the data is initialized,
        but it will be calculated by the worker thread. */
    bool load_wave(const std::string& soundfile_name);

    // ----- callbacks ----

    /** install callback for when wavefile is loaded */
    void cb_wave_loaded(std::function<void()> func) { cb_wave_loaded_ = func; }

    /** install callback for when bands are partially or fully calculated */
    void cb_bands(std::function<void()> func) { cb_bands_ = func; }

    /** install callback for when the SOM has been allocated */
    void cb_som_ready(std::function<void()> func) { cb_som_ready_ = func; }

    /** install callback for when the SOM has made progress */
    void cb_som(std::function<void()> func) { cb_som_ = func; }

    // --------------

    // these are not really needed right now but handled internally

    /** start the thread, if not already running. return if running. */
    bool start_worker();

    /** stop worker thread, if running */
    void stop_worker();


    // _____ PRIV _______
private:

    void work_loop_();

    // -- properties --
    std::string
        filename_, name_;

    // BAND params
    float band_amp_, band_exp_;

    // SOM parameters
    size_t som_sizex_, som_sizey_, som_seed_;
    float som_alpha_, som_radius_;

    // -- data --

    Wave * wave_;
    Som * som_;

    // -- runtime --

    std::thread * thread_;
    std::mutex mutex_;

    volatile bool
        run_,
        restart_,
    /** a restart causes the wave to be re-analyzed */
        wave_changed_;

    std::function<void()>
        cb_wave_loaded_,
        cb_bands_,
        cb_som_ready_,
        cb_som_;
};

#endif // PROJECT_H
