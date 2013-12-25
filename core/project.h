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
#include <future>

#include "wavefile.h"
#include "som.h"


/** Settings and task manager.
    A Project contains all settings and data
    and runs the analysis and training in separate threads.

    To work with this class, you need to respond to some callbacks.
    cb_bands_finished() is the callback from where you can start
    training with set_som().
*/
class Project
{
public:

    Project();
    ~Project();

    // --- get properties ---

    const std::string  info_str()       const;

    const std::string& filename()       const { return filename_; }
    const std::string& name()           const { return name_; }
    const std::string& wavename()       const { return wave_->filename; }

    const Wave&        wave()           const { return *wave_; }
    const Som&         som()            const { return *som_; }
          Som&         som()                  { return *som_; }

    bool               analyzing()      const { return run_wave_; }
    bool               running()        const { return run_som_; }

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
    float              som_search_radius()
                                        const { return som_search_radius_; }
    /** only true when the som is allocated. */
    bool               som_ready()      const { return som_ready_; }

    // --- access ---------

    /* all of these are locked and sanity checked as individually required. */

    /** set name of project */
    void name(const std::string& project_name);

    /** set spectrum parameters, restarts wave analysis.
        On return of call, the band data will be reallocated.
        if band_amp_ == 0, bands will be normalized.
        <p></p> */
    void set(size_t nr_bands, float min_freq, float max_freq, size_t grain_size, size_t window_width,
             float band_amp_ = 0.f, float band_exp_ = 1.f);

    /** set som parameters, restarts the SOM.
        On return of the call, the som will be initialized and the cb_som_ready callback
        will be send.
        Subsequently, startSomThread() and stopSomThread() can be used to start/stop training. */
    void set_som(size_t sizex, size_t sizey, int rand_seed);

    // -------- live training parameters ----------

    void set_som_alpha(float alpha) { som_alpha_ = alpha; }
    void set_som_radius(float radius) { som_radius_ = radius; }
    void set_som_search_radius(float radius) { som_search_radius_ = radius; }

    // ------- IO ---------

    /** Tries to load a sample.
        Discards all current calculations and restarts Wave thread.
        After succesful return of this function, the data is initialized
        and will be calculated by the worker thread. */
    bool load_wave(const std::string& soundfile_name);

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

    // these are not really part of the interface but handled internally

    /** throw threads at band data, calls cb_bands and cb_bands_finished */
    void startWaveThread();
    /** stop wave analysis thread(s), if running */
    void stopWaveThread();

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
        filename_, name_;

    // BAND params
    float band_amp_, band_exp_;

    // SOM parameters
    size_t som_sizex_, som_sizey_, som_seed_;
    float som_alpha_, som_radius_, som_search_radius_;

    // -- data --

    Wave * wave_;
    Som * som_;

    // -- runtime --

    std::thread
        * wave_thread_,
        * som_thread_;
    //std::mutex mutex_;

    volatile bool
    /** flag for stopping the analysis thread */
        run_wave_,
    /** flag for stopping the som worker thread */
        run_som_,

        som_ready_;

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
