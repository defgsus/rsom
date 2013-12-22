#ifndef LOG_H
#define LOG_H

#include <iostream>
#include <thread>

#define SOM_DEBUG(stream_arg__) \
    { std::cerr << std::hex << std::this_thread::get_id() << ": " \
                << stream_arg__ << "\n"; }



/** SOM_DEBUGN(level, arg) - noisy debug with level */
#if (1)
#   define SOM_DEBUGN(level__, stream_arg__) \
        if (level__<=SOM_DEBUGN_LEVEL) SOM_DEBUG(stream_arg__)
#else
#   define SOM_DEBUGN(level__, stream_arg__) { }
#endif

#define SOM_DEBUGN_LEVEL 2


#define SOM_ERROR(stream_arg__) \
    SOM_DEBUG("*error* " << stream_arg__)

#define SOM_LOG(stream_arg__) \
    SOM_DEBUG("log: " << stream_arg__)

/*
#define SOM_LOG(stream_arg__) \
    Logger::singleInstance() << stream_arg__ << "\n";

class Logger
{
    public:

    Logger() : instance_(0) { }

    static Logger & singleInstance();

    template <class T>
    friend Logger& operator << (Logger& l, const T& arg)
    {
        l.stream_ << arg;
        return l;
    }

private:
    Logger * instance_;

    std::streambuf stream_;
};



Logger & Logger::singleInstance()
{
    if (!instance_) instance_ = new Logger();
    return *instance_;
}
*/


#endif // LOG_H
