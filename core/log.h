#ifndef LOG_H
#define LOG_H

#include <iostream>

#define SOM_DEBUG(stream_arg__) \
    std::cerr << stream_arg__ << "\n"

/** noisy debug */
#if (0)
#   define SOM_DEBUGN(stream_arg__) SOM_DEBUG(stream_arg__)
#else
#   define SOM_DEBUGN(stream_arg__)
#endif

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
