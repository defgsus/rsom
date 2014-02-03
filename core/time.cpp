#include "time.h"
#include <time.h>

double sysTime()
{
    timespec cls;
    clock_gettime(CLOCK_MONOTONIC, &cls);
    // second + nanoseconds
    return cls.tv_sec - 193000 +
            0.000000001 * cls.tv_nsec;
}
