
#include <thrust/device_vector.h>
#include "som_types.h"

namespace RSOM
{

struct ThrustInterface
{
    thrust::device_vector<float>
        map,
        dmap,
        vec,
        diff;

    Index size, dim;
};


} // namespace RSOM
