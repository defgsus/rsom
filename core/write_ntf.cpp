#include "write_ntf.h"

// no multi-thread saving!!
static FILE* fileh;


// some shortcuts
inline void sfloat(const float val)
{
    fwrite(&val, sizeof(float), 1, fileh);
}

inline void sint(const int32_t val)
{
    fwrite(&val, sizeof(int32_t), 1, fileh);
}


bool save_ntf(const std::string& filename, const Som& som)
{
    fileh = fopen(filename.c_str(), "wb");
    if (!fileh) return false;

    // be sure that this value fits with your platform!
    sint(0);    // INTEL (little endian) = 0, MOTOROLA = 1
	sint(0);    // Reaktor 3.0 = 0
    sint(1);    // Undefined = 0, Float32Bits = 1

    sint(som.size);            // X size (horizontal)
    sint(som.size);            // Y size (vertical)

	sfloat(0.0f);                       // Min - Value Properties
	sfloat(som.wave->length_in_secs);   // Max - Value Properties
	sfloat(0.01f);                      // Stepsize - Value Properties
    sfloat(0.0f);                       // Default - Value Properties
    sint(0);                            // Display Format - Value Properties
                                        // 0 = Numeric, 1 = Midi Note, 2 = %

    // unfortunately, these don't seem to be used by reaktor
	sint(0x000000);     // DefaultValueColor
    sint(0x004000);     // MinValueColor
    sint(0x80ffff);     // MaxValueColor

    sint(0);          // X-Units  0 = Index, 1 = [0...1], 2 = milliseconds, 3 = tempo ticks
    // the following values don't mean much for
    // the kind of table we produce. i just chose some arbitrary values.
	sfloat(44100.f);  // float  X-SamplesPerSecond
    sfloat(120.f);    // float  X-BPM
    sfloat(1.f);      // float  X-SamplesPerTick
    sint(1);          // int    X-TicksPerBeat
    sint(1);          // int    X-BeatsPerBar

	sfloat(0.f);      // float  X-Offset
    sfloat(1.f);      // float  X-CustomRange
    sfloat(1.f);      // float  X-CustomRatio

    sint(0);          // int    Y-Units
	sfloat(44100.f);  // float  Y-SamplesPerSecond
    sfloat(120.f);    // float  Y-BPM
    sfloat(1.f);      // float  Y-SamplesPerTick
    sint(1);          // int    Y-TicksPerBeat
    sint(1);          // int    Y-BeatsPerBar

	sfloat(0.f);      // float  Y-Offset
    sfloat(1.f);      // float  Y-CustomRange
    sfloat(1.f);      // float  Y-CustomRatio

    // write the table data
    for (size_t j=0; j<som.size; ++j)
        for (size_t i=0; i<som.size; ++i)
            sfloat(som.umap[j*som.size+i]);

    fclose(fileh);
    return true;
}
