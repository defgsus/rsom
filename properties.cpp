#include "properties.h"

Properties::Properties()
{
}


void Properties::add(Property *p)
{
    auto_remove_.push_back(std::shared_ptr<Property>(p));

    property.push_back(p);
}
