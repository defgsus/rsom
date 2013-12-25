#ifndef PROPERTIES_H
#define PROPERTIES_H

#include "property.h"

#include <vector>
#include <memory>

class Properties
{
public:
    Properties();

    void add(Property * p);

    std::vector<Property*> property;

private:
    std::vector<std::shared_ptr<Property>> auto_remove_;
};

#endif // PROPERTIES_H
