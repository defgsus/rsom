#include "property.h"

#include "core/log.h"

#include <QLabel>
#include <QCheckBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QLayout>

Property::Property(const QString& id, const QString& name, const QString& help)
    :   type    (UNKNOWN),
        id      (id),
        name    (name),
        help    (help),
        dim     (0)
{
    SOM_DEBUG("Property::Property(" << id.toStdString() << ", " << name.toStdString() << ")");
}

const char * Property::typeName[] =
{ "bool", "int", "float", "select", "string" };

int Property::name2int(const QString& n)
{
    for (int i=0; i<numTypes; ++i)
        if (!n.compare(typeName[i]))
            return i;
    return -1;
}


void Property::init(float min_val, float max_val, float value)
{
    type = FLOAT;
    min_float = min_val;
    max_float = max_val;
    dim = 1;
    v_float.resize(dim);
    v_float[0] = value;
}

void Property::init(int min_val, int max_val, int value)
{
    type = INT;
    min_int = min_val;
    max_int = max_val;
    dim = 1;
    v_int.resize(dim);
    v_int[0] = value;
}



// ---------------------------- widgets -----------------------------

void Property::onValueChanged_()
{
    SOM_DEBUG("Property::onValueChanged_()");

    if (cb_value_changed_) cb_value_changed_();
}

void Property::createWidget(QWidget * parent, QLayout * l0, LayoutType ltype)
{
    if (type == UNKNOWN)
    {
        SOM_ERROR("Property::createWidget() for unknown type requested");
        return;
    }

    switch (ltype)
    {
        case LABEL_WIDGET:
        {
            getLabel_(parent, l0);
            for (size_t i=0; i<dim; ++i) getWidget_(parent, l0, i);
        } break;

        case WIDGET_LABEL:
        {
            for (size_t i=0; i<dim; ++i) getWidget_(parent, l0, i);
            getLabel_(parent, l0);
        } break;

        case V_LABEL_WIDGET:
        {
            auto l1 = new QVBoxLayout(0);
            addLayout_(l0, l1);
            getLabel_(parent, l1);
            for (size_t i=0; i<dim; ++i) getWidget_(parent, l1, i);
        } break;

        case H_WIDGET_LABEL:
        {
            auto l1 = new QHBoxLayout(0);
            addLayout_(l0, l1);
            for (size_t i=0; i<dim; ++i) getWidget_(parent, l1, i);
            getLabel_(parent, l1);
        } break;
    }


}

void Property::addLayout_(QLayout * parent, QLayout * child)
{
    if (auto vbox = dynamic_cast<QVBoxLayout*>(parent))
        vbox->addLayout(child);
    else
    if (auto hbox = dynamic_cast<QHBoxLayout*>(parent))
        hbox->addLayout(child);
    else
        SOM_ERROR("Property::addLayout_:: unhandled layout class");
}

QWidget * Property::getLabel_(QWidget * parent, QLayout * l0)
{
    auto label = new QLabel(parent);
    label->setText(name);
    l0->addWidget(label);
    return label;
}

QWidget * Property::getWidget_(QWidget * parent, QLayout * l0, size_t i)
{
    switch (type)
    {
        case UNKNOWN: return 0;

        case BOOL:
        {
            auto cb = new QCheckBox(parent);
            l0->addWidget(cb);
            cb->setChecked(v_bool[i]);

            // get change event
            parent->connect(cb, &QCheckBox::stateChanged, [=]()
            {
                v_bool[i] = cb->isChecked();
                onValueChanged_();
            });
            // get destroy event
            parent->connect(cb, &QWidget::destroyed, [=](QObject * obj) { SOM_DEBUG("---- destroyed " << obj); });
            widgets_.push_back(cb);
            return cb;
        }

        case INT:
        {
            auto spin = new QSpinBox(parent);
            l0->addWidget(spin);
            spin->setMinimum(min_float);
            spin->setMaximum(max_float);
            spin->setValue(v_int[i]);

            // get change event
            parent->connect(spin, static_cast<void(QSpinBox::*)(int)>(&QSpinBox::valueChanged), [=]()
            {
                v_int[i] = spin->value();
                onValueChanged_();
            });

            widgets_.push_back(spin);
            return spin;
        }

        case FLOAT:
        {
            auto spin = new QDoubleSpinBox(parent);
            l0->addWidget(spin);
            spin->setDecimals(4);
            spin->setMinimum(min_float);
            spin->setMaximum(max_float);
            spin->setValue(v_float[i]);

            widgets_.push_back(spin);
            return spin;
        }

    }

    return 0; /* compiler shutup */
}

void Property::disconnectWidget()
{
    widgets_.clear();
}

void Property::updateWidget()
{
    if (widgets_.empty()) return;

    for (size_t i=0; i<dim; ++i)
    {
        switch (type)
        {
            case BOOL:
                static_cast<QCheckBox*>(widgets_[i])->setChecked(v_int[i]);
            break;

            case INT:
                static_cast<QSpinBox*>(widgets_[i])->setValue(v_int[i]);
            break;

            case FLOAT:
                static_cast<QDoubleSpinBox*>(widgets_[i])->setValue(v_float[i]);
            break;
        }
    }
}

void Property::setActive(bool active)
{
    for (auto i = widgets_.begin(); i!=widgets_.end(); ++i)
    {
        (*i)->setEnabled(active);
    }
}
