/*  This is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License
    as published by the Free Software Foundation; either
    version 3.0 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this software; if not, write to the Free Software
    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
*/
#include "property.h"

#include "core/log.h"

#include <QLabel>
#include <QCheckBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QListWidget>
#include <QLayout>
#include <QSizePolicy>

#define SOM_CALLBACK(cb_name__) \
{ \
    SOM_DEBUGN(0, "Project:: callback " #cb_name__ ); \
    if (cb_name__##_) cb_name__##_(); \
}

Property::Property(const QString& id, const QString& name, const QString& help)
    :   type    (UNKNOWN),
        id      (id),
        name    (name),
        help    (help),
        dim     (0),
        active_ (true),
        ignore_value_changed_once_(false)
{
    SOM_DEBUGN(0, "Property::Property(" << id.toStdString() << ", " << name.toStdString() << ")");
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


void Property::init(bool v0)
{
    type = BOOL;
    dim = 1;
    v_bool.resize(dim);
    v_bool[0] = v0;
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

void Property::init(int min_val, int max_val, int v0, int v1)
{
    type = INT;
    min_int = min_val;
    max_int = max_val;
    dim = 2;
    v_int.resize(dim);
    v_int[0] = v0;
    v_int[1] = v1;
}

void Property::init(float min_val, float max_val, float v0)
{
    type = FLOAT;
    min_float = min_val;
    max_float = max_val;
    dim = 1;
    v_float.resize(dim);
    v_float[0] = v0;
}

void Property::init(float min_val, float max_val, float v0, float v1)
{
    type = FLOAT;
    min_float = min_val;
    max_float = max_val;
    dim = 2;
    v_float.resize(dim);
    v_float[0] = v0;
    v_float[1] = v1;
}

void Property::init(const std::vector<int>& item_values,
                    const QStringList& item_ids,
                    const QStringList& item_names, int value)
{
    type = SELECT;
    this->item_ids = item_ids;
    this->item_names = item_names;
    this->item_values = item_values;
    dim = 1;
    v_int.resize(dim);
    v_int[0] = value;
}



void Property::setMinMax(int new_min_value, int new_max_value)
{
    if (type != INT) return;

    min_int = new_min_value;
    max_int = new_max_value;

    for (auto i = widgets_.begin(); i!=widgets_.end(); ++i)
    if (auto spin = dynamic_cast<QSpinBox*>(*i))
    {
        spin->setMinimum(min_int);
        spin->setMaximum(max_int);
    }
}

void Property::setMinMax(float new_min_value, float new_max_value)
{
    if (type != FLOAT) return;
    min_float = new_min_value;
    max_float = new_max_value;

    for (auto i = widgets_.begin(); i!=widgets_.end(); ++i)
    if (auto spin = dynamic_cast<QDoubleSpinBox*>(*i))
    {
        spin->setMinimum(min_float);
        spin->setMaximum(max_float);
    }
}




// ---------------------------- widgets -----------------------------

void Property::onValueChanged_()
{
    SOM_DEBUGN(0, "Property::onValueChanged_()");

    if (ignore_value_changed_once_)
    {
        ignore_value_changed_once_ = false;
        SOM_DEBUGN(0, "Property::onValueChanged_:: not sending");

        return;
    }

    SOM_CALLBACK(cb_value_changed);
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
            l1->addStretch(2);
        } break;

        case H_WIDGET_LABEL:
        {
            auto l1 = new QHBoxLayout(0);
            addLayout_(l0, l1);
            for (size_t i=0; i<dim; ++i) getWidget_(parent, l1, i);
            getLabel_(parent, l1);
            //l1->addStretch(2);
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
    QWidget * widget = 0;

    switch (type)
    {
        case UNKNOWN: return 0;

        case BOOL:
        {
            auto cb = new QCheckBox(parent);
            l0->addWidget(cb);
            cb->setChecked(v_bool[i]);

            // get change event
#ifdef WTF
            parent->connect(cb, &QCheckBox::stateChanged, [=]()
            {
                v_bool[i] = cb->isChecked();
                onValueChanged_();
            });
#endif
            widgets_.push_back(cb);
            widget = cb;
            break;
        }

        case INT:
        {
            auto spin = new QSpinBox(parent);
            l0->addWidget(spin);
            spin->setMinimum(min_int);
            spin->setMaximum(max_int);
            spin->setValue(v_int[i]);

            // get change event
#ifdef WTF
            parent->connect(spin, static_cast<void(QSpinBox::*)(int)>(&QSpinBox::valueChanged), [=]()
            {
                v_int[i] = spin->value();
                onValueChanged_();
            });
#endif
            widgets_.push_back(spin);
            widget = spin;
            break;
        }

        case FLOAT:
        {
            auto spin = new QDoubleSpinBox(parent);
            l0->addWidget(spin);
            spin->setDecimals(4);
            spin->setMinimum(min_float);
            spin->setMaximum(max_float);
            spin->setValue(v_float[i]);

            // get change event
#ifdef WTF
            parent->connect(spin, static_cast<void(QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), [=]()
            {
                v_float[i] = spin->value();
                onValueChanged_();
            });
#endif
            widgets_.push_back(spin);
            widget = spin;
            break;
        }

        case SELECT:
        {
            // create list widget
            auto list = new QListWidget(parent);
            l0->addWidget(list);

            // add items
            for (size_t k=0; k<item_values.size(); ++k)
            {
                list->addItem(item_names[k]);
                // select the one
                if (v_int[i] == item_values[k])
                    list->setCurrentRow(k);
            }
            /// @todo nicer list height deduction
            list->setFixedHeight(100);
            list->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);

            // get change event
#ifdef WTF
            parent->connect(list, &QListWidget::currentItemChanged, [=](QListWidgetItem*, QListWidgetItem*)
            {
                if (list->currentRow() >= 0
                        && list->currentRow() < (int)item_values.size())
                {
                    v_int[i] = item_values[list->currentRow()];
                    onValueChanged_();
                }
                else SOM_ERROR("unknown item index " << list->currentRow()
                               << " for Property '" << id.toStdString() << "'");
            });
#endif
            widget = list;
            break;
        }

    }

    // can't be actually
    if (!widget) return 0;

    widget->setToolTip(help);

    // get destroy event
#ifdef WTF
    parent->connect(widget, &QWidget::destroyed, [=](QObject * obj)
    {
        Q_UNUSED(obj);
        SOM_DEBUGN(0, "Property widget '" << id.toStdString() << "'destroyed " << obj);
        disconnectWidget();
    } );
#endif
    // activity
    widget->setEnabled(active_);

    return widget;
}

void Property::disconnectWidget()
{
    SOM_DEBUGN(0, "Property::disconnectWidget()");

    widgets_.clear();
    cb_value_changed_ = 0;
}

void Property::updateWidget(bool do_callback)
{
    SOM_DEBUGN(0, "Property::updateWidget(" << do_callback << ")");

    if (widgets_.empty()) return;

    for (size_t i=0; i<dim; ++i)
    {
        // prepare to not send the cb_value_changed_
        if (!do_callback)
            ignore_value_changed_once_ = true;

        switch (type)
        {
            case UNKNOWN: break;

            case BOOL:
                static_cast<QCheckBox*>(widgets_[i])->setChecked(v_bool[i]);
            break;

            case INT:
                static_cast<QSpinBox*>(widgets_[i])->setValue(v_int[i]);
            break;

            case FLOAT:
                static_cast<QDoubleSpinBox*>(widgets_[i])->setValue(v_float[i]);
            break;

            case SELECT:
            {
                auto list = static_cast<QListWidget*>(widgets_[i]);
                // find the row for the value
                for (size_t k=0; k<item_values.size(); ++k)
                if (item_values[k] == v_int[i])
                {
                    list->setCurrentRow(k);
                    break;
                }
            } break;
        }
    }
}

void Property::setActive(bool active)
{
    active_ = active;
    for (auto i = widgets_.begin(); i!=widgets_.end(); ++i)
    {
        (*i)->setEnabled(active);
    }
}
