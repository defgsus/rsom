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
/** @file
    @brief Property / value handler with GUI support

    @version 2013/12/20 started

    copyright 2013 stefan.berke @ modular-audio-graphics.com
*/
#ifndef PROPERTY_H
#define PROPERTY_H

#include <functional>

#include <QString>
#include <QStringList>

class QLayout;


/** A multi-type property.

    This class holds one property or many properties of the same type,
    a persistent property id and a user-friendly name.

    It can create QWidget classes to expose properties in the GUI.

    @todo
    It is planned to handle XML import/export, which would allow
    for whole projects to be saved and restored.

    This is the Qt version of Property. It would be desirable
    to have a non-Qt version in core/ and handle project properties
    there. Right now, the core/ code can not be executed trivially
    because ProjectView handles the properties right now.
    However XML i/o would require QT or another library.
*/

class Property
{
public:
    enum Type
    {
        BOOL = 0,
        INT,
        FLOAT,
        SELECT,
//        STRING,
        /** must be LAST (for numTypes below) */
        UNKNOWN
    };

    enum LayoutType
    {
        LABEL_WIDGET,
        WIDGET_LABEL,
        V_LABEL_WIDGET,
        H_WIDGET_LABEL
    };

    static const int     numTypes = UNKNOWN-1;
    static const char *  typeName[];
    static       int     name2int(const QString& n);

    Type                 type;
    QString              id, name, help;
    size_t               dim;

    // the values
    std::vector<bool>    v_bool;
    std::vector<int>     v_int;
    std::vector<float>   v_float;
    std::vector<QString> v_str;

    // ranges
    int                  min_int, max_int;
    float                min_float, max_float;

    // SELECT values
    QStringList          item_ids;
    QStringList          item_names;
    std::vector<int>     item_values;

    // --------- functions ----------

    Property(const QString& id, const QString& name, const QString& help = "");

    /** single bool */
    void init(bool value);

    /** single int */
    void init(int   min_val, int   max_val, int   value);
    /** two ints */
    void init(int   min_val, int   max_val, int   value0, int   value1);
    /** single float */
    void init(float min_val, float max_val, float value);
    /** two floats */
    void init(float min_val, float max_val, float value0, float value1);

    /** one select item */
    void init(const std::vector<int>& item_values,
              const QStringList& item_ids,
              const QStringList& item_names, int value);

    /** set new maximum/maximum value (update widget) */
    void setMinMax(int new_min_value, int new_max_value);
    void setMinMax(float new_min_value, float new_max_value);
    /** set new maximum value (update widget) */
    void setMax(int new_max_value) { setMinMax(min_int, new_max_value); }
    void setMax(float new_max_value) { setMinMax(min_float, new_max_value); }

    // --------- callbacks ----------

    /** Sets the callback for when the user changed the value of the widget.
        This callback can be installed before or after a call to createWidget(). */
    void cb_value_changed(std::function<void()> func) { cb_value_changed_ = func; }

    // --------- widgets ------------

    /** Creates a widget for the Property into the layout.
        The widget remains associated with the Property until
        disconnectWidget() is called, or the widget is deleted. */
    void createWidget(QWidget * parent, QLayout * into_layout, LayoutType ltype = LABEL_WIDGET);

    /** Removes the association with the widget.
        This happens automatically on destruction of the widget. */
    void disconnectWidget();

    /** Sets the associated widget(s) to the current Property value.
        Causes a cb_value_changed if 'do_callback' == true. */
    void updateWidget(bool do_callback = true);

    /** activate or deactivate the associated widget */
    void setActive(bool active);

    // _______ PRIVATE AREA ________
private:
    /** default callback from widgets. generates cb_ type callbacks */
    void onValueChanged_();

    /** Adds child as layout to parent. */
    void addLayout_(QLayout * parent, QLayout * child);

    /** Returns the appropriate, connected widget. Also installs in widgets_ */
    QWidget * getWidget_(QWidget * parent, QLayout * layout, size_t value_index);
    /** create a label */
    QWidget * getLabel_(QWidget * parent, QLayout * layout);

    /** a widget for each value [0,dim-1] */
    std::vector<QWidget*> widgets_;

    // callbacks
    std::function<void()> cb_value_changed_;

    /** active state */
    bool active_,
    /** ignore a cb_value_changed once. used for updateWidget(false) */
        ignore_value_changed_once_;
};

#endif // PROPERTY_H
