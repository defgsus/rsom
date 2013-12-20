/** @file
    @brief Property / value handler with GUI support

    @version 2013/12/20 started

    copyright 2013 stefan.berke @ modular-audio-graphics.com

    This program is coverd by the GNU General Public License
*/
#ifndef PROPERTY_H
#define PROPERTY_H

#include <functional>

#include <QString>
#include <QStringList>

class QLayout;


class Property
{
public:
    enum Type
    {
        BOOL = 0,
        INT,
        FLOAT,
        SELECT,
        STRING,
        UNKNOWN
    };

    enum LayoutType
    {
        LABEL_WIDGET,
        WIDGET_LABEL,
        V_LABEL_WIDGET,
        H_WIDGET_LABEL
    };

    static const int     numTypes = STRING;
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

    Property(const QString& id, const QString& name, const QString& help);

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

    // --------- callbacks ----------

    /** Sets the callback for when the user changed the value of the widget.
        This callback can be installed before or after a call to createWidget(). */
    void cb_value_changed(std::function<void()> func) { cb_value_changed_ = func; }

    // --------- widgets ------------

    /** Creates a widget for the Property into the layout.
        The widget remains associated with the Property until
        disconnectWidget() is called. */
    void createWidget(QWidget * parent, QLayout * into_layout, LayoutType ltype = LABEL_WIDGET);

    /** Removes the association with the widget. */
    void disconnectWidget();

    /** Sets the associated widget to the current Property value. */
    void updateWidget();

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

    /** a widget for each value (dim) */
    std::vector<QWidget*> widgets_;

    // callbacks
    std::function<void()> cb_value_changed_;
};

#endif // PROPERTY_H
