import bokeh.model
import bokeh.plotting
import bokeh.io

from bokeh import palettes


def plot_2d_data(
        data,
        text_label,
        cls,
        show_text=True,
        subset=None,
        tools="hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,"
        palette="Category20"
):
    palette = palettes.d3[palette]
    x, y = data[:, 0], data[:, 1]
    source_df = pd.DataFrame({'x': x, 'y': y, 'text_label': text_label, 'color': [palette[c + 3][c] for c in cls]})
    source = bokeh.models.ColumnDataSource(source_df)
    p = bokeh.plotting.figure(tools=tools, plot_width=800, plot_height=600)
    p.scatter(x='x', y='y', source=source, fill_color='color', line_color='color')


    if subset is not None:
        text_labels = bokeh.models.LabelSet(x='x', y='y', text='text_label', level='glyph',
                      x_offset=5, y_offset=5, source=bokeh.models.ColumnDataSource(source_df.iloc[subset]), render_mode='canvas', text_font_size='7pt')
        p.add_layout(text_labels)
    bokeh.plotting.show(p)
