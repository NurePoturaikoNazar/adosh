import os
import numpy as np
from PIL import Image
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(SCRIPT_DIR, 'АДОШ_Лр_4_jpg_100')
ALLOWED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

#fix for white text with a css
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .Select-control {
                background-color: #222 !important;
                border-color: #555 !important;
            }
            .Select-menu {
                background-color: #222 !important;
                border-color: #555 !important;
            }
            .Select-option {
                color: #fff !important;
                background-color: #222 !important;
            }
            .Select-option.is-selected {
                background-color: #375a7f !important;
            }
            .Select-option.is-focused {
                background-color: #375a7f !important;
            }
            .Select-value {
                color: #fff !important;
            }
            .Select-input input {
                color: #fff !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

def get_image_list():
    if not os.path.exists(IMAGE_FOLDER):
        return []
    files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(ALLOWED_EXTENSIONS)]
    return sorted(files)

def image_to_arrays(image_path):
    img = Image.open(image_path).convert('RGB')
    img_arr = np.array(img)
    r = img_arr[:, :, 0]
    g = img_arr[:, :, 1]
    b = img_arr[:, :, 2]
    m = (r.astype(np.float32) + g.astype(np.float32) + b.astype(np.float32)) / 3.0
    return r, g, b, m.astype(np.uint8)

def integrate_and_fire_row(row, p):
    width = len(row)
    pulse_row = np.full(width, 255, dtype=np.uint8)
    s = 0.0
    for j in range(width):
        val = float(row[j])
        s += val
        if s >= p:
            pulse_row[j] = 0
            s -= p
    return pulse_row

def convert_to_pulse_image(channel_data, k=0.8):
    height, width = channel_data.shape
    mx = float(np.max(channel_data))
    mn = float(np.min(channel_data))
    
    p = k * (mx - mn)
    if p < 1:
        p = 10.0
        
    pulse_image = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(height):
        pulse_image[i] = integrate_and_fire_row(channel_data[i], p)
        
    return pulse_image

def logical_operations(pr, pg, pb, pm):
    # Logic: 0 is True (Black/Pulse), 255 is False (White)
    # AND: Result is 0 (Black) if A=0 AND B=0. Else 255.
    # OR: Result is 0 (Black) if A=0 OR B=0. Else 255.
    
    def logic_and(img1, img2):
        # 0 if both are 0, else 255
        # Using max: max(0, 0)=0, max(0, 255)=255, max(255, 255)=255
        return np.maximum(img1, img2)

    def logic_or(img1, img2):
        # 0 if any is 0
        # Using min: min(0, 255)=0
        return np.minimum(img1, img2)
    
    def logic_and3(i1, i2, i3):
        return np.maximum(np.maximum(i1, i2), i3)

    def logic_or3(i1, i2, i3):
        return np.minimum(np.minimum(i1, i2), i3)

    results = {}
    
    results['R_AND_G'] = logic_and(pr, pg)
    results['R_OR_G'] = logic_or(pr, pg)
    
    results['R_AND_B'] = logic_and(pr, pb)
    results['R_OR_B'] = logic_or(pr, pb)
    
    results['G_AND_B'] = logic_and(pg, pb)
    results['G_OR_B'] = logic_or(pg, pb)
    
    results['RGB_AND'] = logic_and3(pr, pg, pb)
    results['RGB_OR'] = logic_or3(pr, pg, pb)
    
    results['M_AND_R'] = logic_and(pm, pr)
    results['M_OR_R'] = logic_or(pm, pr)
    
    results['M_AND_G'] = logic_and(pm, pg)
    results['M_OR_G'] = logic_or(pm, pg)
    
    results['M_AND_B'] = logic_and(pm, pb)
    results['M_OR_B'] = logic_or(pm, pb)
    
    return results

def fig_from_img(img_arr, title):
    fig = px.imshow(img_arr, binary_string=True, binary_format='jpg')
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=10)),
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False),
        coloraxis_showscale=False,
        template='plotly_dark',
        height=250
    )
    return fig

app.layout = dbc.Container([
    html.H2("ЛР №4: Імпульсне зображення та логічні операції", className="my-3 text-center"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Label("Оберіть зображення:"),
                    dbc.Row([
                        dbc.Col(dcc.Dropdown(
                            id='image-dropdown',
                            options=[],
                            value=None,
                            clearable=False
                        ), width=10),
                        dbc.Col(dbc.Button("↻", id='refresh-btn', color="secondary", size="sm"), width=2)
                    ], className="g-2"),
                    html.Br(),
                    html.Label("Коефіцієнт k (Поріг):"),
                    dcc.Slider(
                        id='slider-k', min=0.1, max=2.0, step=0.1, value=0.8,
                        marks={0.5: '0.5', 0.8: '0.8', 1.0: '1.0', 1.5: '1.5'}
                    ),
                    html.Br(),
                    dbc.Button("Обробити", id='process-btn', color="primary", className="w-100")
                ])
            ], className="mb-3")
        ], width=3),
        
        dbc.Col([
            dcc.Loading(id="loading", type="dot", children=[
                dbc.Tabs([
                    dbc.Tab(label="1. Канали та Імпульси", tab_id="tab-channels"),
                    dbc.Tab(label="2. Логічні операції (Матриця)", tab_id="tab-logic"),
                ], id="tabs", active_tab="tab-channels"),
                html.Div(id="content-area", className="mt-3")
            ])
        ], width=9)
    ])
], fluid=True)

@app.callback(
    Output("content-area", "children"),
    [Input("process-btn", "n_clicks"),
     Input("tabs", "active_tab")],
    [State("image-dropdown", "value"),
     State("slider-k", "value")]
)
def update_output(n_clicks, active_tab, filename, k):
    if not filename:
        return dbc.Alert("Папка порожня або файл не обрано", color="warning")
    
    path = os.path.join(IMAGE_FOLDER, filename)
    r, g, b, m = image_to_arrays(path)
    
    pr = convert_to_pulse_image(r, k)
    pg = convert_to_pulse_image(g, k)
    pb = convert_to_pulse_image(b, k)
    pm = convert_to_pulse_image(m, k)
    
    if active_tab == "tab-channels":
        return html.Div([
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_from_img(r, "Red (Input)")), width=3),
                dbc.Col(dcc.Graph(figure=fig_from_img(g, "Green (Input)")), width=3),
                dbc.Col(dcc.Graph(figure=fig_from_img(b, "Blue (Input)")), width=3),
                dbc.Col(dcc.Graph(figure=fig_from_img(m, "M (Average)")), width=3),
            ], className="mb-2"),
            html.Hr(),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_from_img(pr, "Pulse Red")), width=3),
                dbc.Col(dcc.Graph(figure=fig_from_img(pg, "Pulse Green")), width=3),
                dbc.Col(dcc.Graph(figure=fig_from_img(pb, "Pulse Blue")), width=3),
                dbc.Col(dcc.Graph(figure=fig_from_img(pm, "Pulse M")), width=3),
            ])
        ])
        
    elif active_tab == "tab-logic":
        res = logical_operations(pr, pg, pb, pm)
        
        return html.Div([
            dbc.Row([
                dbc.Col(html.H5("Взаємодія R, G, B", className="text-center"), width=12)
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_from_img(res['R_AND_G'], "R AND G")), width=3),
                dbc.Col(dcc.Graph(figure=fig_from_img(res['R_OR_G'], "R OR G")), width=3),
                dbc.Col(dcc.Graph(figure=fig_from_img(res['RGB_AND'], "R AND G AND B")), width=3),
                dbc.Col(dcc.Graph(figure=fig_from_img(res['RGB_OR'], "R OR G OR B")), width=3),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_from_img(res['R_AND_B'], "R AND B")), width=3),
                dbc.Col(dcc.Graph(figure=fig_from_img(res['R_OR_B'], "R OR B")), width=3),
                dbc.Col(dcc.Graph(figure=fig_from_img(res['G_AND_B'], "G AND B")), width=3),
                dbc.Col(dcc.Graph(figure=fig_from_img(res['G_OR_B'], "G OR B")), width=3),
            ]),
            html.Hr(),
            dbc.Row([
                dbc.Col(html.H5("Взаємодія з M (Середнє)", className="text-center"), width=12)
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_from_img(res['M_AND_R'], "M AND R")), width=4),
                dbc.Col(dcc.Graph(figure=fig_from_img(res['M_AND_G'], "M AND G")), width=4),
                dbc.Col(dcc.Graph(figure=fig_from_img(res['M_AND_B'], "M AND B")), width=4),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_from_img(res['M_OR_R'], "M OR R")), width=4),
                dbc.Col(dcc.Graph(figure=fig_from_img(res['M_OR_G'], "M OR G")), width=4),
                dbc.Col(dcc.Graph(figure=fig_from_img(res['M_OR_B'], "M OR B")), width=4),
            ]),
        ])

#callback to update dropdown
@app.callback(
    [Output('image-dropdown', 'options'),
     Output('image-dropdown', 'value')],
    [Input('image-dropdown', 'id'),
     Input('refresh-btn', 'n_clicks')],
    prevent_initial_call=False
)
def update_image_dropdown(_, __):
    images = get_image_list()
    options = [{'label': f, 'value': f} for f in images]
    value = images[0] if images else None
    return options, value

if __name__ == '__main__':
    app.run(debug=True)