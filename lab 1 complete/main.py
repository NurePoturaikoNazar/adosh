import wave
import numpy as np
from scipy.signal import find_peaks
from typing import Tuple, Optional
import base64
import io

import dash
from dash import dcc, html, Input, Output, State, no_update
import plotly.graph_objs as go
import dash_bootstrap_components as dbc

EXPECTED_FRAMERATE = 22050
EXPECTED_CHANNELS = 1
EXPECTED_SAMPLE_WIDTH = 1
SMOOTHING_WINDOW = 16
DEFAULT_WINDOW_T = 500
DEFAULT_L_RANGE = (80, 200)
OPTIMIZATION_T_RANGE = range(200, 701, 50)
OPTIMIZATION_L_MIN_RANGE = range(80, 201, 10)
OPTIMIZATION_L_MAX = 200
MIN_POINTS_FOR_VALID_OPTIMIZATION = 50
MAX_CLUSTER_DISTANCE_SQ = 50 ** 2
PEAK_PROMINENCE = 10


def create_empty_figure(title):
    return go.Figure().update_layout(
        title=title,
        template="plotly_dark",
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=250,
        margin=dict(t=30, b=30, l=30, r=30)
    )


def smooth_signal(signal: np.ndarray, window_size: int) -> np.ndarray:
    print(f"Applying smoothing with window {window_size}...")
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(signal, kernel, mode='same')
    return smoothed


def _find_best_oscillation_in_window(
        window: np.ndarray,
        l_range: Tuple[int, int]
) -> Optional[Tuple[int, int, int]]:
    valleys, _ = find_peaks(-window, distance=SMOOTHING_WINDOW + 1, prominence=PEAK_PROMINENCE)
    peaks, _ = find_peaks(window, distance=SMOOTHING_WINDOW + 1, prominence=PEAK_PROMINENCE)

    if len(valleys) < 2 or len(peaks) < 1:
        return None
    best_oscillation = None
    max_energy = -np.inf
    l_min, l_max = l_range
    for i in range(len(valleys) - 1):
        p0 = valleys[i]
        suitable_peaks = peaks[peaks > p0]
        if len(suitable_peaks) == 0:
            continue
        p1 = suitable_peaks[0]
        suitable_valleys_after_peak = valleys[valleys > p1]
        if len(suitable_valleys_after_peak) == 0:
            continue
        p2 = suitable_valleys_after_peak[0]
        if p2 != valleys[i + 1]:
            continue
        length = p2 - p0
        if l_min <= length <= l_max:
            energy = np.sum(window[p0:p2])
            if energy > max_energy:
                max_energy = energy
                best_oscillation = (p0, p1, p2)
    return best_oscillation


def _calculate_features_for_oscillation(
        p0: int,
        p1: int,
        p2: int
) -> Tuple[float, float]:
    v1 = float(p1 - p0)
    v2 = float(p2 - p1)
    return v1, v2


def extract_features(
        smoothed_signal: np.ndarray,
        window_t: int,
        l_range: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    print(f"Extracting features: T={window_t}, L_range={l_range}...", end=" ")
    features = []
    all_points = []
    current_pos = 0
    while current_pos + window_t < len(smoothed_signal):
        window_data = smoothed_signal[current_pos: current_pos + window_t]
        oscillation_points = _find_best_oscillation_in_window(window_data, l_range)
        if oscillation_points:
            p0_window, p1_window, p2_window = oscillation_points
            p0_abs = current_pos + p0_window
            p1_abs = current_pos + p1_window
            p2_abs = current_pos + p2_window
            v1, v2 = _calculate_features_for_oscillation(
                p0_abs, p1_abs, p2_abs
            )
            if v1 != 0.0 or v2 != 0.0:
                features.append([v1, v2])
                all_points.append([p0_abs, p1_abs, p2_abs])
            current_pos = p2_abs
        else:
            current_pos += window_t // 4
    print(f"Found {len(features)} feature vectors.")
    return np.array(features), np.array(all_points)


def calculate_criterion_R(features: np.ndarray) -> float:
    if features.shape[0] < 2:
        return np.inf
    diffs = features[1:] - features[:-1]
    sum_sq_diffs = np.sum(diffs ** 2, axis=1)
    intra_cluster_distances = sum_sq_diffs[sum_sq_diffs < MAX_CLUSTER_DISTANCE_SQ]
    if len(intra_cluster_distances) < 1:
        return np.inf
    r_values = 0.5 * np.sqrt(intra_cluster_distances)
    return np.mean(r_values)


def run_optimization_task_dash(smoothed_signal: np.ndarray):
    print("\n--- Starting Task 2: Optimization ---")
    best_R = np.inf
    best_params = {}

    for T in OPTIMIZATION_T_RANGE:
        for L_min in OPTIMIZATION_L_MIN_RANGE:
            current_l_range = (L_min, OPTIMIZATION_L_MAX)

            features, _ = extract_features(
                smoothed_signal, T, current_l_range
            )
            R = calculate_criterion_R(features)
            print(f"  Testing T={T}, L_min={L_min}... Result: R = {R:.4f}")

            if R < best_R and features.shape[0] > MIN_POINTS_FOR_VALID_OPTIMIZATION:
                best_R = R
                best_params = {'T': T, 'L_min': L_min}
                print(f"!!! New optimal found: R={best_R:.4f} with T={T}, L_min={L_min}")

    if 'T' not in best_params:
        print("\n--- Optimization failed to find any valid parameters ---")
        return None, "Optimization failed to find any valid parameters."

    print("\n--- Optimization Complete ---")
    opt_T = best_params['T']
    opt_L_range = (best_params['L_min'], OPTIMIZATION_L_MAX)

    final_features, _ = extract_features(
        smoothed_signal, opt_T, opt_L_range
    )

    status = f"Оптимізація завершена. Best R={best_R:.4f}, T={opt_T}, L_min={opt_L_range[0]}"
    return final_features, status


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

initial_figure = go.Figure().update_layout(
    title="Будь ласка, завантажте .WAV файл",
    template="plotly_dark",
    plot_bgcolor='rgba(0, 0, 0, 0)',
    paper_bgcolor='rgba(0, 0, 0, 0)'
)

initial_fig_full = create_empty_figure("Повний сигнал (очікування...)")
initial_fig_noise = create_empty_figure("Фрагмент з шумом (очікування...)")
initial_fig_smoothed_fragment = create_empty_figure("Результат згладжування (очікування...)")
initial_fig_mid = create_empty_figure("Сигнал з маркерами 0-1-2 (очікування...)")
initial_fig_zoom = create_empty_figure("Одне коливання 0-1-2 (очікування...)")

app.layout = dbc.Container([
    dcc.Store(id='loading-store', data=False),
    dcc.Store(id='task1-trigger-store'),
    dcc.Store(id='task2-trigger-store'),
    dcc.Store(id='signal-store'),

    html.H1("Лабораторна робота №1: Трикутник голосних", className="my-4"),

    dbc.Row([
        dbc.Col(md=4, children=[
            dbc.Card(body=True, children=[
                html.H3("Крок 1: Завантаження файлу", className="card-title"),
                dcc.Upload(
                    id='upload-wav',
                    children=html.Div([
                        html.A('Перетягніть або Виберіть .WAV файл')
                    ]),
                    style={
                        'width': '100%', 'height': '60px', 'lineHeight': '60px',
                        'borderWidth': '1px', 'borderStyle': 'dashed',
                        'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px 0'
                    },
                    accept='.wav'
                ),
                html.Div(id='file-status-message', className="text-muted small")
            ]),

            dbc.Card(body=True, className="mt-4", children=[
                html.H3("Завдання 1: Інтерактивна побудова", className="card-title"),
                dbc.Label("Розмір вікна (T):"),
                dbc.Input(id='input-t', type='number', value=DEFAULT_WINDOW_T),
                dbc.Label("Мін. довжина (L_min):", className="mt-3"),
                dbc.Input(id='input-l-min', type='number', value=DEFAULT_L_RANGE[0]),
                dbc.Label("Макс. довжина (L_max):", className="mt-3"),
                dbc.Input(id='input-l-max', type='number', value=DEFAULT_L_RANGE[1]),
                dbc.Button(
                    "Побудувати трикутник",
                    id='btn-run-task1', n_clicks=0, color="primary",
                    className="w-100 mt-4", disabled=True
                )
            ]),

            dbc.Card(body=True, className="mt-4", children=[
                html.H3("Завдання 2: Оптимізація", className="card-title"),
                dbc.Button(
                    "Знайти оптимальні параметри (повільно)",
                    id='btn-run-task2', n_clicks=0, color="success",
                    className="w-100", disabled=True
                ),
                html.Div(
                    id='optimization-status',
                    children="Натисніть для запуску оптимізації...",
                    className="mt-2 text-muted small"
                )
            ])
        ]),

        dbc.Col(md=8, children=[
            html.Div(id="graph-wrapper", style={"position": "relative", "height": "70vh"}, children=[
                html.Div(
                    id="manual-spinner-overlay",
                    style={
                        "position": "absolute", "top": 0, "left": 0,
                        "width": "100%", "height": "100%",
                        "display": "flex", "justifyContent": "center", "alignItems": "center",
                        "backgroundColor": "rgba(0, 0, 0, 0.5)",
                        "visibility": "hidden",
                        "zIndex": 10
                    },
                    children=[
                        dbc.Spinner(
                            type="grow",
                            color="info",
                            spinner_style={"width": "3rem", "height": "3rem"}
                        )
                    ]
                ),
                dcc.Graph(
                    id='vowel-triangle-graph',
                    style={'height': '100%'},
                    figure=initial_figure
                )
            ]),

            html.Hr(),
            html.H4("Детальний аналіз сигналу (з Завдання 1)", className="mt-4"),
            dbc.Alert("Ці графіки оновлюються лише при запуску 'Завдання 1: Інтерактивна побудова'.", color="info",
                      className="small"),

            dbc.Row([
                dbc.Col(md=12, children=[
                    dcc.Graph(id='signal-full-graph', figure=initial_fig_full)
                ])
            ]),
            dbc.Row([
                dbc.Col(md=6, children=[
                    dcc.Graph(id='signal-noise-graph', figure=initial_fig_noise)
                ]),
                dbc.Col(md=6, children=[
                    dcc.Graph(id='signal-smoothed-fragment-graph', figure=initial_fig_smoothed_fragment)
                ])
            ]),
            dbc.Row([
                dbc.Col(md=6, children=[
                    dcc.Graph(id='signal-mid-graph', figure=initial_fig_mid)
                ]),
                dbc.Col(md=6, children=[
                    dcc.Graph(id='signal-zoom-graph', figure=initial_fig_zoom)
                ])
            ]),
        ])
    ])
], fluid=True)


def create_figure(features_data, title_text):
    if features_data is None or features_data.shape[0] == 0:
        fig = go.Figure().update_layout(
            title=f"{title_text} (Не знайдено точок)",
            xaxis_title="v1 (Ознака 1) - Довжина [0:1]",
            yaxis_title="v2 (Ознака 2) - Довжина [1:2]",
            yaxis_scaleanchor='x',
            height=600,
            template="plotly_dark",
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            xaxis_range=[0, 200],
            yaxis_range=[0, 200])
        return fig

    v1_max = np.max(features_data[:, 0]) if features_data.shape[0] > 0 else 200
    v2_max = np.max(features_data[:, 1]) if features_data.shape[0] > 0 else 200
    max_range = max(v1_max, v2_max) * 1.1

    fig = go.Figure(data=go.Scatter(
        x=features_data[:, 0],
        y=features_data[:, 1],
        mode='markers',
        marker=dict(size=8, opacity=0.7)
    ))

    fig.update_layout(
        title=title_text,
        xaxis_title="v1 (Ознака 1) - Довжина [0:1]",
        yaxis_title="v2 (Ознака 2) - Довжина [1:2]",
        yaxis_scaleanchor='x',
        height=600,
        template="plotly_dark",
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        xaxis=dict(showgrid=True, gridcolor='#444', range=[0, max_range]),
        yaxis=dict(showgrid=True, gridcolor='#444', range=[0, max_range])
    )
    return fig


@app.callback(
    [Output('signal-store', 'data'),
     Output('file-status-message', 'children')],
    Input('upload-wav', 'contents'),
    State('upload-wav', 'filename'),
    prevent_initial_call=True
)
def handle_file_upload(contents, filename):
    if contents is None:
        return None, "Файл не завантажено."

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        with wave.open(io.BytesIO(decoded), 'rb') as wf:
            framerate = wf.getframerate()
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()

            if (framerate != EXPECTED_FRAMERATE or
                    channels != EXPECTED_CHANNELS or
                    sampwidth != EXPECTED_SAMPLE_WIDTH):
                raise ValueError(
                    f"Invalid WAV format. Expected: {EXPECTED_FRAMERATE}Hz, {EXPECTED_CHANNELS}ch, {EXPECTED_SAMPLE_WIDTH * 8}-bit. "
                    f"Found: {framerate}Hz, {channels}ch, {sampwidth * 8}-bit"
                )

            n_frames = wf.getnframes()
            frames = wf.readframes(n_frames)
            signal = np.frombuffer(frames, dtype=np.uint8).astype(np.float64) - 128.0

        smoothed_signal = smooth_signal(signal, SMOOTHING_WINDOW)

        store_data = {
            'raw': signal.tolist(),
            'smoothed': smoothed_signal.tolist()
        }

        message = f"Файл '{filename}' успішно завантажено та оброблено."
        print(message)
        return store_data, dbc.Alert(message, color="success", duration=4000)

    except Exception as e:
        print(f"Помилка завантаження файлу: {e}")
        return None, dbc.Alert(f"Помилка обробки файлу: {e}", color="danger")


@app.callback(
    Output('task1-trigger-store', 'data'),
    Input('btn-run-task1', 'n_clicks'),
    [State('input-t', 'value'),
     State('input-l-min', 'value'),
     State('input-l-max', 'value')],
    prevent_initial_call=True
)
def trigger_task1(n_clicks, t_val, l_min_val, l_max_val):
    return {'t': t_val, 'l_min': l_min_val, 'l_max': l_max_val}


@app.callback(
    Output('task2-trigger-store', 'data'),
    Input('btn-run-task2', 'n_clicks'),
    prevent_initial_call=True
)
def trigger_task2(n_clicks):
    return n_clicks


@app.callback(
    Output('loading-store', 'data'),
    [Input('task1-trigger-store', 'data'),
     Input('task2-trigger-store', 'data')],
    prevent_initial_call=True
)
def set_loading_on_any_trigger(_, __):
    return True


@app.callback(
    [Output('btn-run-task1', 'disabled'),
     Output('btn-run-task2', 'disabled'),
     Output('manual-spinner-overlay', 'style')],
    [Input('loading-store', 'data'),
     Input('signal-store', 'data')],
    [State('manual-spinner-overlay', 'style')]
)
def update_button_and_spinner_state(is_loading, signal_data, current_style):
    buttons_disabled = is_loading or (signal_data is None)

    if is_loading:
        current_style['visibility'] = 'visible'
    else:
        current_style['visibility'] = 'hidden'

    return buttons_disabled, buttons_disabled, current_style


@app.callback(
    [Output('vowel-triangle-graph', 'figure'),
     Output('signal-full-graph', 'figure'),
     Output('signal-noise-graph', 'figure'),
     Output('signal-smoothed-fragment-graph', 'figure'),
     Output('signal-mid-graph', 'figure'),
     Output('signal-zoom-graph', 'figure'),
     Output('loading-store', 'data', allow_duplicate=True)],
    Input('task1-trigger-store', 'data'),
    State('signal-store', 'data'),
    prevent_initial_call=True
)
def run_task1_calculation(trigger_data, signal_data):
    if trigger_data is None or signal_data is None:
        return no_update, no_update, no_update, no_update, no_update, no_update, False

    raw_signal = np.array(signal_data['raw'])
    smoothed_signal = np.array(signal_data['smoothed'])

    t_val = trigger_data['t']
    l_min_val = trigger_data['l_min']
    l_max_val = trigger_data['l_max']

    features, points = extract_features(
        smoothed_signal,
        t_val,
        (l_min_val, l_max_val)
    )
    title = f"Vowel Triangle (T={t_val}, L in ({l_min_val}, {l_max_val}))"
    fig_triangle = create_figure(features, title)

    small_graph_height = 250
    full_graph_height = 300

    fig_full = go.Figure(data=go.Scatter(y=smoothed_signal, name='Згладжений сигнал (а-у-і)'))
    fig_full.update_layout(title="Повний мовний сигнал", template="plotly_dark", height=full_graph_height,
                           margin=dict(t=30, b=30, l=30, r=30))

    if points.shape[0] > 0:
        slice_center = points[0, 0]
        noise_slice_start = max(0, slice_center - 500)
        noise_slice_end = min(len(raw_signal), slice_center + 500)
    else:
        noise_slice_start = max(0, len(raw_signal) // 3)
        noise_slice_end = min(len(raw_signal), noise_slice_start + 1000)

    fig_noise = go.Figure(
        data=go.Scatter(y=raw_signal[noise_slice_start:noise_slice_end], name='Сигнал (до згладжування)'))
    fig_noise.update_layout(title="Фрагмент з шумом (Рис. 1.7)", template="plotly_dark", height=small_graph_height,
                            margin=dict(t=30, b=30, l=30, r=30))

    fig_smoothed_fragment = go.Figure(
        data=go.Scatter(y=smoothed_signal[noise_slice_start:noise_slice_end], name='Сигнал (після згладжування)'))
    fig_smoothed_fragment.update_layout(title="Результат згладжування (Рис. 1.8)", template="plotly_dark",
                                        height=small_graph_height, margin=dict(t=30, b=30, l=30, r=30))

    if points.shape[0] > 0:
        mid_range_end = min(len(smoothed_signal), points[min(20, points.shape[0] - 1), 2] + 100)

        p0_pts = points[:, 0]
        p1_pts = points[:, 1]
        p2_pts = points[:, 2]

        p0_pts = p0_pts[p0_pts < mid_range_end]
        p1_pts = p1_pts[p1_pts < mid_range_end]
        p2_pts = p2_pts[p2_pts < mid_range_end]

        fig_mid = go.Figure()
        fig_mid.add_trace(go.Scatter(y=smoothed_signal, name='Згладжений сигнал'))
        fig_mid.add_trace(go.Scatter(x=p0_pts, y=smoothed_signal[p0_pts], mode='markers', name='Точка 0',
                                     marker=dict(color='yellow')))
        fig_mid.add_trace(
            go.Scatter(x=p1_pts, y=smoothed_signal[p1_pts], mode='markers', name='Точка 1', marker=dict(color='red')))
        fig_mid.add_trace(
            go.Scatter(x=p2_pts, y=smoothed_signal[p2_pts], mode='markers', name='Точка 2', marker=dict(color='cyan')))
        fig_mid.update_layout(title="Маркери 0-1-2 (Рис. 1.5)", template="plotly_dark", xaxis_range=[0, mid_range_end],
                              height=small_graph_height, margin=dict(t=30, b=30, l=30, r=30))

        p0, p1, p2 = points[0]
        padding = (p2 - p0)
        start = max(0, p0 - padding)
        end = min(len(smoothed_signal), p2 + padding)
        x_axis = np.arange(start, end)
        y_axis = smoothed_signal[start:end]

        fig_zoom = go.Figure()
        fig_zoom.add_trace(go.Scatter(x=x_axis, y=y_axis, name='Одне коливання', line=dict(width=1)))
        fig_zoom.add_trace(go.Scatter(
            x=[p0, p1, p2],
            y=smoothed_signal[[p0, p1, p2]],
            mode='markers+text',
            name='Точки 0-1-2',
            text=['0', '1', '2'],
            textposition='bottom center',
            marker=dict(size=8)
        ))
        fig_zoom.update_layout(title="Одне коливання (Рис. 1.6)", template="plotly_dark", height=small_graph_height,
                               margin=dict(t=30, b=30, l=30, r=30))

    else:
        fig_mid = create_empty_figure("Точки 0-1-2 (не знайдено)")
        fig_zoom = create_empty_figure("Одне коливання (не знайдено)")

    return fig_triangle, fig_full, fig_noise, fig_smoothed_fragment, fig_mid, fig_zoom, False


@app.callback(
    [Output('vowel-triangle-graph', 'figure', allow_duplicate=True),
     Output('optimization-status', 'children'),
     Output('loading-store', 'data', allow_duplicate=True)],
    Input('task2-trigger-store', 'data'),
    State('signal-store', 'data'),
    prevent_initial_call=True
)
def run_task2_calculation(n_clicks, signal_data):
    if n_clicks is None or signal_data is None:
        return no_update, no_update, False

    smoothed_signal = np.array(signal_data['smoothed'])

    print("Запуск оптимізації...")
    features, status = run_optimization_task_dash(smoothed_signal)
    print(status)
    fig = create_figure(features, f"Оптимальний трикутник: {status}")

    return fig, status, False


@app.callback(
    Output('vowel-triangle-graph', 'figure', allow_duplicate=True),
    Input('signal-store', 'data'),
    prevent_initial_call=True
)
def update_graph_on_upload(signal_data):
    if signal_data is None:
        return go.Figure().update_layout(
            title="Помилка завантаження. Спробуйте ще.",
            template="plotly_dark",
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)'
        )
    else:
        return go.Figure().update_layout(
            title="Файл завантажено. Натисніть 'Побудувати' для початку",
            template="plotly_dark",
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)'
        )


if __name__ == '__main__':
    app.run(debug=False)
