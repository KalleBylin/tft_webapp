import time
import json
import io
import base64
import requests
import numpy as np
import pandas as pd
import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State


SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "20rem",
    "padding": "2rem 1rem",
    "background-color": "#131514",
}

controls = dbc.Card(
    [
        dbc.FormGroup(
            [
                dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select Files')
                    ]),
                    style={
                         'textAlign': 'center',
                         'margin': '10px'
                    },
                    # Allow multiple files to be uploaded
                    multiple=True
                ),
            ]
        )
    ],
)

sidebar = html.Div(
    [
        html.H2("Upload data", className="lead", style={'color': '#fcfdff'}),
        html.Hr(),
        controls
    ],
    style=SIDEBAR_STYLE,
)

CONTENT_STYLE = {
    "margin-left": "10rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}
content = html.Div(
    [
        html.H1('Volatility prediction', 
                style={'textAlign': 'center'}),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="my_close_price_graph"), 
                        width={"size": 8, "offset": 2}),
            ]
        )
    ],
    style=CONTENT_STYLE
)


app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div([sidebar, content])

@app.callback(Output('my_close_price_graph', 'figure'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def parse_contents(contents, filename, date):
    try:
        content_type, content_string = contents[0].split(',')
        decoded = base64.b64decode(content_string)

        if 'csv' in filename[0]:
            graph_data = []
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), index_col=0)
            time_step = 0
            dates = pd.to_datetime(df.index).values.reshape((64,257))[time_step, :]
            data = df.values.reshape((64, 257, 9))[time_step,:,0]
            graph_data.append({'x': dates[:252], 'y': data[:252], 'name': 'Inputs'})

            json_data = {"inputs": df.values.tolist()}
            r = requests.post("http://model_server:8000/predict", json=json_data)
            task_id = r.json()['task_id']
            print("Task id:", task_id, flush=True)

            status = "IN_PROGRESS"
            while status != "DONE":
                print("Result not ready", flush=True)
                r = requests.get(f"http://model_server:8000/predict/{task_id}")
                status = r.json()['status']
                time.sleep(2)

            preds = np.array(json.loads(r.json())["result"]["outputs"])

            graph_data.append({'x': dates[252:], 'y': preds[252:,0], 'name': 'P10'})
            graph_data.append({'x': dates[252:], 'y': preds[252:,1], 'name': 'P50'})
            graph_data.append({'x': dates[252:], 'y': preds[252:,2], 'name': 'P90'})

            fig = {
                'data': graph_data,
                'layout': {'title': f"Predictions for time step {time_step}"}
            }

    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    
    return fig


if __name__ == '__main__':
    time.sleep(5)
    app.run_server(host='0.0.0.0', port='8083')

    
