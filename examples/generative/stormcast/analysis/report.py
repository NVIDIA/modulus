import os 
import glob
import json
import numpy as np
import xarray as xr
import pandas as pd
import base64
from io import BytesIO
import dash
from dash import html
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import matplotlib as mpl
mpl.use('Agg')
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from metpy.plots import ctables
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm

data_path = '/home/yacohen/code/fcn-dev/report/data/'
forecast_path = '/home/yacohen/code/fcn-dev/report/forecast/'
color_palette = px.colors.qualitative.Plotly
def get_color_map(models, palette):
    return {model: palette[i % len(palette)] for i, model in enumerate(models)}


def load_data_for_tabs(csv_files, metrics_list):
    metrics_data = {}
    spectra_data = {}
    dist_data = {}
    for file in csv_files:
        df = pd.read_csv(file)
        if 'Level' in df.columns:
            df['Level'] = df['Level'].fillna(1)
        data_type = df['Data'].unique()[0]
        if data_type in metrics_list:
            metrics_data[file] = df
        elif data_type == 'spectra':
            spectra_data[file] = df
        elif data_type == 'distribution':
            dist_data[file] = df

    # Concatenate DataFrames stored in dictionaries
    combined_metrics_df = pd.concat(metrics_data.values(), ignore_index=True) if metrics_data else pd.DataFrame()
    combined_spectra_df = pd.concat(spectra_data.values(), ignore_index=True) if spectra_data else pd.DataFrame()
    combined_distribution_df = pd.concat(dist_data.values(), ignore_index=True) if dist_data else pd.DataFrame()
    
    return combined_metrics_df, combined_spectra_df, combined_distribution_df

metrics_list = ['rmse', 'crps', 'fss', 'pmm']
csv_files = glob.glob(os.path.join(data_path, '*.csv'))
combined_metrics_df, combined_spectra_df, combined_distribution_df = load_data_for_tabs(csv_files, metrics_list)

app = dash.Dash(__name__)
metrics_model_options = [{'label': model, 'value': model} for model in combined_metrics_df['Model'].unique()]
metrics_type_options = [{'label': Type, 'value': Type} for Type in combined_metrics_df['Type'].unique()]
metrics_type_options_ =[option for option in metrics_type_options if option['value'] != 'target']
metrics_data = [{'label': metric, 'value': metric} for metric in combined_metrics_df['Data'].unique()]
metrics_variable_options = [{'label': var, 'value': var} for var in combined_metrics_df['Variable'].unique()]
metrics_level_options = [{'label': var, 'value': var} for var in combined_metrics_df['Level'].unique() if var != 0.0]
metrics_initial_variable = metrics_variable_options[1]['value'] if metrics_variable_options else None
spectra_model_options = [{'label': model, 'value': model} for model in combined_spectra_df['Model'].unique()]
spectra_type_options = [{'label': Type, 'value': Type} for Type in combined_spectra_df['Type'].unique()]
spectra_type_options_ =[option for option in spectra_type_options if option['value'] != 'target']
spectra_data = [{'label': metric, 'value': metric} for metric in combined_spectra_df['Data'].unique()]
spectra_variable_options = [{'label': var, 'value': var} for var in combined_spectra_df['Variable'].unique()]
spectra_level_options = [{'label': var, 'value': var} for var in combined_spectra_df['Level'].unique() if var != 0.0]
spectra_initial_variable = spectra_level_options[1]['value'] if spectra_level_options else None

channel_options = ["refc", "t_comb", "divergence", "relative_humidity"]
case_options = ["Missouri_nocturnal_mcs", "Texas_drought", "Kansas_mcs", "Iowa_cumulus", "Iow_convection", 
                "texas_ok_squall_line", "Illinois_summer_convection"]

def create_tab_top_section(dropdowns_and_sliders, description_container_id):
    return html.Div([
        html.Div(
            dropdowns_and_sliders, 
            style={'width': '50%', 'display': 'inline-block'}
        ),
        html.Div(
            id=description_container_id, 
            style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'}
        )
    ], style={'display': 'flex', 'flexDirection': 'row'})


tab1_controls = html.Div([
    html.Div([
        html.Label("Model:"),
        dcc.Dropdown(
            id='model-dropdown-tab1',
            options=metrics_model_options,
            value=[metrics_model_options[0]['value']],
            multi=True, 
            style={'width': '600px'}
        ),
    ]),
    html.Div([
        html.Label("Reg/Diff:"),
        dcc.Dropdown(
            id='type-dropdown-tab1',
            options=metrics_type_options_,
            value=[metrics_type_options_[0]['value']],
            multi=True,
            style={'width': '300px'}
        ),
    ]),
    html.Div([
        html.Label("Level:"),
        dcc.Dropdown(
            id='level-dropdown-tab1',
            options=metrics_level_options,
            value=[metrics_level_options[0]['value']][0],
            style={'width': '300px'}
        ),
    ]),
    html.Div([
        html.Label("Variable:"),
        dcc.Dropdown(
            id='variable-dropdown-tab1',
            options=metrics_variable_options,
            value=['refc'][0],
            style={'width': '300px'}
        ),
    ]),
])

tab1_layout = html.Div([
    create_tab_top_section(tab1_controls,'model-description-container1'),
    html.Div(id='metrics-tab'),
])


tab2_controls = html.Div([
    html.Div([
        html.Label("Model:"),
            dcc.Dropdown(
                id='model-dropdown-tab2',
                options=spectra_model_options,
                value=[spectra_model_options[0]['value']],
                multi=True,
                style={'width': '600px'}
            ),
        ]),
        html.Div([
            html.Label("Reg/Diff:"),
            dcc.Dropdown(
                id='type-dropdown-tab2',
                options=spectra_type_options_,
                value=[spectra_type_options_[0]['value']],
                multi=True,
                style={'width': '300px'}
            ),
        ]),
        html.Div([
            html.Label("Level:"),
            dcc.Dropdown(
                id='level-dropdown-tab2',
                options=spectra_level_options,
                value=[spectra_level_options[0]['value']][0],
                style={'width': '300px'}
            ),
        ]),
        html.Div([
            html.Label("Variable:"),
            dcc.Dropdown(
                id='variable-dropdown-tab2',
                options=spectra_variable_options,
                value=['refc'][0],
                style={'width': '300px'}
            ),
        ]),
        html.Div([
            html.Label("Time:"),
            dcc.Slider(
                min=0,
                max=11,
                step=1,
                value=0,
                marks={i: str(i+1) for i in range(0, 12)},
                id='time-slider-tab2'
            ),
        ]),
])


tab2_layout = html.Div([
    create_tab_top_section(tab2_controls,'model-description-container2'),
    html.Div(id='spectra-tab'),
])

tab3_controls = html.Div([
    html.Div([
        html.Label("Model:"),
        dcc.Dropdown(
            id='model-dropdown-tab3',
            options=metrics_model_options,
            value=[metrics_model_options[0]['value']],
            multi=True,
            style={'width': '600px'}
        ),
    ]),
    html.Div([
        html.Label("Case:"),
        dcc.Dropdown(
            id='case-dropdown-tab3',
            options=case_options,
            value=[case_options[0]],
            multi=False,
            style={'width': '300px'}
        ),
    ]),
    html.Div([
        html.Label("Time:"),
        dcc.Slider(
            min=0,
            max=11,
            step=1,
            value=0,
            marks={i: str(i+1) for i in range(0, 12)},
            id='time-slider-tab3'
        ),
    ], style={'width': '600px'}), 
    html.Div([
        html.Label("channel"),
        dcc.Dropdown(
            id='channel-dropdown-tab3',
            options=channel_options,
            value=['refc'][0],
            multi=False,
            style={'width': '300px'}
        ),
    ]),
])


tab3_layout = html.Div([
    create_tab_top_section(tab3_controls,'model-description-container3'),
    html.Div(id='cases-tab'),
])


app.layout = html.Div([
    html.H1("US_DLWP Dashboard"),
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='metrics', style={'fontSize': '20px'}, value='tab-1', children=tab1_layout),
        dcc.Tab(label='spectra and distribution', value='tab-2', children=tab2_layout),
        dcc.Tab(label='case studies', value='tab-3',  children=tab3_layout),
    ]),
])


def set_level_options(selected_variable):
    levels = combined_metrics_df[combined_metrics_df['Variable'] == selected_variable]['Level'].unique()
    levels = [int(level) if not pd.isna(level) else 1 for level in levels]
    level_options = [{'label': str(level), 'value': level} for level in sorted(set(levels)) if level != 0.0]
    level_value = level_options[0]['value']
    return level_options, level_value

@app.callback(
    [Output('level-dropdown-tab1', 'options'),
     Output('level-dropdown-tab1', 'value')],
    [Input('variable-dropdown-tab1', 'value')]
)
def update_level_dropdown(selected_variable):
    if not selected_variable:
        # If no variable is selected, return a placeholder
        return [{'label': '1', 'value': 1}], 1
    level_options, level_value = set_level_options(selected_variable)
    return level_options, level_value


@app.callback(
    [Output('metrics-tab', 'children'), 
     Output('model-description-container1', 'children')],
    [Input('model-dropdown-tab1', 'value'),
     Input('type-dropdown-tab1', 'value'),
     Input('variable-dropdown-tab1', 'value'),
     Input('level-dropdown-tab1', 'value')]
)

def update_metric_panels(selected_models, selected_types, selected_variable, selected_level):
    with open('../config/registry.json') as f:
        model_registry = json.load(f)
    #check the analysis folder for more registries of the format "sweep_*_registry.json" and append them to the model_registry
    for file in os.listdir('../analysis'):
        if file.startswith('sweep_') and file.endswith('_registry.json'):
            with open(f'../analysis/{file}') as f:
                sweep_registry = json.load(f)

            for model in sweep_registry['models']:
                model_registry['models'][model] = sweep_registry['models'][model]

    model_descriptions = []
    
    color_map = get_color_map(selected_models, color_palette)
    line_style_map = {'diffusion': 'solid', 'regression': 'dot'}
    unique_data = [metric for metric in metrics_list if metric in combined_metrics_df['Data'].unique()]
    subset_df = combined_metrics_df[combined_metrics_df['Data'].isin(unique_data)]

    num_subplots = len(unique_data)
    
    fig = make_subplots(rows=num_subplots, cols=1, subplot_titles=unique_data, horizontal_spacing=0.15)
    
    for i, metric in enumerate(unique_data, start=1):
        for model in selected_models:
            if i==1:
                try:
                    model_descriptions.append(model + ": " + model_registry['models'][model]['description'])
                except KeyError:
                    model_descriptions.append(model + ": is not in registry, if new model was added update registry.")
            for _type in selected_types:
                filtered_df = subset_df[(subset_df['Data'] == metric) & 
                                        (subset_df['Model'] == model) & 
                                        (subset_df['Type'] == _type) & 
                                        (subset_df['Variable'] == selected_variable)]
                
                filtered_df = filtered_df[filtered_df['Level'] == selected_level]
                if metric == 'fss':
                    filtered_df = filtered_df[filtered_df['Cutoff'] == 0.5]

                
                if not filtered_df.empty:
                    showlegend = True if i == 1 else False
                    fig.add_trace(
                        go.Scatter(
                            x=filtered_df['Time'], 
                            y=filtered_df['Value'], 
                            mode='lines',
                            name=f'{model} - {_type}',
                            line=dict(color=color_map[model], dash=line_style_map[_type]),
                            showlegend=showlegend, 
                        ),
                        row=i, col=1
                    )
    
    fig.update_traces(fill='none', line=dict(shape='linear'))
    for i in range(1, num_subplots + 1):
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', row=i, col=1)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', row=i, col=1)
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=False)
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=False)
        fig.update_xaxes(title_text='time (h)', row=i, col=1)
        
    fig.update_layout(hovermode='x unified')
    
    fig_height = 700
    fig_width = fig_height*1.5
    
    fig.update_layout(width=fig_width, height=fig_height, 
                      margin=dict(l=100, r=300))
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    
    fig.update_layout(legend=dict(
        x=2.0, y=1.0, xanchor='center', yanchor='top', orientation='v'
    ))

    model_descriptions_elements = []
    for full_description in model_descriptions:
        model_name, description_text = full_description.split(":", 1)
        model_descriptions_elements.append(html.P([
            html.Strong(model_name + ": "),
            description_text.strip()
        ]))

    description_container = html.Div(
        children=model_descriptions_elements,
        style={
            'marginTop': '10px',
            'marginBottom': '10px',
            'padding': '10px'
        }
    )

    return dcc.Graph(figure=fig), description_container


@app.callback(
    [Output('level-dropdown-tab2', 'options'),
     Output('level-dropdown-tab2', 'value')],
    [Input('variable-dropdown-tab2', 'value')]
)
def update_level_dropdown(selected_variable):
    if not selected_variable:
        return [{'label': '1', 'value': 1}], 1
    level_options, level_value = set_level_options(selected_variable)
    return level_options, level_value

@app.callback(
    [Output('spectra-tab', 'children'),
     Output('model-description-container2', 'children')],
    [Input('model-dropdown-tab2', 'value'),
     Input('type-dropdown-tab2', 'value'),
     Input('variable-dropdown-tab2', 'value'),
     Input('level-dropdown-tab2', 'value'), 
     Input('time-slider-tab2', 'value')]
)

def update_spectra_panels(selected_models, selected_types, selected_variable, selected_level, selected_time):
    with open('../config/registry.json') as f:
        model_registry = json.load(f)
    #check the analysis folder for more registries of the format "sweep_*_registry.json" and append them to the model_registry
    for file in os.listdir('../analysis'):
        if file.startswith('sweep_') and file.endswith('_registry.json'):
            with open(f'../analysis/{file}') as f:
                sweep_registry = json.load(f)

            for model in sweep_registry['models']:
                model_registry['models'][model] = sweep_registry['models'][model]

    color_map = get_color_map(selected_models, color_palette)
    line_style_map = {'diffusion': 'solid', 'regression': 'dot'}
    model_descriptions = []
    stats_list = ['spectra', 'distribution']
    unique_data = [stats for stats in stats_list if stats in combined_spectra_df['Data'].unique()]
    subset_df = combined_spectra_df[combined_spectra_df['Data'].isin(unique_data)]
    num_subplots = len(unique_data)
    
    fig = make_subplots(rows=2, cols=1, subplot_titles=unique_data, horizontal_spacing=0.15)
    target_df = subset_df[(subset_df['Data'] == 'spectra') & 
                          (subset_df['Time'] == selected_time) & 
                          (subset_df['Model'] == selected_models[0]) & 
                          (subset_df['Type'] == 'target') & 
                          (subset_df['Level'] == selected_level) & 
                          (subset_df['Variable'] == selected_variable)]
    if not target_df.empty:
        fig.add_trace(
            go.Scatter(
                x=target_df['Frequency'], 
                y=target_df['Value'], 
                mode='lines',
                name='target',
                line=dict(color='#D3D3D3', width=4)
            ),
            row=1, col=1
        )
    
    
    for model in selected_models:
        for _type in selected_types:
            filtered_df = subset_df[(subset_df['Data'] == 'spectra') & 
                                    (subset_df['Time'] == selected_time) & 
                                    (subset_df['Model'] == model) & 
                                    (subset_df['Type'] == _type) & 
                                    (subset_df['Level'] == selected_level) & 
                                    (subset_df['Variable'] == selected_variable)]

            filtered_df = filtered_df[filtered_df['Level'] == selected_level]
            
            if not filtered_df.empty:
                if selected_variable=="Ek":
                    freq = filtered_df['Frequency']
                    y_dotted = freq**(-5/3)
                    fig.add_trace(
                        go.Scatter(
                            x=freq,
                            y=y_dotted,
                            mode='lines',
                            name=f'{model} - {_type} (k^-(5/3))',
                            line=dict(color='black', dash='dot', width=0.5),
                            showlegend=False
                        ),
                        row=1, col=1
                    )

                fig.add_trace(
                    go.Scatter(
                        x=filtered_df['Frequency'], 
                        y=filtered_df['Value'], 
                        mode='lines',
                        name=f'{model} - {_type}',
                        line=dict(color=color_map[model], dash=line_style_map[_type]),
                        showlegend = True
                    ),
                    row=1, col=1
                )
    
    spectra_y_lower = np.log10(0.05*np.nanpercentile(filtered_df['Value'], 1))
    spectra_y_upper = np.log10(20*np.nanpercentile(filtered_df['Value'], 99))
        
    stats_list = ['distribution']
    unique_data = [stats for stats in stats_list if stats in combined_distribution_df['Data'].unique()]
    subset_df = combined_distribution_df[combined_distribution_df['Data'].isin(unique_data)]
    num_subplots = len(unique_data)
    target_df = subset_df[(subset_df['Data'] == 'distribution') & 
                          (subset_df['Time'] == selected_time) & 
                          (subset_df['Model'] == selected_models[0]) & 
                          (subset_df['Type'] == 'target') & 
                          (subset_df['Level'] == selected_level) & 
                          (subset_df['Variable'] == selected_variable)]
    if not target_df.empty:
        fig.add_trace(
            go.Scatter(
                x=target_df['Bin'], 
                y=target_df['Value'], 
                mode='lines',
                name='target',
                line=dict(color='#D3D3D3', width=4),
                showlegend = False
            ),
            row=2, col=1
        )
        
    for model in selected_models:
        try:
            model_descriptions.append(model + ": " + model_registry['models'][model]['description'])
        except KeyError:
            model_descriptions.append(model + ": is not in registry, if new model was added update registry.")
        for _type in selected_types:
            filtered_df = subset_df[(subset_df['Data'] == 'distribution') & 
                                    (subset_df['Time'] == selected_time) & 
                                    (subset_df['Model'] == model) & 
                                    (subset_df['Type'] == _type) & 
                                    (subset_df['Variable'] == selected_variable)]
            
            filtered_df = filtered_df[filtered_df['Level'] == selected_level]
            
            if not filtered_df.empty:
                fig.add_trace(
                    go.Scatter(
                        x=filtered_df['Bin'], 
                        y=filtered_df['Value'], 
                        mode='lines',
                        name=f'{model} - {_type}',
                        line=dict(color=color_map[model], dash=line_style_map[_type]), 
                        showlegend = False
                    ),
                    row=2, col=1
                )
    
    fig.update_yaxes(range=[0.9*spectra_y_lower, 1.1*spectra_y_upper], type='log', row=1, col=1)

    fig.update_traces(fill='none', line=dict(shape='linear'))
    fig.update_xaxes(type='log', row=1, col=1)
    fig.update_xaxes(title_text='frequency (1/km)', row=1, col=1)
    fig.update_xaxes(title_text='values', row=2, col=1)
    fig.update_yaxes(title_text='log PDF', row=2, col=1)
    fig.update_yaxes(title_text='power spectra', row=1, col=1)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', row=1, col=1)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', row=2, col=1)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', row=1, col=1)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', row=2, col=1)
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=False)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=False)
    fig.update_yaxes(type='log', row=1, col=1)
        
    fig.update_layout(hovermode='x unified')
    
    fig_height = 700
    fig_width = fig_height*1.5
    
    fig.update_layout(width=fig_width, height=fig_height, 
                      margin=dict(l=100, r=300))
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(legend=dict(
        x=2.0, y=1.0, xanchor='center', yanchor='top', orientation='v'
    ))
    model_descriptions_elements = []
    for full_description in model_descriptions:
        model_name, description_text = full_description.split(":", 1)  # Split at first colon
        model_descriptions_elements.append(html.P([
            html.Strong(model_name + ": "),
            description_text.strip()
        ]))

    description_container = html.Div(
        children=model_descriptions_elements,
        style={
            'marginTop': '10px',
            'marginBottom': '10px',
            'padding': '10px'
        }
    )
    return dcc.Graph(figure=fig), description_container



@app.callback(
    [Output('cases-tab', 'children'),
     Output('model-description-container3', 'children')],
    [Input('model-dropdown-tab3', 'value'),
     Input('case-dropdown-tab3', 'value'),
     Input('time-slider-tab3', 'value'),
     Input('channel-dropdown-tab3', 'value')]
)


def update_cases_panels(selected_models, selected_case, selected_time, selected_channel):
    with open('../config/registry.json') as f:
        model_registry = json.load(f)
    #check the analysis folder for more registries of the format "sweep_*_registry.json" and append them to the model_registry
    for file in os.listdir('../analysis'):
        if file.startswith('sweep_') and file.endswith('_registry.json'):
            with open(f'../analysis/{file}') as f:
                sweep_registry = json.load(f)

            for model in sweep_registry['models']:
                model_registry['models'][model] = sweep_registry['models'][model]


    model_descriptions = []
    nrows=len(selected_models)
    ncols=4
    selected_channel_str = selected_channel[0] if isinstance(selected_channel, list) else selected_channel
    selected_case_str = selected_case[0] if isinstance(selected_case, list) else selected_case
    
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(nrows, ncols, figure=fig, width_ratios=[1, 1, 1, 1.2])
    for i, model in enumerate(selected_models):
        description = model + ": " + model_registry['models'][model]['description']
        model_descriptions.append(description)

        ds = xr.open_dataset(os.path.join(data_path, (selected_case_str + "__" + model + '.nc')))
        vmin, vmax = np.min(ds[selected_channel_str].values), np.max(ds[selected_channel_str].values)
        
        datasets = ["forecast", "target", "diffusion", "regression"] 
        for j, data_type in enumerate(datasets):
            if data_type=="forecast":
                print(os.path.join(forecast_path, (selected_case_str + ".zarr")))
                hrrr_baseline = xr.open_dataset(os.path.join(forecast_path, (selected_case_str + ".zarr")))
                ax = fig.add_subplot(gs[i, j], projection=ccrs.PlateCarree())
                if selected_channel == 'refc':
                    channel_data = hrrr_baseline['REFC'].isel(time=selected_time + 1).values
                    levels = np.linspace(vmin, vmax, num=100)
                    norm, cmap = ctables.registry.get_with_steps('NWSReflectivity', -0, 5)
                    c = ax.contourf(hrrr_baseline['longitude'].values, hrrr_baseline['latitude'].values, channel_data, cmap=cmap, levels=levels, norm=norm)
                    ax.add_feature(cfeature.STATES.with_scale('50m'))
                    ax.add_feature(cfeature.LAND.with_scale('50m'))
                    ax.add_feature(cfeature.OCEAN.with_scale('50m'))

                    ax.set_title(f"HRRR forecast, valid {str(hrrr_baseline['time'].values[selected_time])[:19]}  \n lead time: {selected_time + 1} hours", fontsize=10)
                else:
                    pass
            else:
                ax = fig.add_subplot(gs[i, j], projection=ccrs.PlateCarree())
                channel_data = ds.sel(dataset=data_type).isel(time=selected_time)[selected_channel_str].values
                levels = np.linspace(vmin, vmax, num=100)
                if selected_channel_str=='refc':
                    norm, cmap = ctables.registry.get_with_steps('NWSReflectivity', -0, 5)
                elif selected_channel_str=='relative_humidity':
                    colors = plt.cm.Blues(np.linspace(0, 1, 10))
                    cmap = LinearSegmentedColormap.from_list('custom_blues', colors, N=10)
                    norm = None
                elif selected_channel_str=='t_comb':
                    colors = plt.cm.YlOrRd(np.linspace(0, 1, 10))
                    cmap = LinearSegmentedColormap.from_list('custom_blues', colors, N=10)
                    norm = None
                else:
                    cmap=plt.cm.seismic
                    norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
                c = ax.contourf(ds.lon.values, ds.lat.values, channel_data, cmap=cmap, levels=levels, norm=norm)    
                ax.add_feature(cfeature.STATES.with_scale('50m'))
                ax.add_feature(cfeature.LAND.with_scale('50m'))
                ax.add_feature(cfeature.OCEAN.with_scale('50m'))

            gl = ax.gridlines(draw_labels=False)
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.xlabel_style = {'size': 7}
            gl.ylabel_style = {'size': 7}            
            gl = ax.gridlines(draw_labels=True)
            if data_type ==  "diffusion":
                ax.set_title(data_type + "\n" + model, fontsize=10)
            else:
                ax.set_title(data_type, fontsize=10)

            
            if  i < nrows-1:
                gl.bottom_labels = False

            gl.top_labels = False
            gl.right_labels = False

        
            if data_type == "regression":
                cbar = fig.colorbar(c, ax=ax, orientation='vertical', shrink=0.3, pad=0.04)
                cbar.ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))
                for label in cbar.ax.get_yticklabels():
                    label.set_fontsize(label.get_fontsize() * 0.7)
            if selected_channel_str=='refc':
                if data_type != "forecast":
                    ax.yaxis.set_visible(False)
                    gl.left_labels = False
            else:
                if data_type != "target":
                    ax.yaxis.set_visible(False)
                    gl.left_labels = False

    buf = BytesIO()
    fig.tight_layout()
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95, left=0.05)
    fig.savefig(buf, format="png")
    plt.close(fig)
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    
    
    figure = html.Img(
        src="data:image/png;base64,{}".format(data),
        style={'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto',  'margin-top': '0px', 'margin-bottom': 'auto'}
    )

    model_descriptions_elements = []
    for full_description in model_descriptions:
        model_name, description_text = full_description.split(":", 1)  # Split at first colon
        model_descriptions_elements.append(html.P([
            html.Strong(model_name + ": "),  # The model name in bold
            description_text.strip()  # The description text in normal font
        ]))

    description_container = html.Div(
        children=model_descriptions_elements,
        style={
            'marginTop': '10px',
            'marginBottom': '10px',
            'padding': '10px'
        }
    )
    return figure, description_container


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')
