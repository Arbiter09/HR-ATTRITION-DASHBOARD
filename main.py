import dash
from dash import dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from dash.exceptions import PreventUpdate
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load dataset
df = pd.read_csv("hr_data.csv")
df['Attrition_Flag'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
df['EducationLevel'] = df['Education'].map({
    1: "High School",
    2: "Associate's Degree",
    3: "Bachelor's Degree",
    4: "Master's Degree",
    5: "Doctoral Degree"
})

# Initialize app with Bootstrap theme and enable dynamic callbacks
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server

# Color palette
colors = {
    'background': '#f9f6f2',
    'card_bg': '#ffffff',
    'primary': '#3a3f44',
    'secondary': '#e89c61',
    'accent': '#d86e3a',
    'text': '#333333',
    'light_text': '#666666',
    'border': '#e0e0e0',
    'highlight': '#ff8c42',
    'retention': '#4a4a4a',
    'attrition': '#ff8c42'
}

# Common layout settings (margin removed to avoid duplicates)
common_layout = dict(
    font_family="Segoe UI, sans-serif",
    font_size=10,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=240
)


# Custom styles HTML wrapper
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>HR Attrition Dashboard</title>
        {%favicon%}
        {%css%}
        <style>
            body {background-color: #f9f6f2; font-family: 'Segoe UI', sans-serif; color: #333; margin:0; padding:0;}
            .dashboard-container {display:flex; height:100vh; overflow:hidden;}
            .sidebar {background:white; box-shadow:2px 0 5px rgba(0,0,0,0.1); padding:15px; width:280px; overflow-y:auto;}
            .company-logo {font-weight:700; text-align:center; padding:10px 0; border-bottom:1px solid #e0e0e0; margin-bottom:15px;}
            .dashboard-title {font-weight:700; letter-spacing:2px; color:#3a3f44; font-size:1.5rem; margin:0 0 15px 0;}
            .section-header {font-size:14px; font-weight:600; color:#555; text-transform:uppercase; margin:10px 0 5px;}
            .filter-label {font-size:12px; font-weight:500; color:#555; margin-bottom:3px;}
            .reset-btn {width:100%; margin-top:15px; padding:5px 10px; background:#6c757d; color:white; border:none; border-radius:4px; cursor:pointer;}
            .reset-btn:hover {background:#5a6268;}
            .content-container {padding:15px; flex-grow:1; overflow-y:auto; height:100vh;}
            .kpi-container {background:white; border-radius:5px; box-shadow:0 1px 3px rgba(0,0,0,0.1); padding:10px; text-align:center; height:100%;}
            .kpi-title {font-size:12px; color:#666; margin-bottom:0;}
            .kpi-value {font-size:20px; font-weight:700; color:#333;}
            .chart-container {height:250px;}
            .reduced-margins .js-plotly-plot {margin:0;}
            .filtered-highlight {border: 2px solid #ff8c42; border-radius: 5px;}
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

# Sidebar for filters
sidebar = html.Div([
    html.Div("EVEREST GROUP", className="company-logo"),
    html.H1("HR DASHBOARD", className="dashboard-title"),
    html.Div("FILTERS", className="section-header"),
    html.Div([
        html.Div("Department", className="filter-label"),
        dcc.Dropdown(
            id="department-filter",
            options=[{"label":"All Departments","value":"all"}] + [{"label":d,"value":d} for d in sorted(df["Department"].unique())],
            value="all", clearable=False
        ),
        html.Div("Gender", className="filter-label mt-2"),
        dcc.Dropdown(
            id="gender-filter",
            options=[{"label":"All Genders","value":"all"}] + [{"label":g,"value":g} for g in sorted(df["Gender"].unique())],
            value="all", clearable=False
        ),
        html.Div("Overtime", className="filter-label mt-2"),
        dcc.Dropdown(
            id="overtime-filter",
            options=[{"label":"All Overtime Status","value":"all"}] + [{"label":o,"value":o} for o in sorted(df["OverTime"].unique())],
            value="all", clearable=False
        ),
        html.Div("Education", className="filter-label mt-2"),
        dcc.Dropdown(
            id="education-filter",
            options=[{"label":"All Education Levels","value":"all"}] + [{"label":e,"value":e} for e in sorted(df["EducationLevel"].dropna().unique())],
            value="all", clearable=False
        ),
        html.Div("Job Level", className="filter-label mt-2"),
        dcc.Dropdown(
            id="joblevel-filter",
            options=[{"label":"All Job Levels","value":"all"}] + [{"label":f"Level {lvl}","value":lvl} for lvl in sorted(df["JobLevel"].unique())],
            value="all", clearable=False
        ),
        html.Button("Reset Filters", id="reset-filters", className="reset-btn")
    ]),
    html.Div(id="active-filters", className="mt-3", style={"font-size": "12px", "color": colors['accent']})
], className="sidebar")

# Main content area
content = html.Div([
    dbc.Row([
        dbc.Col(html.Div([html.Div("Attrition Rate", className="kpi-title"), html.Div(id="attrition-rate", className="kpi-value")], className="kpi-container"), width=3),
        dbc.Col(html.Div([html.Div("Total Employees", className="kpi-title"), html.Div(id="total-employees", className="kpi-value")], className="kpi-container"), width=3),
        dbc.Col(html.Div([html.Div("Avg Monthly Income", className="kpi-title"), html.Div(id="avg-income", className="kpi-value")], className="kpi-container"), width=3),
        dbc.Col(html.Div([html.Div("Avg Job Satisfaction", className="kpi-title"), html.Div(id="avg-satisfaction", className="kpi-value")], className="kpi-container"), width=3)
    ], className="mb-3 g-2"),
    
    # Container for all visualizations
    html.Div(id="visualization-container")
], className="content-container")

app.layout = html.Div([
    sidebar, 
    content,
    # Hidden div to store click data for filter coordination
    html.Div(id='click-data-store', style={'display': 'none'})
], className="dashboard-container")

# Function to apply all filters
def apply_filters(dept, gender, overtime, education, joblevel):
    dff = df.copy()
    if dept != "all": dff = dff[dff['Department'] == dept]
    if gender != "all": dff = dff[dff['Gender'] == gender]
    if overtime != "all": dff = dff[dff['OverTime'] == overtime]
    if education != "all": dff = dff[dff['EducationLevel'] == education]
    if joblevel != "all": dff = dff[dff['JobLevel'] == joblevel]
    return dff

# KPI Callback
@app.callback(
    [Output('attrition-rate','children'), 
     Output('total-employees','children'), 
     Output('avg-income','children'), 
     Output('avg-satisfaction','children'),
     Output('active-filters', 'children')],
    [Input('department-filter','value'), 
     Input('gender-filter','value'), 
     Input('overtime-filter','value'), 
     Input('education-filter','value'), 
     Input('joblevel-filter','value')]
)
def update_kpis(dept, gender, overtime, education, joblevel):
    dff = apply_filters(dept, gender, overtime, education, joblevel)
    
    # KPIs
    attr = f"{round(dff['Attrition_Flag'].mean()*100,2)}%" if not dff.empty else "0%"
    tot = f"{dff.shape[0]}"
    inc = f"${int(dff['MonthlyIncome'].mean()):,}" if 'MonthlyIncome' in dff and not dff.empty else "N/A"
    sat = f"{round(dff['JobSatisfaction'].mean(),1)}/5" if 'JobSatisfaction' in dff and not dff.empty else "N/A"
    
    # Active filters display
    active_filter_text = []
    if dept != "all": active_filter_text.append(f"Department: {dept}")
    if gender != "all": active_filter_text.append(f"Gender: {gender}")
    if overtime != "all": active_filter_text.append(f"Overtime: {overtime}")
    if education != "all": active_filter_text.append(f"Education: {education}")
    if joblevel != "all": active_filter_text.append(f"Job Level: {joblevel}")
    
    active_filters = html.Div([
        html.Div("Active Filters:", className="section-header mt-3 mb-1"),
        html.Div(active_filter_text if active_filter_text else "No active filters")
    ]) if active_filter_text else ""
    
    return attr, tot, inc, sat, active_filters

# Chart creation functions
def create_gender_pie(filtered_df):
    gender_counts = filtered_df['Gender'].value_counts().reset_index(name='Count').rename(columns={'index':'Gender'})
    
    # Add percentage to hover data
    total = gender_counts['Count'].sum()
    gender_counts['Percentage'] = (gender_counts['Count'] / total * 100).round(1).astype(str) + '%'
    
    fig = px.pie(
        gender_counts,
        names='Gender',
        values='Count',
        hole=0.3,
        title='Gender Ratio',
        color_discrete_sequence=[colors['retention'], colors['attrition']]
    )
    
    # Add custom hover template
    fig.update_traces(
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{customdata[0]}<extra></extra>',
        customdata=gender_counts[['Percentage']]
    )
    
    fig.update_layout(**common_layout)
    return fig

def create_education_bar(filtered_df):
    edu_counts = filtered_df.groupby(['EducationLevel', 'Attrition']).size().reset_index(name='Count')
    
    # Sort by education level
    education_order = ["High School", "Associate's Degree", "Bachelor's Degree", "Master's Degree", "Doctoral Degree"]
    edu_counts['EducationLevel'] = pd.Categorical(edu_counts['EducationLevel'], categories=education_order, ordered=True)
    edu_counts = edu_counts.sort_values('EducationLevel')
    
    fig = px.bar(
        edu_counts,
        x='EducationLevel',
        y='Count',
        color='Attrition',
        title='Employees by Education Level',
        barmode='group',
        color_discrete_sequence=[colors['retention'], colors['attrition']]
    )
    
    fig.update_traces(customdata=edu_counts['EducationLevel'])
    fig.update_layout(**common_layout, xaxis_tickangle=45, clickmode='event+select')
    
    return fig

def create_bubble_chart(filtered_df):
    dept_metrics = filtered_df.groupby('Department').agg({
        'Attrition_Flag': 'mean',
        'JobSatisfaction': 'mean',
        'MonthlyIncome': 'mean',
        'EmployeeCount': 'count'
    }).reset_index()
    
    dept_metrics['Attrition_Percentage'] = (dept_metrics['Attrition_Flag'] * 100).round(1)
    
    fig = px.scatter(
        dept_metrics,
        x='JobSatisfaction',
        y='Attrition_Percentage',
        size='EmployeeCount',
        color='MonthlyIncome',
        hover_name='Department',
        labels={
            'JobSatisfaction': 'Avg Job Satisfaction',
            'Attrition_Percentage': 'Attrition Rate (%)',
            'EmployeeCount': 'Employee Count',
            'MonthlyIncome': 'Avg Monthly Income'
        },
        title='Department Risk Analysis',
        color_continuous_scale=px.colors.sequential.Viridis
    )
    
    # Add customdata for interactivity
    fig.update_traces(
        customdata=dept_metrics['Department'],
        hovertemplate='<b>%{hovertext}</b><br>Attrition Rate: %{y}%<br>Job Satisfaction: %{x}/5<br>Employees: %{marker.size}<br>Avg Income: $%{marker.color:,.0f}<extra></extra>'
    )
    
    fig.update_layout(
        **common_layout,
        coloraxis_colorbar=dict(title='Avg Monthly Income'),
        clickmode='event+select'
    )
    
    return fig

def create_age_distribution(filtered_df):
    fig = px.histogram(
        filtered_df, 
        x='Age', 
        color='Attrition', 
        nbins=20, 
        title='Age Distribution by Attrition', 
        color_discrete_sequence=[colors['retention'], colors['attrition']]
    )
    
    # Add interactivity - selecting an age range
    fig.update_layout(**common_layout, clickmode='event+select')
    
    # Add custom hover data showing count and percentage
    fig.update_traces(
        hovertemplate='Age: %{x}<br>Count: %{y}<br>Attrition: %{fullData.name}<extra></extra>'
    )
    
    return fig

def create_job_role_chart(filtered_df):
    role_counts = filtered_df.groupby(['JobRole','Attrition']).size().reset_index(name='Count')
    
    fig = px.bar(
        role_counts, 
        x='JobRole', 
        y='Count', 
        color='Attrition', 
        barmode='group', 
        title='Job Role Distribution by Attrition', 
        color_discrete_sequence=[colors['retention'], colors['attrition']]
    )
    
    # Add customdata for interactivity
    fig.update_traces(
        customdata=role_counts['JobRole'],
        hovertemplate='<b>%{x}</b><br>Count: %{y}<br>Attrition: %{fullData.name}<extra></extra>'
    )
    
    fig.update_layout(
        **common_layout, 
        xaxis_tickangle=45, 
        xaxis={'categoryorder':'total descending'},
        clickmode='event+select'
    )
    
    return fig

def create_treemap(filtered_df):
    tree_data = filtered_df.groupby(['Department','JobRole','Attrition']).size().reset_index(name='Count')
    fig = px.treemap(
        tree_data,
        path=['Department','JobRole','Attrition'],
        values='Count',
        color='Attrition',
        title='Hierarchical View of Attrition',
        color_discrete_map={'Yes': colors['attrition'], 'No': colors['retention']}
    )
    fig.update_traces(hovertemplate='<b>%{label}</b><br>Count: %{value}<extra></extra>')
    fig.update_layout(**common_layout)
    return fig
    
    # Add customdata for interactivity
    fig.update_traces(
        hovertemplate='<b>%{label}</b><br>Count: %{value}<extra></extra>'
    )
    
    fig.update_layout(**common_layout, margin=dict(l=5, r=5, t=30, b=5))
    
    return fig

def create_pca_analysis(filtered_df):
    available_cols = [c for c in ['Age','MonthlyIncome','TotalWorkingYears','YearsAtCompany','YearsInCurrentRole'] if c in filtered_df]
    
    if len(available_cols) >= 2 and not filtered_df.empty:
        num_df = filtered_df[available_cols].fillna(filtered_df[available_cols].mean())
        scaled = StandardScaler().fit_transform(num_df)
        pca = PCA(n_components=2)
        pca_res = pca.fit_transform(scaled)
        
        pca_df = pd.DataFrame({
            'PCA1': pca_res[:,0],
            'PCA2': pca_res[:,1],
            'Attrition': filtered_df['Attrition'],
            'Department': filtered_df['Department'],
            'Gender': filtered_df['Gender'],
            'JobRole': filtered_df['JobRole']
        })
        
        fig = px.scatter(
            pca_df, 
            x='PCA1', 
            y='PCA2', 
            color='Attrition', 
            title=f'PCA Analysis (Var: {pca.explained_variance_ratio_.sum():.2%})', 
            color_discrete_sequence=[colors['retention'], colors['attrition']]
        )
        
        # Add hover data
        fig.update_traces(
            customdata=pca_df[['Department', 'Gender', 'JobRole']],
            hovertemplate='<b>Attrition: %{marker.color}</b><br>Department: %{customdata[0]}<br>Gender: %{customdata[1]}<br>Role: %{customdata[2]}<extra></extra>'
        )
        
        fig.update_layout(**common_layout)
    else:
        fig = go.Figure()
        fig.add_annotation(text='Insufficient data for PCA', xref='paper', yref='paper', x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title='PCA Analysis (Unavailable)', **common_layout)
    
    return fig

def create_radar_chart(filtered_df):
    metrics = [
        'WorkLifeBalance', 
        'JobSatisfaction', 
        'EnvironmentSatisfaction', 
        'JobInvolvement', 
        'RelationshipSatisfaction'
    ]
    
    available_metrics = [m for m in metrics if m in filtered_df.columns]
    
    if not available_metrics or filtered_df.empty:
        fig = go.Figure()
        fig.add_annotation(text='Data unavailable for radar chart', xref='paper', yref='paper', x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title='Satisfaction Metrics (Unavailable)', **common_layout)
        return fig
    
    # Calculate averages by attrition status
    radar_data = filtered_df.groupby('Attrition')[available_metrics].mean().reset_index()
    
    # Create radar chart
    fig = go.Figure()
    
    for i, attrition in enumerate(radar_data['Attrition']):
        color = colors['attrition'] if attrition == 'Yes' else colors['retention']
        fig.add_trace(go.Scatterpolar(
            r=radar_data.loc[i, available_metrics].values,
            theta=available_metrics,
            fill='toself',
            name=f'Attrition: {attrition}',
            line_color=color,
            fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)'
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5]
            )
        ),
        title='Satisfaction Metrics by Attrition',
        **common_layout
    )
    
    return fig

def create_salary_box_plot(filtered_df):
    if 'MonthlyIncome' not in filtered_df.columns or filtered_df.empty:
        fig = go.Figure()
        fig.add_annotation(text='Salary data unavailable', xref='paper', yref='paper', x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title='Salary Distribution (Unavailable)', **common_layout)
        return fig
    
    fig = px.box(
        filtered_df,
        x='Department',
        y='MonthlyIncome',
        color='Attrition',
        title='Salary Distribution by Department',
        color_discrete_sequence=[colors['retention'], colors['attrition']]
    )
    
    # Add customdata for interactivity
    fig.update_traces(
        hovertemplate='<b>%{x}</b><br>Median: $%{median:,.0f}<br>Mean: $%{mean:,.0f}<br>Q1: $%{q1:,.0f}<br>Q3: $%{q3:,.0f}<extra></extra>'
    )
    
    fig.update_layout(
        **common_layout,
        xaxis_tickangle=45,
        yaxis_title='Monthly Income ($)',
        clickmode='event+select'
    )
    
    return fig

# Main visualization callback with interactivity
@app.callback(
    Output('visualization-container', 'children'),
    [Input('department-filter', 'value'), 
     Input('gender-filter', 'value'), 
     Input('overtime-filter', 'value'), 
     Input('education-filter', 'value'), 
     Input('joblevel-filter', 'value')]
)
def update_visualizations(dept, gender, overtime, education, joblevel):
    try:
        dff = apply_filters(dept, gender, overtime, education, joblevel)
        
        # Generate all visualizations
        gender_fig = create_gender_pie(dff)
        bubble_fig = create_bubble_chart(dff)
        education_fig = create_education_bar(dff)
        age_fig = create_age_distribution(dff)
        role_fig = create_job_role_chart(dff)
        treemap_fig = create_treemap(dff)
        pca_fig = create_pca_analysis(dff)
        radar_fig = create_radar_chart(dff)
        box_fig = create_salary_box_plot(dff)
        
        # Style classes for highlighted filters
        gender_class = "card chart-container reduced-margins filtered-highlight" if gender != "all" else "card chart-container reduced-margins"
        dept_class = "card chart-container reduced-margins filtered-highlight" if dept != "all" else "card chart-container reduced-margins"
        edu_class = "card chart-container reduced-margins filtered-highlight" if education != "all" else "card chart-container reduced-margins"
        
        return [
            dbc.Row([
                dbc.Col(dcc.Graph(id='bubble-chart', figure=bubble_fig, className=dept_class), lg=6, md=12),
                dbc.Col(dcc.Graph(id='gender-pie', figure=gender_fig, className=gender_class), lg=6, md=12)
            ], className='g-2'),
            dbc.Row([
                dbc.Col(dcc.Graph(id='education-bar', figure=education_fig, className=edu_class), lg=6, md=12),
                dbc.Col(dcc.Graph(id='age-distribution', figure=age_fig, className="card chart-container reduced-margins"), lg=6, md=12)
            ], className='g-2'),
            dbc.Row([
                dbc.Col(dcc.Graph(id='job-role-chart', figure=role_fig, className="card chart-container reduced-margins"), lg=6, md=12),
                dbc.Col(dcc.Graph(id='treemap-chart', figure=treemap_fig, className="card chart-container reduced-margins"), lg=6, md=12)
            ], className='g-2'),
            dbc.Row([
                dbc.Col(dcc.Graph(id='pca-chart', figure=pca_fig, className="card chart-container reduced-margins"), lg=6, md=12),
                dbc.Col(dcc.Graph(id='radar-chart', figure=radar_fig, className="card chart-container reduced-margins"), lg=6, md=12)
            ], className='g-2'),
            dbc.Row([
                dbc.Col(dcc.Graph(id='salary-box-plot', figure=box_fig, className="card chart-container reduced-margins"), lg=12, md=12)
            ], className='g-2')
        ]
    except Exception as e:
        return html.Div(f"Error processing data: {str(e)}", style={"color":"red","padding":"20px"})

# Callback for storing clicked data
@app.callback(
    Output('click-data-store', 'children'),
    [Input('gender-pie', 'clickData'),
     Input('bubble-chart', 'clickData'),
     Input('education-bar', 'clickData'),
     Input('treemap-chart', 'clickData'),
     Input('salary-box-plot', 'clickData'),
     Input('pca-chart', 'clickData')],
    [State('click-data-store', 'children')]
)
def store_click_data(gender_click, bubble_click, edu_click, treemap_click, salary_click, pca_click, current_data):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    click_data = None
    
    if trigger_id == 'gender-pie' and gender_click:
        click_data = {'type': 'gender', 'value': gender_click['points'][0]['label']}
    elif trigger_id == 'bubble-chart' and bubble_click:
        click_data = {'type': 'department', 'value': bubble_click['points'][0]['customdata']}
    elif trigger_id == 'education-bar' and edu_click:
        click_data = {'type': 'education', 'value': edu_click['points'][0]['customdata']}
    elif trigger_id == 'treemap-chart' and treemap_click:
        if treemap_click['points'][0]['currentPath'].count('/') == 0:
            click_data = {'type': 'department', 'value': treemap_click['points'][0]['label']}
    elif trigger_id == 'salary-box-plot' and salary_click:
        click_data = {'type': 'department', 'value': salary_click['points'][0]['x']}
    elif trigger_id == 'pca-chart' and pca_click:
        try:
            click_data = {'type': 'department', 'value': pca_click['points'][0]['customdata'][0]}
        except (IndexError, KeyError):
            pass
    
    return dash.no_update if click_data is None else str(click_data)

# Unified callback for filters based on click data store
@app.callback(
    [Output('department-filter', 'value'),
     Output('gender-filter', 'value'),
     Output('education-filter', 'value'),
     Output('joblevel-filter', 'value'),
     Output('overtime-filter', 'value')],
    [Input('click-data-store', 'children'),
     Input('reset-filters', 'n_clicks')]
)
def update_filters_from_clicks(click_data_str, reset_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Handle reset button
    if trigger_id == 'reset-filters' and reset_clicks:
        return 'all', 'all', 'all', 'all', 'all'
    
    # Handle chart clicks
    if trigger_id == 'click-data-store' and click_data_str:
        try:
            import ast
            click_data = ast.literal_eval(click_data_str)
            filter_type = click_data.get('type')
            filter_value = click_data.get('value')
            
            if filter_type == 'department':
                return filter_value, no_update, no_update, no_update, no_update
            elif filter_type == 'gender':
                return no_update, filter_value, no_update, no_update, no_update
            elif filter_type == 'education':
                return no_update, no_update, filter_value, no_update, no_update
        except:
            pass
    
    return no_update, no_update, no_update, no_update, no_update

if __name__ == "__main__":
    app.run(debug=True)