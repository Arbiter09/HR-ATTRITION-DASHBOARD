import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from dash.exceptions import PreventUpdate
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ==============================
# Data loading & transformation
# ==============================
df = pd.read_csv("hr_data.csv")
df["Attrition_Flag"] = df["Attrition"].apply(lambda x: 1 if x == "Yes" else 0)
df["EducationLevel"] = df["Education"].map(
    {
        1: "High School",
        2: "Associate's Degree",
        3: "Bachelor's Degree",
        4: "Master's Degree",
        5: "Doctoral Degree",
    }
)

# ================
# App & styling
# ================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "HR Attrition Dashboard"

colors = {
    "background": "#f9f6f2",
    "card_bg": "#ffffff",
    "primary": "#3a3f44",
    "secondary": "#e89c61",
    "accent": "#d86e3a",
    "text": "#333333",
    "light_text": "#666666",
    "border": "#e0e0e0",
    "highlight": "#ff8c42",
    "retention": "#4a4a4a",
    "attrition": "#ff8c42",
}


# Inject super‑compact CSS straight into index_string
app.index_string = f"""
<!DOCTYPE html>
<html>
<head>
    {{%metas%}}
    <title>HR Attrition Dashboard</title>
    {{%favicon%}}
    {{%css%}}
    <style>
        body {{
            background-color:{colors['background']};
            font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;
            color:{colors['text']};
            margin:0;padding:0;overflow:hidden;
        }}
        .dashboard-title{{font-weight:700;letter-spacing:2px;color:{colors['primary']};
            padding:5px 0;margin-bottom:0;border-bottom:1px solid {colors['border']};font-size:1.5rem;}}
        .dashboard-container{{padding:0 10px;height:100vh;overflow:hidden;}}
        .section-header{{font-size:11px;font-weight:600;color:#555;text-transform:uppercase;margin:2px 0;}}
        .card{{background:white;border-radius:3px;box-shadow:0 1px 3px rgba(0,0,0,.1);
            margin-bottom:3px;border:none;padding:0;}}
        .filter-container{{background:white;border-radius:3px;box-shadow:0 1px 3px rgba(0,0,0,.1);
            padding:5px 10px;margin-bottom:3px;}}
        .filter-label{{font-size:10px;font-weight:500;color:#555;margin-bottom:1px;}}
        .reset-btn,.clear-selections-btn{{background:#6c757d;color:white;border:none;padding:4px 8px;border-radius:3px;
            cursor:pointer;transition:background-color .3s;font-size:11px;}}
        .reset-btn:hover{{background:#5a6268;}}
        .clear-selections-btn{{background:{colors['highlight']};}}
        .clear-selections-btn:hover{{background:#e07b32;}}
        .kpi-container{{background:white;border-radius:3px;box-shadow:0 1px 3px rgba(0,0,0,.1);
            padding:6px;height:100%;}}
        .kpi-title{{font-size:10px;color:{colors['light_text']};margin-bottom:0;}}
        .kpi-value{{font-size:16px;font-weight:700;line-height:1;}}
        .company-logo{{font-weight:700;text-align:right;color:{colors['text']};padding-top:5px;font-size:1rem;}}
        /* plotly sizing tweaks */
        .graph-container{{margin:0!important;padding:0!important;height:32vh!important;width:100%!important;}}
        .row{{margin-left:-3px!important;margin-right:-3px!important;margin-bottom:3px!important;}}
        .col,[class*="col-"]{{padding-right:3px!important;padding-left:3px!important;}}
        .selection-badge{{font-size:10px;color:white;background:{colors['highlight']};
            border-radius:10px;padding:2px 6px;margin-left:5px;display:inline-block;}}
    </style>
</head>
<body>
    {{%app_entry%}}
    <footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer>
</body>
</html>
"""

# =========
# Layout
# =========
app.layout = dbc.Container(
    [
        dcc.Store(id="selected-data-store", data={}),

        # Title Row
       dbc.Row(
    [
        dbc.Col(
            html.H1("HR ATTRITION DASHBOARD", className="dashboard-title"),
            width="auto",            # shrink‑to‑fit column
        ),
    ],
    justify="center",              # flex‑box horizontal centering
    className="g-0",
),


        # Filter Row
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div("FILTERS", className="section-header"),
                        html.Div(
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Div("Department", className="filter-label"),
                                            dcc.Dropdown(
                                                id="department-filter",
                                                options=[{"label": "All Departments", "value": "all"}]
                                                + [{"label": i, "value": i} for i in sorted(df["Department"].unique())],
                                                value="all",
                                                clearable=False,
                                            ),
                                        ],
                                        width=3,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Div("Gender", className="filter-label"),
                                            dcc.Dropdown(
                                                id="gender-filter",
                                                options=[{"label": "All Genders", "value": "all"}]
                                                + [{"label": i, "value": i} for i in sorted(df["Gender"].unique())],
                                                value="all",
                                                clearable=False,
                                            ),
                                        ],
                                        width=2,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Div("Overtime", className="filter-label"),
                                            dcc.Dropdown(
                                                id="overtime-filter",
                                                options=[{"label": "All Overtime Status", "value": "all"}]
                                                + [{"label": i, "value": i} for i in sorted(df["OverTime"].unique())],
                                                value="all",
                                                clearable=False,
                                            ),
                                        ],
                                        width=2,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Div("Education", className="filter-label"),
                                            dcc.Dropdown(
                                                id="education-filter",
                                                options=[{"label": "All Education Levels", "value": "all"}]
                                                + [{"label": i, "value": i} for i in df["EducationLevel"].dropna().unique()],
                                                value="all",
                                                clearable=False,
                                            ),
                                        ],
                                        width=2,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Div("Job Level", className="filter-label"),
                                            dcc.Dropdown(
                                                id="joblevel-filter",
                                                options=[{"label": "All Job Levels", "value": "all"}]
                                                + [{"label": f"Level {i}", "value": i} for i in sorted(df["JobLevel"].unique())],
                                                value="all",
                                                clearable=False,
                                            ),
                                        ],
                                        width=1,
                                    ),
                                    dbc.Col(
                                        html.Button("Reset", id="reset-filters", className="reset-btn", style={"marginTop": "14px"}),
                                        width=1,
                                    ),
                                    dbc.Col(
                                        html.Button("Clear Selections", id="clear-selections", className="clear-selections-btn", style={"marginTop": "14px"}),
                                        width=1,
                                    ),
                                ],
                                className="g-0",
                            ),
                            className="filter-container",
                        ),
                    ],
                    width=12,
                )
            ],
            className="g-0",
        ),

        # Selection indicator
        dbc.Row(
            dbc.Col(
                html.Div(id="selection-indicator", style={"fontSize": "10px", "marginBottom": "3px"}),
                width=12,
            ),
            className="g-0",
        ),

        # KPI Row
        dbc.Row(
            dbc.Col(
                [
                    html.Div("OVERVIEW", className="section-header"),
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Div(
                                    [html.Div("Attrition Rate", className="kpi-title"), html.Div(id="attrition-rate", className="kpi-value")],
                                    className="kpi-container",
                                ),
                                width=3,
                            ),
                            dbc.Col(
                                html.Div(
                                    [html.Div("Total Employees", className="kpi-title"), html.Div(id="total-employees", className="kpi-value")],
                                    className="kpi-container",
                                ),
                                width=3,
                            ),
                            dbc.Col(
                                html.Div(
                                    [html.Div("Avg Monthly Income", className="kpi-title"), html.Div(id="avg-income", className="kpi-value")],
                                    className="kpi-container",
                                ),
                                width=3,
                            ),
                            dbc.Col(
                                html.Div(
                                    [html.Div("Avg Job Satisfaction", className="kpi-title"), html.Div(id="avg-satisfaction", className="kpi-value")],
                                    className="kpi-container",
                                ),
                                width=3,
                            ),
                        ],
                        className="g-0",
                    ),
                ],
                width=12,
            ),
            className="g-0",
        ),

        # Main graph area (populated via callback)
        html.Div(id="visualization-container", className="mt-1"),
    ],
    fluid=True,
    className="dashboard-container",
)

# =========================================================
# Helper: apply both dropdown filters & interactive selections
# =========================================================
def filter_dataframe(base_df: pd.DataFrame, filter_vals: dict, selected_data: dict):
    df_filt = base_df.copy()

    # -- dropdown filters --
    if filter_vals["department"] != "all":
        df_filt = df_filt[df_filt["Department"] == filter_vals["department"]]
    if filter_vals["gender"] != "all":
        df_filt = df_filt[df_filt["Gender"] == filter_vals["gender"]]
    if filter_vals["overtime"] != "all":
        df_filt = df_filt[df_filt["OverTime"] == filter_vals["overtime"]]
    if filter_vals["education"] != "all":
        df_filt = df_filt[df_filt["EducationLevel"] == filter_vals["education"]]
    if filter_vals["joblevel"] != "all":
        df_filt = df_filt[df_filt["JobLevel"] == filter_vals["joblevel"]]

    # -- chart selections --
    if selected_data:
        if selected_data.get("departments"):
            df_filt = df_filt[df_filt["Department"].isin(selected_data["departments"])]
        if selected_data.get("genders"):
            df_filt = df_filt[df_filt["Gender"].isin(selected_data["genders"])]
        if selected_data.get("age_ranges"):
            age_filt = False
            for lo, hi in selected_data["age_ranges"]:
                age_filt |= (df_filt["Age"] >= lo) & (df_filt["Age"] <= hi)
            df_filt = df_filt[age_filt]
        if selected_data.get("job_roles"):
            df_filt = df_filt[df_filt["JobRole"].isin(selected_data["job_roles"])]
        if selected_data.get("employee_ids") and "EmployeeID" in df_filt.columns:
            df_filt = df_filt[df_filt["EmployeeID"].isin(selected_data["employee_ids"])]
        if selected_data.get("attrition_values"):
            df_filt = df_filt[df_filt["Attrition"].isin(selected_data["attrition_values"])]
        if selected_data.get("treemap_paths"):
            for path in selected_data["treemap_paths"]:
                if len(path) >= 1:
                    df_filt = df_filt[df_filt["Department"].isin([path[0]])]
                if len(path) >= 2:
                    df_filt = df_filt[df_filt["JobRole"].isin([path[1]])]
                if len(path) >= 3:
                    df_filt = df_filt[df_filt["Attrition"].isin([path[2]])]

    return df_filt


# =============================
# KPI callback
# =============================
@app.callback(
    Output("attrition-rate", "children"),
    Output("total-employees", "children"),
    Output("avg-income", "children"),
    Output("avg-satisfaction", "children"),
    Input("department-filter", "value"),
    Input("gender-filter", "value"),
    Input("overtime-filter", "value"),
    Input("education-filter", "value"),
    Input("joblevel-filter", "value"),
    Input("selected-data-store", "data"),
)
def update_kpis(dept, gender, overtime, education, joblevel, sel):
    filt_vals = dict(
        department=dept, gender=gender, overtime=overtime, education=education, joblevel=joblevel
    )
    df_view = filter_dataframe(df, filt_vals, sel)

    attrition_rate = f"{df_view['Attrition_Flag'].mean() * 100:.2f}%"
    total_emp = f"{df_view.shape[0]}"
    avg_income = f"${int(df_view['MonthlyIncome'].mean()):,}"
    avg_sat = f"{df_view['JobSatisfaction'].mean():.1f}/5" if "JobSatisfaction" in df_view else "N/A"

    return attrition_rate, total_emp, avg_income, avg_sat


# =============================
# Graphs callback
# =============================
# ─────────  REPLACE the whole update_visuals function with the block below ──────────
@app.callback(
    Output("visualization-container", "children"),
    Input("department-filter", "value"),
    Input("gender-filter", "value"),
    Input("overtime-filter", "value"),
    Input("education-filter", "value"),
    Input("joblevel-filter", "value"),
    Input("selected-data-store", "data"),
)
def update_visuals(dept, gender, overtime, education, joblevel, sel):
    """
    Builds / refreshes every graph each time a filter or a chart‑selection changes.
    Keeps the Attrition colour‑scheme absolutely consistent:
        • "Yes" → orange  (#ff8c42)
        • "No"  → charcoal (#4a4a4a)
    """
    try:
        # 1 ── FILTER DATA ───────────────────────────────────────────────────────
        filt_vals = dict(department=dept, gender=gender, overtime=overtime,
                         education=education, joblevel=joblevel)
        dff = filter_dataframe(df, filt_vals, sel)

        # 2 ── COLOUR & LAYOUT UTILS ─────────────────────────────────────────────
        base_colors      = [colors["primary"], colors["secondary"]]   # for non‑attrition pies/bars
        attr_color_map   = {"Yes": colors["attrition"], "No": colors["retention"]}
        common_layout = dict(
            font_family="Segoe UI", font_size=9,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=30, r=5, t=25, b=25), height=265,
            legend=dict(orientation="h", y=1.02, yanchor="bottom",
                        x=1, xanchor="right", font=dict(size=8)),
        )
        sel_style = dict(
            selected=dict(marker=dict(opacity=1)),
            unselected=dict(marker=dict(opacity=0.35)),
        )

        # 3 ── GRAPHS ────────────────────────────────────────────────────────────
        # 3.1  Attrition by Department
        dept_counts = dff.groupby(["Department", "Attrition"]).size().reset_index(name="Count")
        bar_fig = px.bar(
            dept_counts, x="Department", y="Count", color="Attrition",
            barmode="group", title="Attrition by Department",
            color_discrete_map=attr_color_map,
            category_orders={"Attrition": ["No", "Yes"]},
        )
        bar_fig.update_layout(**common_layout)
        bar_fig.update_xaxes(tickangle=45, tickfont=dict(size=8))
        bar_fig.update_traces(**sel_style, selector=dict(type="bar"))

        # 3.2  Gender distribution (pie) – not tied to Attrition
        pie_fig = px.pie(
            dff, names="Gender", title="Gender Distribution",
            color_discrete_sequence=base_colors,
        )
        pie_fig.update_layout(**common_layout)
        pie_fig.update_traces(textinfo="percent+label", textfont_size=8,
                              textposition="inside")

        # 3.3  Age histogram by Attrition
        age_fig = px.histogram(
            dff, x="Age", color="Attrition", nbins=15,
            title="Age Distribution by Attrition",
            color_discrete_map=attr_color_map,
            category_orders={"Attrition": ["No", "Yes"]},
        )
        age_fig.update_layout(**common_layout, dragmode="select")
        age_fig.update_traces(**sel_style, selector=dict(type="histogram"))

        # 3.4  Job‑Role bar chart by Attrition
        role_counts = dff.groupby(["JobRole", "Attrition"]).size().reset_index(name="Count")
        role_fig = px.bar(
            role_counts, x="JobRole", y="Count", color="Attrition",
            barmode="group", title="Job Role Distribution by Attrition",
            color_discrete_map=attr_color_map,
            category_orders={"Attrition": ["No", "Yes"]},
        )
        role_fig.update_layout(**common_layout,
                               xaxis=dict(categoryorder="total descending",
                                          tickangle=45, tickfont=dict(size=7)))
        role_fig.update_traces(**sel_style, selector=dict(type="bar"))

        # 3.5  Treemap (already uses the map)
        tree_df = dff.groupby(["Department", "JobRole", "Attrition"]).size()\
                     .reset_index(name="Count")
        tree_fig = px.treemap(
            tree_df, path=["Department", "JobRole", "Attrition"], values="Count",
            color="Attrition", color_discrete_map=attr_color_map,
            title="Hierarchical View of Attrition",
        )
        tree_fig.update_layout(
            font_family="Segoe UI", font_size=8,
            paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=5, r=5, t=25, b=5),
            height=220,
        )
        tree_fig.update_traces(hovertemplate="<b>%{label}</b><br>Count: %{value}<extra></extra>")

        # 3.6  PCA scatter by Attrition
        num_cols = ["Age", "MonthlyIncome", "TotalWorkingYears",
                    "YearsAtCompany", "YearsInCurrentRole"]
        avail = [c for c in num_cols if c in dff.columns]
        if len(avail) >= 2 and not dff.empty:
            scaled = StandardScaler().fit_transform(
                dff[avail].fillna(dff[avail].mean())
            )
            pc = PCA(n_components=2).fit_transform(scaled)
            pca_df = pd.DataFrame({
                "PCA1": pc[:, 0], "PCA2": pc[:, 1],
                "Attrition": dff["Attrition"].values
            })
            if "EmployeeID" in dff.columns:
                pca_df["EmployeeID"] = dff["EmployeeID"].values

            pca_fig = px.scatter(
                pca_df, x="PCA1", y="PCA2", color="Attrition",
                title=f"PCA Analysis (Explained Var: "
                      f"{pca_df[['PCA1','PCA2']].var().sum():.2%})",
                color_discrete_map=attr_color_map,
                category_orders={"Attrition": ["No", "Yes"]},
                custom_data=["EmployeeID"] if "EmployeeID" in pca_df else None,
            )
            pca_fig.update_layout(**common_layout, dragmode="select")
            pca_fig.update_traces(**sel_style, selector=dict(mode="markers"))
        else:
            pca_fig = go.Figure()
            pca_fig.add_annotation(
                text="Insufficient numerical data for PCA analysis",
                x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
            pca_fig.update_layout(**common_layout,
                                  title="PCA Analysis (Unavailable)")

        # 4 ── GRID LAYOUT RETURN ────────────────────────────────────────────────
        return [
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="dept-bar-chart", figure=bar_fig,
                                      className="graph-container",
                                      config={"displaylogo": False}), width=4,
                            className="card px-0"),
                    dbc.Col(dcc.Graph(id="gender-pie-chart", figure=pie_fig,
                                      className="graph-container",
                                      config={"displaylogo": False}), width=4,
                            className="card px-0"),
                    dbc.Col(dcc.Graph(id="age-histogram", figure=age_fig,
                                      className="graph-container",
                                      config={"displaylogo": False}), width=4,
                            className="card px-0"),
                ], className="g-0"),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="role-bar-chart", figure=role_fig,
                                      className="graph-container",
                                      config={"displaylogo": False}), width=4,
                            className="card px-0"),
                    dbc.Col(dcc.Graph(id="pca-scatter", figure=pca_fig,
                                      className="graph-container",
                                      config={"displaylogo": False}), width=4,
                            className="card px-0"),
                    dbc.Col(dcc.Graph(id="dept-treemap", figure=tree_fig,
                                      className="graph-container",
                                      config={"displaylogo": False}), width=4,
                            className="card px-0"),
                ], className="g-0"),
        ]

    except Exception as e:
        return html.Div(f"Error processing data: {e}",
                        style={"color": "red", "padding": "10px"})

# ─────────  END OF REPLACEMENT BLOCK ──────────

# =========================================
# Callbacks for reset / clear selections
# =========================================
@app.callback(
    Output("department-filter", "value"),
    Output("gender-filter", "value"),
    Output("overtime-filter", "value"),
    Output("education-filter", "value"),
    Output("joblevel-filter", "value"),
    Input("reset-filters", "n_clicks"),
)
def reset_filters(n):
    if not n:
        raise PreventUpdate
    return "all", "all", "all", "all", "all"


@app.callback(
    Output("selected-data-store", "data"),
    Input("clear-selections", "n_clicks"),
    Input("reset-filters", "n_clicks"),
    State("selected-data-store", "data"),
)
def clear_selections(n_clear, n_reset, current):
    trig = callback_context.triggered
    if not trig:
        raise PreventUpdate
    btn = trig[0]["prop_id"].split(".")[0]
    if btn in {"clear-selections", "reset-filters"}:
        return {}
    return current


# =========================================
# Chart‑selection callbacks (toggle logic)
# =========================================
def toggle_item(array_key, value, store):
    store = store or {}
    lst = store.get(array_key, [])
    if value in lst:
        lst.remove(value)
    else:
        lst.append(value)
    if lst:
        store[array_key] = lst
    else:
        store.pop(array_key, None)
    return store


@app.callback(
    Output("selected-data-store", "data", allow_duplicate=True),
    Input("dept-bar-chart", "clickData"),
    State("selected-data-store", "data"),
    prevent_initial_call=True,
)
def select_dept(click, store):
    if not click:
        raise PreventUpdate
    dept = click["points"][0]["x"]
    return toggle_item("departments", dept, store)


@app.callback(
    Output("selected-data-store", "data", allow_duplicate=True),
    Input("gender-pie-chart", "clickData"),
    State("selected-data-store", "data"),
    prevent_initial_call=True,
)
def select_gender(click, store):
    if not click:
        raise PreventUpdate
    gender = click["points"][0]["label"]
    return toggle_item("genders", gender, store)


@app.callback(
    Output("selected-data-store", "data", allow_duplicate=True),
    Input("role-bar-chart", "clickData"),
    State("selected-data-store", "data"),
    prevent_initial_call=True,
)
def select_role(click, store):
    if not click:
        raise PreventUpdate
    role = click["points"][0]["x"]
    return toggle_item("job_roles", role, store)


@app.callback(
    Output("selected-data-store", "data", allow_duplicate=True),
    Input("age-histogram", "selectedData"),
    State("selected-data-store", "data"),
    prevent_initial_call=True,
)
def select_age(select, store):
    if not select or not select.get("points"):
        raise PreventUpdate
    ranges = []
    for p in select["points"]:
        x = p["x"]
        bin_w = 5
        ranges.append((int(x - bin_w / 2), int(x + bin_w / 2)))
    store = store or {}
    if ranges:
        store["age_ranges"] = ranges
    else:
        store.pop("age_ranges", None)
    return store


@app.callback(
    Output("selected-data-store", "data", allow_duplicate=True),
    Input("pca-scatter", "selectedData"),
    State("selected-data-store", "data"),
    prevent_initial_call=True,
)
def select_pca(select, store):
    if not select or not select.get("points"):
        raise PreventUpdate
    ids = [p["customdata"][0] for p in select["points"] if p.get("customdata")]
    store = store or {}
    if ids:
        store["employee_ids"] = ids
    else:
        store.pop("employee_ids", None)
    return store


@app.callback(
    Output("selected-data-store", "data", allow_duplicate=True),
    Input("dept-treemap", "clickData"),
    State("selected-data-store", "data"),
    prevent_initial_call=True,
)
def select_treemap(click, store):
    if not click:
        raise PreventUpdate
    path = click["points"][0]["id"].split("/")
    store = store or {}
    treemap_paths = store.get("treemap_paths", [])
    if path in treemap_paths:
        treemap_paths.remove(path)
    else:
        treemap_paths.append(path)
    if treemap_paths:
        store["treemap_paths"] = treemap_paths
    else:
        store.pop("treemap_paths", None)
    return store


# =========================
# Selection indicator text
# =========================
@app.callback(Output("selection-indicator", "children"), Input("selected-data-store", "data"))
def indicator_text(sel):
    if not sel:
        return ""
    bits = []
    if sel.get("departments"):
        bits.append(f"Department: {', '.join(sel['departments'])}")
    if sel.get("genders"):
        bits.append(f"Gender: {', '.join(sel['genders'])}")
    if sel.get("age_ranges"):
        bits.append(
            "Age: "
            + ", ".join(f"{lo}-{hi}" for lo, hi in sel["age_ranges"])
        )
    if sel.get("job_roles"):
        bits.append(f"Job Role: {', '.join(sel['job_roles'])}")
    if sel.get("treemap_paths"):
        paths = [" > ".join(p) for p in sel["treemap_paths"]]
        if len(paths) <= 2:
            bits.append(f"Hierarchy: {', '.join(paths)}")

    if not bits:
        return ""

    return [
        html.Span("Active Selections: ", style={"fontWeight": "bold", "fontSize": "10px"}),
        *[html.Span([b, html.Span(" × ", className="selection-badge")]) for b in bits],
        html.Span("  Click 'Clear Selections' to reset", style={"fontSize": "10px", "fontStyle": "italic", "marginLeft": "5px"}),
    ]


# Misc
app.config.suppress_callback_exceptions = True

if __name__ == "__main__":
    app.run()
