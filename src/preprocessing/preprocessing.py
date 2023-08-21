import pandas as pd
import numpy as np

import logging
from collections import defaultdict
from typing import Callable, Dict, List, Tuple, Union






logger = logging.getLogger(__name__)
quality_check_logger = logging.getLogger("quality_check_logger")



from os import listdir
from os.path import isfile, join
import os

def read_files(filenames, folder_path ,header, skip_rows, sheet = 'Sheet1' ):
    """
    Used for reading excel files. Function takes filenames, folder path as inputs and reads and concats all data in the path
    {filenames: list of all file names in the folder,
    folder_path: specified folder path}
    """

    def read_file(filename,header, skip_rows,sheet = sheet, folder=None ):
        # TODO: add docstring
        if folder is not None:
            filename = os.path.join(folder, filename)
        df = pd.read_excel(filename,sheet_name = sheet,header = header,skiprows = skip_rows )[1:].reset_index(drop=True)
        df.reset_index(drop=True, inplace=True)
        return df

    merged_df = pd.concat([read_file(f,header,skip_rows,sheet,folder=folder_path ) for f in filenames])
    return merged_df







def read_prep_actuals(path):
    """
    Used for reading and preprocessing actual sales file. Function takes actual sales, reads and preprocess actual sales file in the path
    {path: specified folder path}
    """

    onlyfiles = [f for f in listdir(path) if isfile(join(path, f)) if '.xls' in f]
    actual_data = read_files(onlyfiles,path,header=2,skip_rows=0)
    raw_data = actual_data.copy()



    return raw_data



def round_timestamps(
    frequency: str,
    data: pd.DataFrame,
    datetime_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Rounds timestamps in order to reduce minor timestamp noise.
    Different frequency aliases can be found here:
    https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases

    Args:
       frequency: timeseries offset aliases.
       data: input data
       datetime_col: timestamp column

    Returns:
       data with rounded timestamps
    """
    data = data.copy()
    data[datetime_col] = pd.to_datetime(data[datetime_col]).dt.round(frequency)
    logger.info(f"Rounding '{datetime_col}' to '{frequency}' frequency.")
    return data



def replace_inf_values(data: pd.DataFrame) -> pd.DataFrame:
    """Replace any infinite values in dataset with NaN.

    Args:
        data: input data

    Returns:
        Dataframe with infinite values replaced by NaN & dropped only if explicitly
        asked to drop those
    """
    df_new = data.copy()
    infinity_set = [np.inf, -np.inf]
    df_new = df_new.replace(infinity_set, np.nan)
    summary = pd.DataFrame()
    summary["before_cleaning"] = data.isin(infinity_set).sum()
    summary["after_cleaning"] = df_new.isin(infinity_set).sum()

    summary_count = summary.loc[summary["before_cleaning"] > 0]

    logger.info(f"\nnumber of inf values in data: \n{summary_count}")

    return df_new



def remove_null_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Identify columns that contain all NaN/NaTs and drop them

    Args:
       data: input pandas dataframe

    Returns:
       data
    """
    # Get initial columns from data
    data_columns = set(data.columns)

    # Drop all columns that are comprised of ALL NaN's/NaT's
    data = data.dropna(axis=1, how="all")

    # Get a set of all the columns that have been dropped
    dropped_columns = data_columns.difference(set(data.columns))

    if dropped_columns:
        logger.info(
            f"Dropped columns: {dropped_columns} due to all"
            " column values being NaN/NaT",
        )
        quality_check_logger.info(
            f"Dropped columns: {dropped_columns} due to all"
            " column values being NaN/NaT",
        )
    else:
        logger.info("All columns have values. Continuing...")
    return data



def unify_timestamp_col_name(
    data: pd.DataFrame,
    datetime_col: str,
    unified_name: str = "timestamp",
) -> pd.DataFrame:
    """Unify all timestamp column names that will be further used as index

    Args:
       params: dictionary of parameters
       data: input data
       unified_name: desired unified column name

    Returns:
       data
    """

    # check if a duplicate unified_name will be created
    # raise an error if so
    if (unified_name in data.columns) and (unified_name != datetime_col):
        raise ValueError(
            f"column name '{unified_name}' already exists. "
            f"Renaming another column to '{unified_name}' "
            f"will lead to duplicate column names",
        )

    df = data.rename(columns={datetime_col: unified_name})
    logger.info(f"Rename column '{datetime_col}' to '{unified_name}'.")

    return df




def deduplicate_pandas(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Drop duplicates for pandas dataframe

    Args:
       data: input data
       **kwargs: keywords feeding into the pandas `drop_duplicates`

    Returns:
       data with duplicates removed
    """
    logger.info(f"Dataframe shape before dedup: {data.shape}")

    sub = data.drop_duplicates(**kwargs)
    sub.reset_index(inplace=True, drop=True)

    logger.info(f"Dataframe shape after dedup: {sub.shape}")

    n_dropped = data.shape[0] - sub.shape[0]
    if n_dropped > 0:
        quality_check_logger.info(f"Dropped {n_dropped} duplicate timestamps")
    else:
        quality_check_logger.info("No duplicate timestamps in data source.")
    return sub



def remove_outlier_auto(df,quantiles=[0.01,1],cols= []):
    for col in cols:
        percentiles = df[col].quantile([quantiles[0], quantiles[1]]).values
        df[col][df[col] <= percentiles[0]] = percentiles[0]
        df[col][df[col] >= percentiles[1]] = percentiles[1]
    return df


def bins_generate(bins,data,lst_bins):
    lst = []
    for i in lst_bins:
        col_name = i+'_bins'
        lst.append(col_name)
        data[col_name] = pd.qcut(data[i].rank(method='first'), bins).astype(str)
        data = pd.merge(data,data.groupby(col_name)[i].agg(['min', 'max']).reset_index(),on=col_name,how='left')
        data[col_name] = round(data['min'],4).astype(str) + ' - ' + round(data['max'],4).astype(str)
        data.drop(['min','max'],axis=1,inplace=True)
        
    return data , lst


def days_to_event(df,col_name):
    lst = []
    k = 0
    for i in range(0,len(df)):
        if len(lst)==0:
            lst.append(1)
            k = 1
        elif df[col_name][i] == 0:
            k = k+1
            lst.append(k)
        else : 
            lst.append(1)
            k = 1
    df['days_'+col_name] = [ x-1 for x in lst]

    return df




def apply_outlier_remove_rule(  
    df: pd.DataFrame,
    rule: str,
    num_tag_range: Dict[str, Tuple[float, float]],
) -> pd.DataFrame:
    """Remove outliers with selected rule

    Args:
       df: input data
       rule: ways to remove outlier. either 'clip' or 'drop'
       num_tag_range: dict with col name and its value range

    Returns:
       df
    """
    for col in df.columns:
        # skip columns that are not numeric
        if col not in num_tag_range.keys():
            continue

        td_low, td_up = num_tag_range[col]

        # skip columns that don't have lower and upper limits
        if np.isnan(td_low) and np.isnan(td_up):
            continue

        lower = None if np.isnan(td_low) else td_low
        upper = None if np.isnan(td_up) else td_up

        if rule == "clip":
            df[col].clip(lower, upper, inplace=True)
            logger.info(f"Clipping {col} to [{lower}, {upper}] range.")
        elif rule == "drop":
            outside_range_mask = (df[col] < lower) | (df[col] > upper)  
            df[col].mask(outside_range_mask, inplace=True)
        else:
            raise ValueError(
                f"Invalid outlier removal rule `{rule}`. "
                "Choose supported rules 'clip' or 'drop'",
            )

    return df



def remove_outlier(
    data: pd.DataFrame,
    td: dict,
    rule: str = "clip",
) -> pd.DataFrame:
    """Remove outliers based on value range set in tagdict
    and selected rule

    Args:
       data: input data
       td: tag dictionary
       rule: ways to remove outlier. either 'clip' or 'drop'

    Returns:
       df_new
    """
    # construct a dict with col name and its value range
    tag_range = {}
    for col in data.columns:
        if col in td:
            tag_range[col] = (
                td[col]["range_min"], 
                td[col]["range_max"],  
            )

    # apply outlier removing rule
    df_new = apply_outlier_remove_rule(data.copy(), rule, tag_range)

    if rule == "drop":
        n_dropped = df_new.isnull().sum() - data.isnull().sum()
        perc_dropped = n_dropped / data.shape[0] * 100

        summary = pd.DataFrame()
        summary["n_dropped"] = n_dropped
        summary["perc_dropped(%)"] = perc_dropped
        dropped_outliers = summary.loc[summary["n_dropped"] > 0].round(2)
        quality_check_logger.info(
            f"number of outliers dropped per column: \n{dropped_outliers}",
        )

    return df_new


def smape(y_true, y_pred):
    """
    Used for calculating symmetric mapes. Function takes actuals and predicted and calculates smape
    {y_true: actuals,
    y_pred: predicted}
    """
    import numpy as np
    return 100 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
   

def dash_dashboard(plot_data,lst_bins,list_of_variables_2, date_column, color_col):
    from dash import Dash, dcc, html, Input, Output
    from plotly.subplots import make_subplots
    import plotly.express as px
    import plotly.graph_objects as go

    app = Dash(__name__)


    app.layout = html.Div(children=[


        html.Div([
            html.H1(id = 'H1', children = "Analysis of the feature variablity", style = {'textAlign':'center',\
                                                'marginTop':40,'marginBottom':40}),
        html.P("x-axis:"),
        dcc.Dropdown(
            id='x-axis', 
            options = lst_bins,
            value= lst_bins[0], 
        ),
        html.P("y-axis:"),
        dcc.Dropdown(
            id='y-axis', 
            options=list_of_variables_2,
            value=list_of_variables_2[0], 
        ),
        dcc.Graph(id="graph"),
        ]),

        html.Div([
            html.H1(id = 'H2', children = "Time series plot", style = {'textAlign':'center',\
                                                'marginTop':40,'marginBottom':40}),
        
        dcc.Graph(id="graph2"),
        ]),


        html.Div([
            html.H1(id = 'H3', children = "Scatter Plot", style = {'textAlign':'center',\
                                                'marginTop':40,'marginBottom':40}),
        
        dcc.Graph(id="graph3"),
        ]),
    ])






    @app.callback([
                Output('graph' , 'figure'),
                Output('graph2' , 'figure'), 
                Output('graph3' , 'figure'),
                Input("x-axis", "value"), 
                Input("y-axis", "value")
                ])

    def generate_chart(x, y):
        df = plot_data.copy()
        df = df.sort_values(x[:-5])
        
        df_2 = plot_data.sort_values(date_column)
        
        df_3 = plot_data.copy()


        fig = px.box(df, x=x, y=y)
        
        
        fig3 = px.scatter(df_3, x=x[:-5], y=y, opacity=.6 , color=color_col)


        fig2 = make_subplots(specs=[[{"secondary_y": True}]])

        fig2.add_trace(
            go.Scatter(x=df_2['timestamp'], y=df_2[y], mode="lines"),
            secondary_y=True
        )

        fig2.add_trace(
            go.Scatter(x=df_2['timestamp'], y=df_2[x[:-5]], mode="lines"),
            secondary_y=False
        )

        fig2.update_xaxes(title_text="Time stamp")
        # Set y-axes titles
        fig2.update_yaxes(title_text=y, secondary_y=False)
        fig2.update_yaxes(title_text=x[:-5], secondary_y=True)
        fig2.update_layout(showlegend=False, autosize=True, width=1800, height=500)



        

        
        return [fig,fig2,fig3]

    app.run_server(debug=False,host='0.0.0.0', port = 8080)