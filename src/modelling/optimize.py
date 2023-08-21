from itertools import product
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import optuna
import numpy as np
import pandas as pd
import os

from optuna.visualization import plot_slice
from optuna.visualization import plot_edf
from optuna.visualization import plot_contour
from optuna.visualization import plot_parallel_coordinate


from joblib import Parallel,delayed
import multiprocessing
num_core=multiprocessing.cpu_count()


def generate_opt_data(cycle,path):
    sequences = []
    for x in product('01', repeat=cycle):
        x = list(x)

        try:
            if '1 1' not in ' '.join(x) and len(set(x))!=1 and ' '.join(x) .startswith("0 0 0 0") and '1 0 1' not in ' '.join(x):
                sequences.append(x)
        except:
            pass
    opt_df = pd.DataFrame(sequences).T.astype(int)
    opt_df.columns = [str(x) for x in opt_df.columns.to_list()]
    opt_df.to_csv(os.path.join(path,'blowdown_seq.csv'),index=False)
    return opt_df




def days_to_event(df, bdqty, kettle_level_range, increa_afterblowdown_kettle, increa_blowdown_qty, opt_parameters, function_ = 'linear'):
    
    col_name = df.columns.to_list()[0]
    high_boiler_cutoff = opt_parameters['high_boiler_cutoff']
    contribution_margin = opt_parameters['contribution_margin']
    shutdown_rev = opt_parameters['shutdown_rev']
    bdqty_ = opt_parameters['bdqty_']
    kettle_level_range_u = opt_parameters['kettle_level_range_u']
    kettel_level_range_l = opt_parameters['kettel_level_range_l']
    kettel_level_max = opt_parameters['kettel_level_max']
    etac_tp = opt_parameters['etac_tp']
    n_working_days = opt_parameters['n_working_days']

    

    

    if function_ != 'linear': 
        functions = opt_parameters['functions']['non_linear']
        intercept = functions['intercept']
        b_1 = functions['b1']
        b_2 = functions['b2']
    
    else: 
        functions = opt_parameters['functions']['linear']
        intercept =  functions['intercept']
        b_1 = functions['b1']



    lst = []
    k = 0
    diction = {}
    
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
    if function_ != 'linear': lst_2 = [b_1*((x)**2) + b_2*(x) + (intercept) for x in lst]
    else: lst_2 = [(b_1*x) + (intercept) for x in lst]
    
    lst_3 = lst_2.copy()
    
    
    count = []
    after_blowdown_levl_lst = [0]
    bdqty_lst = [0]
    
    for i in range(0,len(lst_2)-1):
        if len(count) == 0:
       
            if lst_2[i] < lst_2[i+1]:
                lst_3[i] = lst_3[i]
                after_blowdown_levl_lst.append(0)
                bdqty_lst.append(0)
            elif (lst_2[i] > lst_2[i+1]) & (lst[i+1] < lst[i]):

                concentration = (kettle_level_range_u/kettle_level_range[0])*lst_3[i]
                after_blowdown_level = (((kettle_level_range[0]/100)*kettel_level_max) - bdqty[0])/kettel_level_max
                after_blowdown_levl_lst.append(after_blowdown_level)
                bdqty_lst.append(bdqty[0])
                after_blowdown = (after_blowdown_level*concentration)/(kettle_level_range_u/100)
                
                if function_ != 'linear': 
                    new_intercept = after_blowdown - b_1*((i+1)**2) - b_2*(i+1)
                    lst_3[i+1:] =  [b_1*((x+1)**2) + b_2*(x+1) + (new_intercept) for x in list(range(i,len(lst_3)))]
                else: 
                    new_intercept =  after_blowdown - (b_1*(i+1))
                    lst_3[i+1:] =  [ (b_1*(x+1)) + (new_intercept) for x in list(range(i,len(lst_3)))]
                
                
                count.append(1)
        else:
            if lst_2[i] < lst_2[i+1]:
                lst_3[i] = lst_3[i]
                after_blowdown_levl_lst.append(0)
                bdqty_lst.append(0)
            elif (lst_2[i] > lst_2[i+1]) & (lst[i+1] < lst[i]):

                concentration = (kettle_level_range_u/kettle_level_range[len(count)])*lst_3[i]
                after_blowdown_level = (((kettle_level_range[len(count)]/100)*kettel_level_max) - bdqty[len(count)])/kettel_level_max
                after_blowdown_levl_lst.append(after_blowdown_level)
                bdqty_lst.append(bdqty[len(count)])
                after_blowdown = (after_blowdown_level*concentration)/(kettle_level_range_u/100)
                
                
                 
                if function_ != 'linear': 
                    new_intercept = after_blowdown - b_1*((i+1)**2) - b_2*(i+1)
                    lst_3[i+1:] =  [b_1*((x+1)**2) + b_2*(x+1) + (new_intercept) for x in list(range(i,len(lst_3)))]
                else: 
                    new_intercept =  after_blowdown - (b_1*(i+1))
                    lst_3[i+1:] =  [ (b_1*(x+1)) + (new_intercept) for x in list(range(i,len(lst_3)))]
                    


                count.append(1)
            
                
    ind = [i for i,v in enumerate(lst_3) if v > high_boiler_cutoff]
    try:
        min_ = min(ind)
        a = [x for x in range(min_,len(lst_3))]
        lst_3 = np.asarray(lst_3)
        lst_3[a]=0
        lst_3 = list(lst_3.flatten())
        
        
        b = [x for x in range(min_,len(after_blowdown_levl_lst))]
        after_blowdown_levl_lst = np.asarray(after_blowdown_levl_lst)
        after_blowdown_levl_lst[b]=0
        after_blowdown_levl_lst = list(after_blowdown_levl_lst.flatten())

        c = [x for x in range(min_,len(bdqty_lst))]
        bdqty_lst = np.asarray(bdqty_lst)
        bdqty_lst[c]=0
        bdqty_lst = list(bdqty_lst.flatten())


    except:
        pass
    lst_3 = lst_3[0:len(lst)]
    diction['sequence_'+col_name] =  lst_3[0:len(lst)]
    
    
    bdqty_lst_cal = [ x for x in bdqty_lst if x > 0]
    after_blowdown_levl_lst_cal = [ x for x in after_blowdown_levl_lst if x > 0]
    
    cost_bld_qty = [((x*increa_blowdown_qty)*1000) for x in bdqty_lst_cal]
    cost_kettel_level = [contribution_margin + (((kettle_level_range_u/100)-x)*increa_afterblowdown_kettle)*100 for x in after_blowdown_levl_lst_cal]
    
    total_cost = [sum(x) for x in zip(cost_bld_qty, cost_kettel_level)]
    blowdn_contrib = [x*etac_tp for x in total_cost]
    
    
    
    rev = blowdn_contrib + [etac_tp*contribution_margin  for i in range(1,len([ x for x in lst_3 if x!=0]) - len(blowdn_contrib))] 
    
    if lst_3[-1] == 0:
        sd_cost = shutdown_rev # ETAC * 6 hours + fixed cost
        rev.extend([sd_cost])
    
    rev_lst = [ x for x in rev if x != 0]
    number_of_sq = n_working_days/len(rev_lst)
    annual_rev = (number_of_sq*sum(rev_lst))/(10**7)
    
    # print(len(total_cost),len([ x for x in lst_3 if x!=0]) - len(blowdn_contrib),len([ x for x in lst_3 if x!=0]), round(annual_rev,3))
    # annual_rev = (np.mean(rev)*330)/(10**7)
    
    if function_ != 'linear': lst_no_blowdown = [ b_1*((x+1)**2) + b_2*(x+1) + intercept for x in list(range(0,len(lst_3)))]
    else: lst_no_blowdown = [ (b_1*(x+1)) + intercept for x in list(range(0,len(lst_3)))]
    

    diction_2 = pd.DataFrame({'bdqty':bdqty_lst,'blowdown_levl_lst':after_blowdown_levl_lst,'high_boilers':lst_3[0:len(lst)], 'No BD' : lst_no_blowdown, 'sequence_id':col_name})
        
    
    

    
    
    return round(annual_rev,3), diction_2




def optimize(optuna_disable_logging, X, n_trials, increa_afterblowdown_kettle, increa_blowdown_qty, function_, opt_parameters, save_df):

        kettle_level_range_u = opt_parameters['kettle_level_range_u']
        kettel_level_range_l = opt_parameters['kettel_level_range_l']
        kettel_level_max = opt_parameters['kettel_level_max']
        bdqty_ = opt_parameters['bdqty_']
        max_blowdowns = X.sum(axis=0).max()
        opt_datafame = []
                
        count = 0
        if optuna_disable_logging: optuna.logging.disable_default_handler()
        def objective(trial):
                n_iter = 1
                
                for step in range(n_iter):
                        lst_paramsbdqty = []
                        for i in range(0,max_blowdowns):
                                lst_paramsbdqty.append({"bdqty_" + str(i) : trial.suggest_float("bdqty_"+str(i), bdqty_[0], bdqty_[1])}["bdqty_" + str(i) ])
                        

                        lst_lower_level = []  
                        for i in range(0,max_blowdowns):
                                lst_lower_level.append((((kettel_level_max*(kettel_level_range_l/100) ) + (lst_paramsbdqty[i]))/kettel_level_max )*100 )
                        
                        
                        lst_paramskettle_level_range = []
                        for i in range(0,max_blowdowns):
                                lst_paramskettle_level_range.append({"kettle_level_range_" + str(i) :trial.suggest_float("kettle_level_range_"+ str(i), lst_lower_level[i], kettle_level_range_u)}["kettle_level_range_" + str(i)])

                        
                        rev, df_optimized = days_to_event(X, lst_paramsbdqty, lst_paramskettle_level_range, increa_afterblowdown_kettle, increa_blowdown_qty, opt_parameters, function_ )
                        
                        intermediate_value = rev
                        trial.report(intermediate_value, step)

                        if trial.should_prune(): raise optuna.TrialPruned()
                
                rev, df_optimized = days_to_event(X, lst_paramsbdqty, lst_paramskettle_level_range, increa_afterblowdown_kettle, increa_blowdown_qty, opt_parameters, function_)
                
                if save_df:
                        df_optimized['revenue'] = rev
                        opt_datafame.append(df_optimized)

                                        
                return rev   


        study = optuna.create_study(study_name="mv_hyperparameters",directions=["maximize"],sampler=optuna.samplers.TPESampler(seed=0),
                       pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0))
        study.optimize(objective, n_trials=n_trials, timeout= 500)
        
        return study.best_params, study.best_value, study , opt_datafame, X.columns[0]




def wrapper(optuna_disable_logging, df, col_lst, iterations, increa_afterblowdown_kettle, increa_blowdown_qty, opt_parameters, min_thresh, function_, multiply, drop_per_iteration = 10):

    
    
    if len(col_lst)<=min_thresh:
        temp_var = Parallel(n_jobs=num_core)(delayed(optimize)(optuna_disable_logging, df[[sequence]], iterations*multiply, increa_afterblowdown_kettle, increa_blowdown_qty, function_, opt_parameters, save_df = 1) for sequence in col_lst)
        best_params_dic = {x[4]:{x[1]:x[0]} for x in list(temp_var)}
        study_opt = {x[4]:x[2] for x in list(temp_var)}
        opt_datafame_final = pd.concat([pd.concat(x[3]) for x in list(temp_var)])
        return best_params_dic, study_opt, opt_datafame_final
        
        

    else:
        
        temp_var = Parallel(n_jobs=num_core)(delayed(optimize)(optuna_disable_logging, df[[sequence]], iterations, increa_afterblowdown_kettle, increa_blowdown_qty, function_, opt_parameters, save_df = 0) for sequence in col_lst)
        best_params_dic = {x[4]:{x[1]:x[0]} for x in list(temp_var)}
        study_opt = {x[4]:x[2] for x in list(temp_var)}
        
            
        max_rev =  {k: list(best_params_dic[k].keys())[0] for k in list(best_params_dic)}
        max_rev_list = list(dict(sorted(max_rev.items(), key=lambda x: x[1], reverse=True)[:int(len(col_lst)/drop_per_iteration)]).keys())
        return wrapper(optuna_disable_logging, df, max_rev_list, iterations, increa_afterblowdown_kettle, increa_blowdown_qty, opt_parameters, min_thresh, function_, multiply)
        
        



def plot_animation(df,study_opt,seq, show=False):

    num = df._get_numeric_data()
    num[num < 0] = 0


    fig = px.line(title= "Optimization History", data_frame=df,y=["HB","BD qty","KL after BD"],animation_frame="Revenue",markers=True,line_shape='spline')

    # update approprate traces to use secondary yaxis
    for t in fig.data:
        if t.name=="KL after BD": t.update(yaxis="y2")
    for f in fig.frames:
        for t in f.data:
            if t.name=="KL after BD": t.update(yaxis="y2")


    # configure yaxis2 and give it some space
    fig.update_layout(yaxis2={"overlaying":"y", "side":"right"}, xaxis={"domain":[0,1]}, width=1800, height=600, yaxis2_range=[0,.6], yaxis_range=[0,20])
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    fig.update_yaxes(rangemode='tozero', scaleanchor='y', scaleratio=1, constraintoward='bottom', secondary_y=True)
    fig.update_yaxes(rangemode='tozero', scaleanchor='y2', scaleratio=0.02, constraintoward='bottom', secondary_y=False)
  
   


    fig.layout.xaxis.title="Days"
    fig.layout.yaxis.title="High Boilers & Blow down qaunaity"
    fig.layout.yaxis2.title="Kettle level before blowdown"

    



    fig2 = optuna.visualization.plot_optimization_history(study_opt[seq])
    fig2.update_layout(showlegend=True, autosize=True, width=1500, height=500,yaxis_title="Annual net contribution(cr)",xaxis_title="Iteration")
    

    
    fig3 = plot_parallel_coordinate(study_opt[seq])
    fig3.update_layout(showlegend=False, autosize=True, width=1500, height=500)
    


    # fig4 = plot_slice(study_opt[seq])
    # fig4.update_layout(showlegend=False, autosize=True, width=1500, height=500)
    # fig4.show()

    # fig5 = plot_contour(study_opt[seq])
    # fig5.update_layout(showlegend=False, autosize=True, width=1500, height=500)
    # fig5.show()


    fig6 = optuna.visualization.plot_param_importances(study_opt[seq])
    fig6.update_layout(showlegend=False, autosize=True, width=1000, height=500)
    


    fig7 = plot_edf(study_opt[seq])
    fig7.update_layout(showlegend=False, autosize=True, width=1000, height=500)
    

    if show:
        fig.show()
        fig2.show()
        fig3.show()
        fig6.show()
        fig7.show()


    return fig,fig2,fig3,fig6,fig7