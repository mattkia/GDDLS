import plotly.graph_objects as go
import plotly.express as px


adaptive_beta = False
l1_values_mae = False
l1_values_f = False
l2_values_mae = False
l2_values_f = False
epsilon_mae = False
epsilon_f = False
r_mae = False
r_f = True

if adaptive_beta:
    ##################################################
    # Plotting the Adaptive F scores
    ##################################################
    methods = ['DLS', 'LEGS', 'ELD', 'MCDL', 'MDF', 'MTDS', 'Ours']

    # PASCAL dataset
    values = [0.72, 0.69, 0.70, 0.67, 0.70, 0.65, 0.78]
    # ECSSD dataset
    values2 = [0.83, 0.78, 0.81, 0.76, 0.80, 0.81, 0.85]
    # OMRON dataset
    values3 = [0.65, 0.58, 0.62, 0.61, 0.63, 0.61, 0.70]

    fig = go.Figure([go.Bar(x=methods, y=values3)])
    fig.update_layout(title='Adaptive F Score',
                      xaxis_title='Method',
                      yaxis_title='Value')
    fig.show()
    ##################################################
elif l1_values_mae:
    ##################################################
    # Plotting the effect of lambda 1 on MAE
    ##################################################
    l1_values_mae = [0.001, 0.01, 0.3, 1, 5]
    # CAMO dataset
    camo_f_values = [0.21, 0.17, 0.13, 0.15, 0.27]
    # ECSSD dataset
    ecssd_f_values = [0.15, 0.11, 0.08, 0.10, 0.20]
    # PASCAL dataset
    pascal_f_values = [0.17, 0.15, 0.13, 0.16, 0.23]
    # DUT-OMRON dataset
    omron_mae_values = [0.13, 0.10, 0.07, 0.09, 0.12]
    # HKU-IS dataset
    hku_mae_values = [0.14, 0.11, 0.07, 0.10, 0.19]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=l1_values_mae, y=camo_f_values, line=dict(color='royalblue', width=4), name='CAMO'))
    fig.add_trace(go.Scatter(x=l1_values_mae, y=ecssd_f_values, line=dict(color='firebrick', width=4), name='ECSSD'))
    fig.add_trace(go.Scatter(x=l1_values_mae, y=pascal_f_values, line=dict(color='darkgreen', width=4), name='PASCAL'))
    fig.add_trace(go.Scatter(x=l1_values_mae, y=omron_mae_values, line=dict(color='darkslategrey', width=4), name='DUT-OMRON'))
    fig.add_trace(go.Scatter(x=l1_values_mae, y=hku_mae_values, line=dict(color='springgreen', width=4), name='HKU-IS'))

    fig.update_layout(title='Lambda 1 effect on MAE',
                      xaxis_title='Lambda 1',
                      yaxis_title='MAE')
    fig.show()
    ##################################################
elif l1_values_f:
    ##################################################
    # Plotting the effect of lambda 1 on F score
    ##################################################
    l1_values_mae = [0.001, 0.01, 0.3, 1, 5]
    # CAMO dataset
    camo_f_values = [0.53, 0.57, 0.61, 0.60, 0.51]
    # ECSSD dataset
    ecssd_f_values = [0.78, 0.80, 0.83, 0.79, 0.77]
    # PASCAL dataset
    pascal_f_values = [0.63, 0.71, 0.75, 0.73, 0.66]
    # DUT-OMRON dataset
    omron_f_values = [0.65, 0.68, 0.70, 0.69, 0.63]
    # HKU-IS dataset
    hku_f_values = [0.72, 0.78, 0.81, 0.79, 0.74]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=l1_values_mae, y=camo_f_values, line=dict(color='royalblue', width=4), name='CAMO'))
    fig.add_trace(go.Scatter(x=l1_values_mae, y=ecssd_f_values, line=dict(color='firebrick', width=4), name='ECSSD'))
    fig.add_trace(
        go.Scatter(x=l1_values_mae, y=pascal_f_values, line=dict(color='darkgreen', width=4), name='PASCAL'))
    fig.add_trace(
        go.Scatter(x=l1_values_mae, y=omron_f_values, line=dict(color='darkslategrey', width=4), name='DUT-OMRON'))
    fig.add_trace(go.Scatter(x=l1_values_mae, y=hku_f_values, line=dict(color='springgreen', width=4), name='HKU-IS'))

    fig.update_layout(title='Lambda 1 effect on F score',
                      xaxis_title='Lambda 1',
                      yaxis_title='F score')
    fig.show()
    ##################################################
elif l2_values_mae:
    ##################################################
    # Plotting the effect of lambda 2 on MAE
    ##################################################
    l2_values_mae = [-4, -3, -2, -1, 0]
    # CAMO dataset
    camo_f_values = [0.29, 0.21, 0.13, 0.15, 0.20]
    # ECSSD dataset
    ecssd_f_values = [0.17, 0.14, 0.08, 0.10, 0.23]
    # PASCAL dataset
    pascal_f_values = [0.25, 0.17, 0.13, 0.16, 0.30]
    # DUT-OMRON dataset
    omron_mae_values = [0.15, 0.10, 0.07, 0.09, 0.17]
    # HKU-IS dataset
    hku_mae_values = [0.16, 0.10, 0.07, 0.11, 0.19]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=l2_values_mae, y=camo_f_values, line=dict(color='royalblue', width=4), name='CAMO'))
    fig.add_trace(go.Scatter(x=l2_values_mae, y=ecssd_f_values, line=dict(color='firebrick', width=4), name='ECSSD'))
    fig.add_trace(
        go.Scatter(x=l2_values_mae, y=pascal_f_values, line=dict(color='darkgreen', width=4), name='PASCAL'))
    fig.add_trace(
        go.Scatter(x=l2_values_mae, y=omron_mae_values, line=dict(color='darkslategrey', width=4), name='DUT-OMRON'))
    fig.add_trace(go.Scatter(x=l2_values_mae, y=hku_mae_values, line=dict(color='springgreen', width=4), name='HKU-IS'))

    fig.update_layout(title='Lambda 2 effect on MAE',
                      xaxis_title='Log(Lambda2)',
                      yaxis_title='MAE')
    fig.show()
    ##################################################
elif l2_values_f:
    ##################################################
    # Plotting the effect of lambda 2 on F score
    ##################################################
    l2_values_mae = [-4, -3, -2, -1, 0]
    # CAMO dataset
    camo_f_values = [0.49, 0.55, 0.61, 0.57, 0.53]
    # ECSSD dataset
    ecssd_f_values = [0.72, 0.79, 0.83, 0.76, 0.70]
    # PASCAL dataset
    pascal_f_values = [0.67, 0.73, 0.75, 0.71, 0.63]
    # DUT-OMRON dataset
    omron_f_values = [0.59, 0.66, 0.70, 0.63, 0.57]
    # HKU-IS dataset
    hku_f_values = [0.69, 0.77, 0.81, 0.76, 0.71]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=l2_values_mae, y=camo_f_values, line=dict(color='royalblue', width=4), name='CAMO'))
    fig.add_trace(go.Scatter(x=l2_values_mae, y=ecssd_f_values, line=dict(color='firebrick', width=4), name='ECSSD'))
    fig.add_trace(
        go.Scatter(x=l2_values_mae, y=pascal_f_values, line=dict(color='darkgreen', width=4), name='PASCAL'))
    fig.add_trace(
        go.Scatter(x=l2_values_mae, y=omron_f_values, line=dict(color='darkslategrey', width=4), name='DUT-OMRON'))
    fig.add_trace(go.Scatter(x=l2_values_mae, y=hku_f_values, line=dict(color='springgreen', width=4), name='HKU-IS'))

    fig.update_layout(title='Lambda 2 effect on F score',
                      xaxis_title='Log(Lambda2)',
                      yaxis_title='F score')
    fig.show()
    ##################################################
elif epsilon_mae:
    ##################################################
    # Plotting the effect of epsilon on MAE
    ##################################################
    eps_values = [1, 1/4, 1/32, 1/64, 1/128]
    # CAMO dataset
    camo_eps_values = [0.23, 0.19, 0.13, 0.14, 0.15]
    # ECSSD dataset
    ecssd_eps_values = [0.14, 0.11, 0.08, 0.09, 0.10]
    # PASCAL dataset
    pascal_eps_values = [0.21, 0.16, 0.13, 0.12, 0.14]
    # DUT-OMRON dataset
    omron_eps_values = [0.17, 0.10, 0.07, 0.08, 0.08]
    # HKU-IS dataset
    hku_eps_values = [0.16, 0.09, 0.07, 0.07, 0.08]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=eps_values, y=camo_eps_values, line=dict(color='royalblue', width=4), name='CAMO'))
    fig.add_trace(go.Scatter(x=eps_values, y=ecssd_eps_values, line=dict(color='firebrick', width=4), name='ECSSD'))
    fig.add_trace(
        go.Scatter(x=eps_values, y=pascal_eps_values, line=dict(color='darkgreen', width=4), name='PASCAL'))
    fig.add_trace(
        go.Scatter(x=eps_values, y=omron_eps_values, line=dict(color='darkslategrey', width=4), name='DUT-OMRON'))
    fig.add_trace(go.Scatter(x=eps_values, y=hku_eps_values, line=dict(color='springgreen', width=4), name='HKU-IS'))

    fig.update_layout(title='epsilon effect on MAE',
                      xaxis_title='epsilon',
                      yaxis_title='MAE')
    fig.show()

    ##################################################
elif epsilon_f:
    ##################################################
    # Plotting the effect of epsilon on F score
    ##################################################
    eps_values = [1, 1/4, 1/32, 1/64, 1/128]
    # CAMO dataset
    camo_eps_values = [0.55, 0.57, 0.61, 0.53, 0.49]
    # ECSSD dataset
    ecssd_eps_values = [0.77, 0.79, 0.83, 0.75, 0.70]
    # PASCAL dataset
    pascal_eps_values = [0.67, 0.73, 0.75, 0.66, 0.63]
    # DUT-OMRON dataset
    omron_eps_values = [0.65, 0.67, 0.70, 0.63, 0.56]
    # HKU-IS dataset
    hku_eps_values = [0.76, 0.78, 0.81, 0.76, 0.69]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=eps_values, y=camo_eps_values, line=dict(color='royalblue', width=4), name='CAMO'))
    fig.add_trace(go.Scatter(x=eps_values, y=ecssd_eps_values, line=dict(color='firebrick', width=4), name='ECSSD'))
    fig.add_trace(
        go.Scatter(x=eps_values, y=pascal_eps_values, line=dict(color='darkgreen', width=4), name='PASCAL'))
    fig.add_trace(
        go.Scatter(x=eps_values, y=omron_eps_values, line=dict(color='darkslategrey', width=4), name='DUT-OMRON'))
    fig.add_trace(go.Scatter(x=eps_values, y=hku_eps_values, line=dict(color='springgreen', width=4), name='HKU-IS'))

    fig.update_layout(title='epsilon effect on F score',
                      xaxis_title='epsilon',
                      yaxis_title='F score')
    fig.show()
    ##################################################
elif r_mae:
    ##################################################
    # Plotting the effect of R on MAE
    ##################################################
    r_values = [1, 2, 4, 6, 8]
    # CAMO dataset
    camo_r_values = [0.133, 0.137, 0.140, 0.139, 0.131]
    # ECSSD dataset
    ecssd_r_values = [0.083, 0.083, 0.082, 0.082, 0.080]
    # PASCAL dataset
    pascal_r_values = [0.136, 0.133, 0.134, 0.132, 0.131]
    # DUT-OMRON dataset
    omron_r_values = [0.077, 0.076, 0.075, 0.074, 0.074]
    # HKU-IS dataset
    hku_r_values = [0.076, 0.075, 0.068, 0.065, 0.063]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=r_values, y=camo_r_values, line=dict(color='royalblue', width=4), name='CAMO'))
    fig.add_trace(go.Scatter(x=r_values, y=ecssd_r_values, line=dict(color='firebrick', width=4), name='ECSSD'))
    fig.add_trace(
        go.Scatter(x=r_values, y=pascal_r_values, line=dict(color='darkgreen', width=4), name='PASCAL'))
    fig.add_trace(
        go.Scatter(x=r_values, y=omron_r_values, line=dict(color='darkslategrey', width=4), name='DUT-OMRON'))
    fig.add_trace(go.Scatter(x=r_values, y=hku_r_values, line=dict(color='springgreen', width=4), name='HKU-IS'))

    fig.update_layout(title='R effect on MAE',
                      xaxis_title='R',
                      yaxis_title='MAE')
    fig.show()
    ##################################################
elif r_f:
    ##################################################
    # Plotting the effect of R on F Score
    ##################################################
    r_values = [1, 2, 4, 6, 8]
    # CAMO dataset
    camo_r_values = [0.611, 0.606, 0.608, 0.613, 0.617]
    # ECSSD dataset
    ecssd_r_values = [0.828, 0.825, 0.826, 0.830, 0.832]
    # PASCAL dataset
    pascal_r_values = [0.752, 0.755, 0.757, 0.756, 0.759]
    # DUT-OMRON dataset
    omron_r_values = [0.708, 0.709, 0.713, 0.714, 0.718]
    # HKU-IS dataset
    hku_r_values = [0.813, 0.815, 0.821, 0.827, 0.833]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=r_values, y=camo_r_values, line=dict(color='royalblue', width=4), name='CAMO'))
    fig.add_trace(go.Scatter(x=r_values, y=ecssd_r_values, line=dict(color='firebrick', width=4), name='ECSSD'))
    fig.add_trace(
        go.Scatter(x=r_values, y=pascal_r_values, line=dict(color='darkgreen', width=4), name='PASCAL'))
    fig.add_trace(
        go.Scatter(x=r_values, y=omron_r_values, line=dict(color='darkslategrey', width=4), name='DUT-OMRON'))
    fig.add_trace(go.Scatter(x=r_values, y=hku_r_values, line=dict(color='springgreen', width=4), name='HKU-IS'))

    fig.update_layout(title='R effect on F score',
                      xaxis_title='R',
                      yaxis_title='F score')
    fig.show()
    ##################################################
