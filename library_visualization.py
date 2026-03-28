# basic packages
import os
os.environ["QT_LOGGING_RULES"] = "qt.qpa.plugin=false"
import json
import numpy as np
import seaborn as sns
import progressbar as pb
from matplotlib import cm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.style.use('seaborn-v0_8-paper')

# packages for defect plotting
from plot_defect import nematic_plot
from basic_tools import defect_detector, static_com_frame

def add_colorbar(axis, min_val: float=0.0, max_val: float=1.0, n_val: int=9,
                 my_color_map: str='icefire_r', my_color_palt=None,
                 plot_color_bar=False, my_color_label: str='color_bar'):

    # colobar palatte and normalization
    color_vals = np.linspace(min_val, max_val, n_val)
    color_norm = cm.colors.Normalize(min_val, max_val)
    color_palt = sns.color_palette(my_color_map, as_cmap=True)
    if my_color_palt:
        color_palt = my_color_palt

    if plot_color_bar:
        color_axs = make_axes_locatable(axis).append_axes("bottom", "5%", pad="3%")
        color_bar = plt.colorbar(cm.ScalarMappable(cmap=color_palt, norm=color_norm),
            cax=color_axs, orientation='horizontal', ticks = color_vals)
        color_bar.ax.tick_params(labelsize=20)
        color_bar.set_label(my_color_label, fontsize=20, labelpad=10)
        color_bar.ax.set_xticklabels(np.around(color_vals, decimals=2))

    return color_norm, color_palt

def plot_scalar_field_2D(axis, x_grid, y_grid, s_data, s_min: float=0.0,
                         s_max: float=1.0, s_val: int=9, s_map='icefire_r',
                         if_color_palt=None, if_color_bar=False,
                         scalar_label='color_bar', plot_val=None):

    cnorm, cpalt = add_colorbar(axis, s_min, s_max, s_val, s_map,
                                my_color_palt=if_color_palt,
                                plot_color_bar=if_color_bar,
                                my_color_label=scalar_label)
    if plot_val:
        axis.contourf(x_grid, y_grid, s_data, plot_val, norm=cnorm, cmap=cpalt)
    else:
        axis.contourf(x_grid, y_grid, s_data, 400, norm=cnorm, cmap=cpalt)

def plot_membrane_position_1D(axis, position, box, pt=None):

    # membrane position in lab frame
    pos_x = np.mod(position[0,] - np.mean(position[0,]) + box[0]/2, box[0])
    pos_y = np.mod(position[1,] - np.mean(position[1,]) + box[1]/2, box[1])

    axis.scatter(pos_x, pos_y, s=64, c="#a05600", marker='o', zorder=2)    
    axis.scatter(pos_x[0], pos_y[0], s=100, marker='o', color='red')
    if pt:
        axis.scatter(pos_x[pt], pos_y[pt], s=100, marker='o', color='red')

def plot_defect_positions(axis, nematic, scale):

    # find centroid of the defect 
    centroids_p, centroids_m = defect_detector(
        np.transpose(nematic, (0,2,1)), filter_radius=1, area_threshold=1)

    axis.scatter(centroids_p[:,0] * scale, centroids_p[:,1] * scale, alpha=1.0,
        s=64, facecolors='none', edgecolors='#fb5324', linewidths=2, zorder=3)
    axis.scatter(centroids_m[:,0] * scale, centroids_m[:,1] * scale, alpha=1.0,
        s=64, facecolors='none', edgecolors='#9ac140', linewidths=2, zorder=3)

def plot_vortex_cores(axis, velocity, scale):

    v_scaled = np.empty_like(velocity)
    v_mag = np.linalg.norm(velocity, axis=0)

    v_scaled[0,] = np.divide(velocity[0,], v_mag, out=np.zeros_like(v_mag),
        where=v_mag!=0)
    v_scaled[1,] = np.divide(velocity[1,], v_mag, out=np.zeros_like(v_mag),
        where=v_mag!=0)

    # find centroid of the defect 
    centroids, _ = defect_detector(np.transpose(v_scaled, (0,2,1)),
        filter_radius=1, area_threshold=1)
    axis.scatter(centroids[:,0] * scale, centroids[:,1] * scale,
        s=180, c='#4a00b3', marker='*')

def movie_maker(
    par,
    nematic=None, n_color=None, n_colormap=None, n_label=None,
    velocity=None, v_color=None, v_colormap=None, v_label=None,
    membrane=None, m_color=None, m_colormap=None, m_label=None,
    shape_zero=None):

    # SAVE PARAMETERS
    fps = par['fps']
    save = par['save']

    # GRID AND TIME PARAMETERS
    time = par['time']                               
    x_list, y_list = par['x_series'], par['y_series']     
    y_grid, x_grid = np.meshgrid(y_list, x_list)           
    dh = x_list[1]                                          
    lx, ly = int(dh + x_list[-1]), int(dh + y_list[-1])     
    box = [lx, ly]                                          

    ratio = int(lx/ly)
    figure_box = [ratio * 10, 10 + 2]

    # CALCULATE COLORMAP RANGE
    if par['plot_nematics']:
        n_min = n_color.min()
        n_max = n_color.max()
    if par['plot_velocity']:
        v_min = v_color.min()
        v_max = v_color.max()
        v_max = np.max((v_max, np.abs(v_min)))
        v_min = -v_max    
    if par['plot_membrane']:
        pt = par['red_point']

    for t in pb.progressbar(range(len(time))):

        if par['plot_velocity']:

            fig = plt.figure(1, dpi=100, figsize=figure_box)
            ax = fig.add_subplot(111, box_aspect=1/ratio)
            ax.set_xlim(x_list[0], x_list[-1])
            ax.set_ylim(y_list[0], y_list[-1])
            ax.tick_params(
                left=False, bottom=False, labelleft=False, labelbottom=False)
            ax.set_title('time ({0:.2f})'.format(time[t]), fontsize=20,
                color='black', pad=10)

            ax.streamplot(
                np.array(x_list), np.array(y_list),
                velocity[t,0,].T, velocity[t,1,].T,
                density=4.0, linewidth=2, color='#30303b')
            plot_scalar_field_2D(
                ax, x_grid, y_grid, v_color[t,], v_min, v_max,
                s_map=v_colormap, if_color_bar=True, scalar_label=v_label)
            plot_vortex_cores(ax, velocity[t,], dh)

            if par['plot_membrane']:
                plot_membrane_position_1D(ax, membrane[t, ], box, pt)
            
            fig.set_tight_layout(True)
            plt.savefig(f'{save}_media/.u_{t:05d}.png')
            plt.close(fig)

        if par['plot_nematics']:

            fig = plt.figure(1, dpi=600, figsize=figure_box)
            ax = fig.add_subplot(111, box_aspect=1/ratio)
            ax.set_xlim(x_list[0], x_list[-1])
            ax.set_ylim(y_list[0], y_list[-1])
            ax.tick_params(
                left=False, bottom=False, labelleft=False, labelbottom=False)
            ax.set_title('time ({0:.2f})'.format(time[t]), fontsize=20,
                color='black', pad=10)

            nematic_plot(
                np.array(x_list), np.array(y_list),
                nematic[t,0,].T, nematic[t,1,].T,
                density=4, linewidth=2, color="#1d407d")
            plot_scalar_field_2D(
                ax, x_grid, y_grid, n_color[t,], n_min, n_max,
                s_map=n_colormap, if_color_bar=True,
                scalar_label=n_label, plot_val=20)
            plot_defect_positions(ax, nematic[t,], dh)

            if par['plot_membrane']:
                plot_membrane_position_1D(ax, membrane[t, ], box, pt)

            fig.set_tight_layout(True)
            plt.savefig(f'{save}_media/.n_{t:05d}.png')
            plt.close(fig)

    # STITCH AND DELETE FRAMES
    if par['plot_velocity']:
        os.system(f'ffmpeg -r {fps:02d} -y -i {save}_media/.u_%05d.png' +
            ' -vf "scale=1920:trunc(ih/2)*2" -vcodec libx264 -pix_fmt yuv420p' +
            f'-crf 18 -preset slow -nostats -loglevel 0 {save}__velocity.mov')
        if par['remove_frames']:
            os.system(f'rm -f {save}_media/.u_*')

    if par['plot_nematics']:
        os.system(f'ffmpeg -r {fps:02d} -y -i {save}_media/.n_%05d.png' +
            ' -vf "scale=1920:trunc(ih/2)*2" -vcodec libx264 -pix_fmt yuv420p' +
            f'-crf 18 -preset slow -nostats -loglevel 0 {save}__nematic.mov')
        if par['remove_frames']:
            os.system(f'rm -f {save}_media/.n_*')

if __name__ == '__main__':

    from solver_fields import membraneField
    from basic_tools import (load_data, fluid_measures,
        nematic_measures, nematic_elastic_energy)

    # DATA AND SAVE PATH
    path = '/home/saaransh/Active/data'   # PATH 
    tag = '260328_175238'                 # FILE TIME TAG
    load = f'{path}/{tag}'
    os.system(f'mkdir {load}_media')

    # CHOOSE THE FILE NUMBER (USUALLY ONE TO AVOID LOADING TOO MUCH DATA)
    file_list = [1]
    # CHOOSE THE TIME START:END:SKIP
    # START,END (float) -> [0,1], SKIP (int) -> [1,max)
    t_sample = [0.0,1.0,10]
    # SKIP SPATIAL POINTS 
    skip = 1

    # MOVIE PARAMETERS
    ap = {'fps': 25,
        'plot_velocity': True,
        'plot_nematics': True,
        'plot_membrane': True,
        'remove_frames': False}

    # LOAD PARAMETERS
    assert os.path.isfile(f'{load}__parameters.json'), 'no parameter file found'
    with open(f'{load}__parameters.json') as jsonFile:
        P = json.load(jsonFile)

    if ap['plot_membrane']:
        
        membrane = membraneField(P)
        n_b = membrane.n_b

        ap['red_point'] = 0
        if membrane.count == 2:
            ap['red_point'] = membrane.n_out

        next_idx = membrane.fore
        last_idx = membrane.back

    sampling_dt =  P['TIME_STEP'] * P['SKIP']
    n_time = int(P['MAX_TIME']/sampling_dt)
    t_sample[0] = int(t_sample[0] * n_time)
    t_sample[1] = int(t_sample[1] * n_time)
    time = np.linspace(0, P['MAX_TIME'], n_time, endpoint=False)
    time = time[t_sample[0]:t_sample[1]:t_sample[2]]

    # generate spatial domain
    dh = P['SPACE_STEP']
    n_x = int(P['X_SIZE']/dh)
    n_y = int(P['Y_SIZE']/dh)
    box = np.array([P['X_SIZE'], P['Y_SIZE']])
    x_list = np.linspace(0, P['X_SIZE'], n_x, endpoint=False)[::skip]
    y_list = np.linspace(0, P['Y_SIZE'], n_y, endpoint=False)[::skip]

    # ADD PARAMETERS
    ap['x_series'] = x_list.tolist()
    ap['y_series'] = y_list.tolist()
    ap['time'] = time.tolist()
    ap['save'] = load

    # INITIATE DATA
    N_data, N_color, N_map, N_label = None, None, None, None
    U_data, U_color, U_map, U_label = None, None, None, None
    X_data, X_color, X_map, X_label = None, None, None, None
    X_init = None

    print('... loading data')
    if ap['plot_membrane']:

        X = load_data(f'{load}_0__membrane.h5', file_list)
        M = load_data(f'{load}_0__mask.h5', file_list, t_sample) > 0.9

        X_init = X[0,]
        X_data = X[t_sample[0]:t_sample[1]:t_sample[2],]
        M = static_com_frame(X_data, M, dh, box)
        del X

    if ap['plot_velocity']:

        U = load_data(f'{load}_0__velocity.h5', file_list, t_sample)
        U_color = fluid_measures(
            U, dh, return_this='vorticity')[...,::skip,::skip]

        U_data = U[..., ::skip, ::skip]
        if ap['plot_membrane']:
            U_data *= M[..., None, ::skip, ::skip]

        U_label = 'vorticity'
        U_map = 'coolwarm_r'
        del U

    if ap['plot_nematics']:

        Q = load_data(f'{load}_0__nematic.h5', file_list, t_sample)
        Q = static_com_frame(X_data, Q, dh, box)
        N_data = nematic_measures(
            Q, only_director=True)[..., ::skip, ::skip]
        _, NB = nematic_elastic_energy(
            Q, dh, P['FRANK_CONST'])
        del Q

        if ap['plot_membrane']:
            N_data *= M[..., None, ::skip, ::skip]

        N_color = NB[..., ::skip, ::skip]
        N_label = 'elastic energy'
        N_map = 'GnBu_r'

    print('... plotting')
    movie_maker(ap,
        N_data, N_color, N_map, N_label,
        U_data, U_color, U_map, U_label,
        X_data, X_color, X_map, X_label, X_init)      