import h5py
import numpy as np
from scipy.signal import convolve2d
from skimage.measure import label, regionprops

# METHODS FOR MATHEMATICAL OPERATIONS

def roll(
    data: np.ndarray,
    shift: int=0,
    axis: int=0) -> np.ndarray:

    ''' ROLL ARRAY ELEMENTS ALONG A GIVEN AXIS

        ARGUMENTS
        ---------
        data: numpy.ndarray
            input data
        shift: int
            # of places moved (- backward, + forward)
        axis: int
            choice of axis

        RETURNS
        -------
        rolled: numpy.ndarray
            output data '''

    # 0. CHECK IF INPUT DIMENSIONS ARE VALID
    dim = data.ndim
    if axis >= dim or axis < -dim:
        raise ValueError('axis is out of bounds for the data')

    if shift != 0: 

        rolled = np.empty_like(data)

        # generate index slices for all axis
        fill_1 = [slice(None)] * dim
        fill_2 = [slice(None)] * dim
        grab_1 = [slice(None)] * dim
        grab_2 = [slice(None)] * dim

        # create slice for shift
        fill_1[axis] = slice(shift,None,None)
        grab_1[axis] = slice(None,-shift,None)
        fill_2[axis] = slice(None,shift,None)
        grab_2[axis] = slice(-shift,None,None)

        # grab and fill the data
        rolled[tuple(fill_1)] = data[tuple(grab_1)]
        rolled[tuple(fill_2)] = data[tuple(grab_2)]

        return rolled

    else: return data

def first_order_derivative(
    data: np.ndarray,
    step: float=0.5,
    axis: int=0,
    boundary: str='periodic',
    bulk: str='central'):
    ''' CALCULATE FIRST ORDER DERIVATIVE
    
        ARGUMENTS
        ---------
        data: numpy nd.array
            input data
        step: float
            discrete step size for the derivative 
        axis: int
            axis for higher dimensional data
        boundary: str
            choice of boundary condition
        bulk: str
            choice of derivative scheme
    
        RETURN
        ------
        derivative: numpy nd.array
            same shape as input data '''
    
    # CHECK IF INPUT DIMENSIONS ARE VALID #
    dim = data.ndim
    if axis >= dim or axis < -dim:
        raise ValueError('axis is out of bounds for the data')

    # assume periodic boundary condition for the bulk
    if bulk == 'central':
        derivative = (roll(data, -1, axis) - roll(data, 1, axis))/(2 * step)
    elif bulk == 'forward':
        derivative = (roll(data, -1, axis) - data)/step
    elif bulk == 'backward':
        derivative = (data - roll(data, 1, axis))/step

    # create a list of index slicing for the boundary
    indices = [slice(None)] * dim

    if boundary == 'dirichlet':

        front = indices.copy()
        front[axis] = 0
        front_one = indices.copy()
        front_one[axis] = 1

        back = indices.copy()
        back[axis] = - 1
        back_one = indices.copy()
        back_one[axis] = - 2

        derivative[tuple(front)] = (data[tuple(front_one)] - data[tuple(front)]) / step
        derivative[tuple(back)] = (data[tuple(back)] - data[tuple(back_one)]) / step

    if boundary == 'neumann':

        front = indices.copy()
        front[axis] = 0

        back = indices.copy()
        back[axis] = - 1

        derivative[tuple(front)] = 0
        derivative[tuple(back)] = 0

    return derivative

def second_order_derivative(
    data: np.ndarray,
    step: float=1.0,
    axis: int=0,
    boundary: str='periodic'):
    ''' CALCULATE SECOND ORDER DERIVATIVE USING CENTRAL DIFFERENCE (upto 3D array)
    
        ARGUMENTS
        ---------
        data: numpy nd.array
            input data
        step: float
            discrete step size for the derivative 
        axis: int
            axis for higher dimensional data
        boundary: str
            choice of boundary condition

        RETURN
        ------
        derivative: numpy nd.array
            same shape as input data '''

    # CHECK IF INPUT DIMENSIONS ARE VALID #
    dim = data.ndim
    if axis >= dim or axis < -dim:
        raise ValueError('axis is out of bounds for the data')
    
    # calculate derivative ---
    derivative = (roll(data, -1, axis) + roll(data, 1, axis) - 2*data)/(step**2)

    # create a list of index slicing for the boundary
    indices = [slice(None)] * data.ndim

    if boundary == 'dirichlet':

        front = indices.copy()
        front[axis] = 0
        front_one = indices.copy()
        front_one[axis] = 1
        front_two = indices.copy()
        front_two[axis] = 2

        back = indices.copy()
        back[axis] = - 1
        back_one = indices.copy()
        back_one[axis] = - 2
        back_two = indices.copy()
        back_two[axis] = - 3

        derivative[tuple(front)] = (data[tuple(front_two)] - 2*data[tuple(front_one)]
                                    + data[tuple(front)])/step**2
        derivative[tuple(back)] = (data[tuple(back_two)] - 2*data[tuple(back_one)]
                                   + data[tuple(back)])/step**2

    if boundary == 'neumann':

        front = indices.copy()
        front[axis] = 0
        front_one = indices.copy()
        front_one[axis] = 1

        back = indices.copy()
        back[axis] = - 1
        back_one = indices.copy()
        back_one[axis] = - 2

        derivative[tuple(front)] = 2*(data[tuple(front_one)] - data[tuple(front)])/step**2
        derivative[tuple(back)] = 2*(data[tuple(back_one)] - data[tuple(back)])/step**2

    return derivative

def laplacian(
    data: np.ndarray,
    step: float=1.0,
    time: bool=False,
    boundary: str='periodic'):
    ''' CALCULATE LAPLACIAN USING CENTRAL DIFFERENCE 
    
        ARGUMENTS
        ---------
        data: numpy.ndarray
            input data
        step: float
            discrete step size for the derivative 
        time: bool
            if first axis is time
        boundary: str
            choice of boundary condition
    
        RETURN
        ------
        derivative: numpy.ndarray
            output data ''' 

    if time:

        # declare output array
        derivative = np.zeros_like(data)

        for i in range(data.shape[0]):
            derivative[i,] = laplacian(data[i,], step, boundary=boundary)

    else: 

        # declare output array
        derivative = np.zeros_like(data)

        for j in range(data.ndim):
            derivative += second_order_derivative(data, step, j, boundary)

    return derivative

def interpolate_grid(
    data,
    grid_from,
    grid_to):
    ''' INTERPOLATE DATA FROM ONE 2D GRID TO ANOTHER (assumes PBC, upto 3d array)
    
    ARGUMENTS
    ---------
    data: numpy nd.array [data.shape()] input data
    grid_from: str [1] choose 'centered' or 'stagger_x' or 'stagger_y'
    grid_to: str [1] choose 'centered' or 'stagger_x' or 'stagger_y'

    RETURNS
    -------
    data_new: numpy array [N1, N2] interpolated data '''

    # no of spatial dimensions
    n_dim = data.ndim

    if n_dim == 2:

        if grid_from == 'centered':

            if grid_to == 'stagger_x': data_new = (roll(data, -1, 0) + data)/2

            if grid_to == 'stagger_y': data_new = (roll(data, -1, 1) + data)/2

        if grid_from == 'stagger_x':

            if grid_to == 'centered': data_new = (roll(data, 1, 0) + data)/2

            if grid_to == 'stagger_y': data_new = (roll(data, 1, 0) + roll(data, -1, 1)
                                                    + roll(roll(data, 1, 0), -1, 1) + data)/2

        if grid_from == 'stagger_y':
            
            if grid_to == 'centered': data_new = (roll(data, 1, 1) + data)/2

            if grid_to == 'stagger_x': data_new = (roll(data, -1, 0) + roll(data, 1, 1)
                                                    + roll(roll(data, -1, 0), 1, 1) + data)/2

    if n_dim == 3:

        data_new = np.empty(data.shape)

        for i in range(data.shape[0]):
            data_new[i,] = interpolate_grid(data[i,], grid_from, grid_to)

    return data_new

# METHODS FOR HANDLING DATA

def load_data(
    load_path: str,
    file_list: list,
    time: tuple=(0,None,1),
    skips: tuple=(1,1)) -> np.ndarray:

    ''' Load multiple H5 files in one numpy array

    ARGUMENTS
    ---------
        load_path: str
            full path of the file to be loaded (must contain _0_ tag)
        file_list: list
            list of file numbers to load
        time: tuple
            (0,None,1) selects the entire data
        skips: tuple
            (1,1) selects the entire data

    RETURNS
    -------
        data: np.ndarray
            data from the file numbers specified '''

    data = []

    # 1. CREATE SLICE FOR 1st DATASET ----
    selection_1 = np.s_[time[0]:time[1]:time[2], ..., ::skips[0], ::skips[1]]
    selection_2 = np.s_[0:None:time[2], ..., ::1, ::1]

    # 2. LOAD THE DATA ----
    for i,num in enumerate(file_list):
        file_path = load_path.replace('_0_', f'_{num}_')
        with h5py.File(file_path, 'r') as file:
            dset = file.get('dataset')
            if i == 0:
                chunk = dset[selection_1]
            else:
                chunk = dset[selection_2]
            data.append(chunk)

    return np.concatenate(data, axis=0)

def discrete_delta(
    pos_x,
    pos_y):
    ''' GENERATES A FOUR POINT REGULARIZED DELTA FUNTION IN 2D

    ARGUMENTS
    ---------
    pos_x : numpy 1d.array [Nb] normalized x position of membrane points
    pos_y : numpy 1d.array [Nb] normalized y position of membrane points

    RETURNS
    -------
    delta_xy : numpy array [Nb, 4, 4] regularized delta function '''

    # declare output arrays
    n_memb = pos_x.shape[0]
    delta_x = np.empty((n_memb, 4, 4))
    delta_y = np.empty((n_memb, 4, 4))
 
    qx = np.sqrt(1 + 4*pos_x*(1-pos_x))
    qy = np.sqrt(1 + 4*pos_y*(1-pos_y))

    delta_x[:, 0, :] = ((3 - 2*pos_x - qx)/8)[:, None]
    delta_x[:, 1, :] = ((3 - 2*pos_x + qx)/8)[:, None]
    delta_x[:, 2, :] = ((1 + 2*pos_x + qx)/8)[:, None]
    delta_x[:, 3, :] = ((1 + 2*pos_x - qx)/8)[:, None]

    delta_y[:, :, 0] = ((3 - 2*pos_y - qy)/8)[:, None]
    delta_y[:, :, 1] = ((3 - 2*pos_y + qy)/8)[:, None]
    delta_y[:, :, 2] = ((1 + 2*pos_y + qy)/8)[:, None]
    delta_y[:, :, 3] = ((1 + 2*pos_y - qy)/8)[:, None]

    delta_xy = delta_x * delta_y
    del delta_x, delta_y

    return delta_xy

def static_com_frame(
    membrane_position: np.ndarray,
    data: np.ndarray,
    grid_size: float,
    box: tuple) -> np.ndarray:

    box_array = np.array(box).reshape(2, 1)
    n_box = (box_array/grid_size).astype(int)

    if data.ndim == 3:

        X = membrane_position
        n_time = X.shape[0]

        # 1. COMPUTE CENTER OF MASS ----
        # logic: Average over the last axis (N points) -> new shape (..., 2, 1)
        # keepdims=True ensures we can broadcast back for subtraction
        com = np.mean(X, axis=-1, keepdims=True)
        index_com = np.floor_divide(com, grid_size).astype(int) - n_box/2
        move = np.mod(index_com, n_box).astype(int)

        for i in range(n_time):
            data[i,] = roll(roll(data[i,], -move[i,0,0], 0), -move[i,1,0], 1)

    if data.ndim == 4:

        n_comp = data.shape[1]

        for j in range(n_comp):
            data[:,j,] = static_com_frame(membrane_position, data[:,j,],
                grid_size, box)

    return data

# METHODS FOR MEMBRANE 

def forward_vector(
    membrane_position: np.ndarray,
    forward_index: np.ndarray,
    simulation_box: np.ndarray):
    ''' CALCULATES THE FORWARD CONNECTION VECTOR AND MAGNITUDE

    ARGUMENTS
    ---------
        membrane_position : numpy.ndarray
            either in D=1, D=2 spacetime dimensions
        forward_index : numpy.ndarray
            index of the next point
        simulation_box : numpy.ndarray
            simulation domain array

    RETURNS
    -------
        f_vect: numpy.ndarray
            same size as the input data 
        f_norm: numpy.ndarray
            length of f_vect '''

    X = membrane_position
    J = forward_index
    B = simulation_box

    # forward connection index
    f_connectome = (np.arange(2)[:,None], J)

    if X.ndim == 2:

        # calculate forward vector
        f_vect = X[f_connectome] - X
        # f_vect -= B * np.floor((f_vect + B/2.0)/B)
        f_vect = (f_vect.T - B*((f_vect.T + B/2)//B)).T
        f_norm = np.sqrt(f_vect[0,:]**2 + f_vect[1,:]**2)

    if X.ndim == 3:

        # declare output np.ndarray
        (n_time, _, n_space) = X.shape
        f_vect = np.empty((n_time, 2, n_space))
        f_norm = np.empty((n_time, n_space))

        for t in range(n_time):

            f_vect[t,], f_norm[t,] = forward_vector(X[t,], J, B)

        del X

    return f_vect, f_norm

def normal_and_tangent(
    membrane_position: np.ndarray,
    forward_index: np.ndarray,
    simulation_box: np.ndarray):
    ''' CALCULATES THE NORMAL AND TANGENT OF THE 1D MEMBRANE 

    ARGUMENTS
    ---------
    membrane_position: (np.ndarray) either in D=1, D=2 spacetime dimensions
    forward_index: (np.ndarray) index of the next point
    simulation_box: (np.ndarray) simulation domain array

    RETURNS
    -------
    norm: (np.ndarray) size of the forward vector
    normal: (np.ndarray) normal vector same shape as input
    tangent: (np.ndarray) tangent vector same shape as input '''

    X = membrane_position
    J = forward_index
    B = simulation_box

    vect, norm = forward_vector(X, J, B)

    if X.ndim == 2:

        tangent = vect/norm
        normal = (roll(tangent, 1, 0).T * np.array([-1,1])).T

    if X.ndim == 3:

        # declare output np.ndarray
        (n_time, _, n_space) = X.shape
        tangent = np.empty((n_time, 2, n_space))
        normal = np.empty((n_time, 2, n_space))

        for t in range(n_time):
            tangent[t,] = vect[t,]/norm[t,]
            normal[t,] = (roll(tangent[t,], 1, 0).T * np.array([-1,1])).T

    return norm, normal, tangent

def stretch_force(
    membrane_position,
    node_idx,
    next_idx,
    last_idx,
    delta,
    box):
    ''' CALCULATES THE STRETCH FORCE IN THE LAGRANGIAN COORDINATES

    ARGUMENTS
    ---------
    membrane_position : numpy 2d.array [2, Nb] membrane position
    forward_index
    delta : float [1] measure of the membrane discretization
    box_size : str [1] periodic or non-periodic

    RETURNS
    -------
    stretch_force : numpy 2d.array [2, Nb] stretching force '''

    # extract input data
    X = membrane_position
    L = box

    # connection indexing
    f_connectome = (np.arange(2)[:,None], next_idx)
    b_connectome = (np.arange(2)[:,None], last_idx)
    pick = np.where((node_idx - last_idx) != 0, 1, 0)

    # calculate forward and backward vectors
    f_vect = X[f_connectome] - X
    del X

    f_vect = (f_vect.T - L*((f_vect.T + L/2)//L)).T
    f_norm = np.sqrt(f_vect[0,:]**2 + f_vect[1,:]**2)

    # regularize 0/0 artifacts
    eps = 1E-10
    f_vect[:,(np.abs(f_norm) < eps)] = 0
    f_norm[(np.abs(f_norm) < eps)] = 1

    # calculate forward and backward forces
    f_force = (1/delta - 1/f_norm) * f_vect
    b_force = pick * (f_force[b_connectome])
    del f_vect, f_norm

    stretch_force = f_force - b_force
    del f_force, b_force

    return stretch_force/delta

def bending_force(
    membrane_position,
    spont_curvature,
    node_idx,
    next_idx,
    last_idx,
    delta,
    box):
    ''' CALCULATES THE BENDING FORCE IN THE LAGRANGIAN COORDINATES

    ARGUMENTS
    ---------
    membrane_position : numpy 2d.array [2, Nb] membrane position
    delta : float [1] measure of the membrane discretization
    spont_curvature : float/numpy 2d.array [1]/[2, Nb] local curvature 
    boundary : str [1] periodic or non-periodic

    RETURNS
    -------
    bending_force : numpy 2d.array [2, Nb] bending_force '''

    # extract input data
    X = membrane_position
    L = box

    # connection indexing
    f_connectome = (np.arange(2)[:,None], next_idx)
    b_connectome = (np.arange(2)[:,None], last_idx)
    f_pick = np.where((next_idx - node_idx) != 0, 1, 0)
    b_pick = np.where((node_idx - last_idx) != 0, 1, 0)
    pick = f_pick * b_pick

    # calculate forward and backward vectors
    f_vect = X[f_connectome] - X
    b_vect = X - X[b_connectome]
    del X

    f_vect = (f_vect.T - L*((f_vect.T + L/2)//L)).T
    b_vect = (b_vect.T - L*((b_vect.T + L/2)//L)).T

    # calculate curvature force vector
    X_vect = f_vect - b_vect
    X_norm = np.sqrt(X_vect[0,:]**2 + X_vect[1,:]**2)
    del f_vect, b_vect

    # regularize 0/0 artifacts
    eps = 1E-8
    X_vect[:, (np.abs(X_norm) < eps)] = 0
    X_norm[(np.abs(X_norm) < eps)] = 1

    force = (1/delta**2 - spont_curvature/X_norm) * X_vect
    force = pick * force
    del X_vect, X_norm

    bend_force = force[f_connectome] - 2*force + force[b_connectome]
    del force

    return -bend_force/(delta**2)

# METHODS FOR ACTIVE FLUID FIELDS

def defect_detector(
    data,
    filter_radius=1,
    area_threshold=1):
    ''' DETECT DEFECTS IN A 2D VECTOR FIELD (USES MIKE NORTON'S CODE)

    ARGUMENTS
    ---------
    data: numpy 3d.array [2, Nx, Ny] 2D vector field
    filter_radius: int [1] filter radius for defect detection
    area_threshold: float [1] threshold for defect area

    RETURNS
    -------
    centroid_p/m: numpy 2d.array [2, N] positions for N defects '''

    # rename input data
    nx, ny = data[0,], data[1,]

    # calculate components of Q tensor
    Qxx = nx**2-1/2
    Qxy = nx*ny

    Qxx_x, Qxx_y = np.gradient(Qxx)
    Qxy_x, Qxy_y = np.gradient(Qxy)

    denom = (1+4*Qxx+4*Qxy**2+4*Qxx**2)
    dphidx_num = 2*(-2*Qxy*Qxx_x+(1+2*Qxx)*Qxy_x)
    dphidy_num = 2*(-2*Qxy*Qxx_y+(1+2*Qxx)*Qxy_y)

    dphidx = np.divide(dphidx_num, denom, out=np.zeros_like(denom),
                             where=denom!=0)
    dphidy = np.divide(dphidy_num, denom, out=np.zeros_like(denom),
                             where=denom!=0)
    # dphidx = dphidx_num/denom
    # dphidy = dphidy_num/denom

    eps_mine = 1E0

    #remove ~0/0 artifacts
    dphidx[(np.abs(denom) < eps_mine) & (np.abs(dphidx_num) < eps_mine)] = 0
    dphidy[(np.abs(denom) < eps_mine) & (np.abs(dphidy_num) < eps_mine)] = 0

    # construct the filter radius
    r = filter_radius
    d = 2*r + 1
    x_grid, y_grid = np.indices((d, d))
    r_grid = ((r - x_grid)**2 + (r - y_grid)**2)**0.5

    ring_filter = (np.abs(r_grid - r) < 0.5).astype(int)

    x_grid -= r
    y_grid -= r

    x_grid[ring_filter == 0] = 0
    y_grid[ring_filter == 0] = 0

    r_grid[r,r] = 1
    filter_x = -y_grid/r_grid
    filter_y = x_grid/r_grid

    # construct map for periodic box (directional derivative along ring)
    map = (convolve2d(dphidy, filter_y, boundary='wrap', mode='same') +
           convolve2d(dphidx, filter_x, boundary='wrap', mode='same'))

    Nrows, Ncolumns = np.shape(map)

    map_m = np.zeros((Nrows, Ncolumns))
    map_p = np.zeros((Nrows, Ncolumns))

    map_m[map < -1] = 1
    map_p[map >  1] = 1

    regions_p = regionprops(label(map_p))
    regions_m = regionprops(label(map_m))

    # detect + defects
    centroid_p_xs = []
    centroid_p_ys = []
    for props in regions_p:
        y0, x0 = props.centroid
        area = props.area
        if area > area_threshold:
            centroid_p_xs.append(x0)
            centroid_p_ys.append(y0)

    # detect - defects
    centroid_m_xs = []
    centroid_m_ys = []
    for props in regions_m:
        y0, x0 = props.centroid
        area = props.area
        if area > area_threshold:
            centroid_m_xs.append(x0)
            centroid_m_ys.append(y0)

    centroid_p = np.array([centroid_p_xs, centroid_p_ys]).T
    centroid_m = np.array([centroid_m_xs, centroid_m_ys]).T

    return centroid_p, centroid_m

def fluid_measures(
    data: np.ndarray,
    delta: float,
    return_this: str='vorticity'):
    ''' COMPUTE FLUID QUANTITIES FROM VELOCITY FIELD

        ARGUMENTS
        ---------
        data: numpy nd.array
            Fluid velocity data. Shape must be either:
                - (2, Nx, Ny): 2D velocity field (u, v) for a single time frame.
                - (Nt, 2, Nx, Ny): 2D velocity field (u, v) for Nt time frames.
        delta: float
            Grid spacing (distance between adjacent grid points).
        return_this: str, optional
            Specifies which fluid quantity to compute and return. Options are:
                - 'vorticity': Local vorticity (default).
                - 'divergence': Velocity field divergence.
                - 'diagonal_shear': Diagonal shear component.
                - 'lateral_shear': Lateral shear component.
                - 'shear': Tuple of (lateral_shear, diagonal_shear).
                - 'okubo_weiss': Okubo-Weiss parameter.

        RETURN
        ------
        output: numpy nd.array or tuple of nd.array
            The requested fluid quantity, shape:
                - (Nx, Ny) for single frame input.
                - (Nt, Nx, Ny) for time series input.
            For 'shear', returns a tuple of arrays: (lateral_shear, diagonal_shear).

        RAISES
        ------
        ValueError
            If `data` is not a 3D or 4D array.

        NOTES
        -----
        - Assumes velocity data is staggered and interpolates to cell centers as needed.
        - Uses finite difference approximations for spatial derivatives.
        - Requires helper functions: `roll`, `ddh`, and `move`. '''

    # Validate input arguments
    dim = data.ndim
    if dim not in (3, 4):
        raise ValueError('data must be a 3d or 4d array')

    if dim == 3:
            
        # create a view of the input data
        u = data[0,]
        v = data[1,]

        # interpolate data to cell centers
        u_c = (roll(u, 1, 0) + u)/2
        v_c = (roll(v, 1, 1) + v)/2
        del u, v

        # calculate derivatives
        dudx = first_order_derivative(u, delta, 0, bulk='backward')
        dudy = first_order_derivative(u_c, delta, 1)
        dvdx = first_order_derivative(v_c, delta, 0)
        dvdy = first_order_derivative(v, delta, 1, bulk='backward')
        del u_c, v_c

    if dim == 4:

        # create a view of the input data
        u = data[:, 0,]
        v = data[:, 1,]

        # interpolate data
        u_c = interpolate_grid(u, 'stagger_x', 'centered')
        v_c = interpolate_grid(v, 'stagger_y', 'centered')

        # calculate derivatives
        dudx = first_order_derivative(u, delta, 1, bulk='backward')
        dudy = first_order_derivative(u_c, delta, 2)
        del u_c, u
        dvdx = first_order_derivative(v_c, delta, 1)
        dvdy = first_order_derivative(v, delta, 2, bulk='backward')
        del v_c, v

    # compute requested output from only the derivatives we prepared
    if return_this == 'divergence':
        np.add(dudx, dvdy, out=dudx)
        dudx *= 0.5
        return dudx
    if return_this == 'vorticity':
        np.subtract(dvdx, dudy, out=dvdx) 
        dvdx *= 0.5 
        return dvdx
    if return_this == 'shear':
        np.subtract(dudx, dvdy, out=dudx)
        np.add(dvdx, dudy, out=dvdx)
        dudx *= 0.5
        dvdx *= 0.5
        return dudx, dvdx
    if return_this == 'okubo_weiss':
        np.multiply(dudy, dvdx, out=dudy) 
        np.multiply(dudx, dvdy, out=dvdx) 

        dudy -= dvdx 

        np.add(dudx, dvdy, out=dudx)
        dudx *= 0.5
        np.square(dudx, out=dudx)

        np.add(dudx, dudy, out=dudy)

        return dudy

def nematic_measures(
    data: np.ndarray,
    only_director: bool=True,
    save_path: str=None):

    # Validate input arguments
    dim = data.ndim
    if dim not in (3, 4):
        raise ValueError('data must be a 3d or 4d array')

    # Access views without copying memory
    # Assumes nematic_order is [..., 2, Nx, Ny]
    qx = data[..., 0, :, :]
    qy = data[..., 1, :, :]

    # Vectorized calculation (handles 3D or 4D automatically)
    theta = 0.5 * np.arctan2(qy, qx)
    
    # Pre-allocate output array
    shape = list(data.shape)

    shape[-3] = 2
    if only_director is False:
        shape[-3] = 3
    scalar_director = np.empty(shape, dtype=data.dtype)

    scalar_director[..., -2, :, :] = np.cos(theta)
    scalar_director[..., -1, :, :] = np.sin(theta)
    if only_director is False:
        scalar_director[..., -3, :, :] = 2 * np.sqrt(qx**2 + qy**2)

    if save_path is True: 
        np.save(save_path, scalar_director)
    else: 
        return scalar_director

def nematic_elastic_energy(
    Q: np.ndarray,
    delta: float,
    K: float = 1.0):

    # 1. Setup Output Buffer immediately (Shape: [..., H, W])
    # We use the shape of the first component slice to determine output size
    energy_density = np.zeros(Q.shape[:-3] + Q.shape[-2:], dtype=Q.dtype)

    # 2. Iterative Accumulation (Reduces Peak Memory by ~4x)
    # We process gradients one by one and add their square directly to the sum
    # Mapping: axis -1 is X (cols), axis -2 is Y (rows)
    
    # --- Component Qxx ---
    qxx = Q[..., 0, :, :]
    
    # dQxx/dx (Axis -1)
    grad_tmp = first_order_derivative(qxx, delta, axis=-1) 
    energy_density += grad_tmp**2
    
    # dQxx/dy (Axis -2)
    grad_tmp = first_order_derivative(qxx, delta, axis=-2)
    energy_density += grad_tmp**2
    
    # --- Component Qxy ---
    qxy = Q[..., 1, :, :]
    
    # dQxy/dx (Axis -1)
    grad_tmp = first_order_derivative(qxy, delta, axis=-1)
    energy_density += grad_tmp**2
    
    # dQxy/dy (Axis -2)
    grad_tmp = first_order_derivative(qxy, delta, axis=-2)
    energy_density += grad_tmp**2
    
    # 3. Finalize Energy Density
    energy_density *= K
    
    # 4. Integrate for Total Energy
    # Sum over spatial dimensions (-2, -1) and scale by area element
    total_energy = np.sum(energy_density, axis=(-2, -1)) * (delta**2)

    return total_energy, energy_density
