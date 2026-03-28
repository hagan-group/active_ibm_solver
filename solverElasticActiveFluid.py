#!/usr/bin/env python3
import json
import h5py
import numpy as np
from basic_tools import discrete_delta
from basic_tools import laplacian as nabla
from basic_tools import interpolate_grid as move
from basic_tools import first_order_derivative as ddh
from basic_tools import bending_force as bending
from basic_tools import stretch_force as stretch
from basic_tools import normal_and_tangent as interface_geometry


''' COMMENT IT OUT ON HPCC '''
import progressbar as pb
import matplotlib.pyplot as plt

class solverFluidStructure(object):

    def __init__(self, par, memb_field):

        ''' FETCH SIMULATION PARAMETERS '''
        for key in par:
            setattr(self, key, par[key])

        ''' SET ALL PHYSICS PARAMETERS '''

        self.μ = self.DYN_VISCOSITY
        self.β = self.BLK_VISCOSITY
        self.Γ = self.FLUID_FRICTION
        self.Y = self.STRETCH_STIFFNESS
        self.B = self.BENDING_STIFFNESS
        self.α = - self.ACTIVITY

        self.λ = self.FLOW_ALIGNMENT
        self.S0 = self.S_INITIAL
        self.C = self.NEM_ENERGY/self.ROT_VISCOSITY
        self.κ = self.FRANK_CONST/self.ROT_VISCOSITY
        self.W = self.ANCHORING/self.ROT_VISCOSITY
        
        # decides if the nematic director is aligned parallel or perpendicular
        # near the elastic membrane. +1 for parallel, -1 for perpendicular
        self.δ = self.ALIGNMENT

        # decides if the active fluid is inside or outside
        # the direction is decided by the placement of the
        # membrane markers. Example: a circle is created moving in an
        # anticlockwise direction. ACTIVE_IN=1 makes the fluid inside the
        # circle active. ACTIVE_IN=0 makes the fluid outside active.
        self.Λ = 2*self.ACTIVE_IN - 1

        ''' SET ALL MEMBRANE FIELD PROPERTIES '''
        self.node = memb_field.node
        self.fore = memb_field.fore
        self.back = memb_field.back
        self.n_b = memb_field.n_b
        self.κ_0 = memb_field.κ
        self.G = memb_field.G
        self.X_0 = memb_field.X

        ''' NUMBER OF TIME AND SPACE POINTS '''
        self.dt = self.TIME_STEP
        self.dh = self.SPACE_STEP
        self.ds = self.dh/2

        self.n_t = int(self.MAX_TIME/self.dt)
        self.n_x = int(self.X_SIZE/self.dh)
        self.n_y = int(self.Y_SIZE/self.dh)
        self.box = np.array([self.X_SIZE, self.Y_SIZE])

        ''' FOURIER MODES AND FOURIER PROJECTOR '''

        qy, qx = np.meshgrid(np.fft.fftfreq(self.n_y, 1/self.n_y),
                             np.fft.fftfreq(self.n_x, 1/self.n_x))

        x_p = np.exp(2*np.pi*qx*1j/self.n_x)
        y_p = np.exp(2*np.pi*qy*1j/self.n_y)
        x_n = np.exp(-2*np.pi*qx*1j/self.n_x)
        y_n = np.exp(-2*np.pi*qy*1j/self.n_y)

        self.denom = ((x_n+y_n+x_p+y_p - 4)/(self.dh**2)).real
        self.denom[0,0] = 1

        # initiate projector
        self.P_xx = 1
        self.P_xy = 0
        self.P_yx = 0
        self.P_yy = 1

        # switch on Tr(E) terms
        self.bulk_on = 1

        # change projector for incompressible flows
        if self.β == 0:

            self.P_xx += np.divide(((1-x_n)*(1-x_p)/self.dh**2).real, self.denom)
            self.P_xy += np.divide(((1-x_p)*(1-y_n)/self.dh**2), self.denom)
            self.P_yx += np.divide(((1-y_p)*(1-x_n)/self.dh**2), self.denom)
            self.P_yy += np.divide(((1-y_n)*(1-y_p)/self.dh**2).real, self.denom)

            # switch off Tr(E) terms
            self.bulk_on -= 1

        del qx, qy, x_n, y_n, x_p, y_p

        ''' INITIALIZE ALL THE DYNAMICAL FIELDS '''

        self.U = np.empty([1, 2, self.n_x, self.n_y])
        self.Q = np.empty([1, 2, self.n_x, self.n_y])
        self.X = np.empty([1, 2, self.n_b])
        self.M = np.empty([1, self.n_x, self.n_y])

        if self.preload:

            self.U[0,] = np.asarray(h5py.File(f'{self.where}_{self.which}__velocity.h5', 'r')['dataset'][self.when,])
            self.X[0,] = np.asarray(h5py.File(f'{self.where}_{self.which}__membrane.h5', 'r')['dataset'][self.when,])
            self.M[0,] = np.asarray(h5py.File(f'{self.where}_{self.which}__mask.h5', 'r')['dataset'][self.when,])
            self.Q[0,] = np.asarray(h5py.File(f'{self.where}_{self.which}__nematic.h5', 'r')['dataset'][self.when,])

        else:

            ''' FLOW FIELD INITIALIZATION'''

            self.U[0,0,] = np.zeros([self.n_x, self.n_y])
            self.U[0,1,] = np.zeros([self.n_x, self.n_y])

            ''' MEMBRANE INITIALIZATION '''
            self.X[0,] = self.X_0
            
            ''' ACTIVE MASK INITIALIZATION '''

            _, _, normal_x, normal_y = self.grid_exchange_operator(
                membrane_position=self.X[0,], use='spread')
            self.M[0,], _ = self.level_set_function(normal_x, normal_y)
            del normal_x, normal_y

            ''' NEMATIC INITIALIZATION '''

            rng = np.random.default_rng()
            seed = rng.random(size=(self.n_x, self.n_y))
            if self.S0 < 0.5:
                θ = 2*np.pi*seed            # isotropic state
            else:
                θ = 0.2 * (2*seed - 1)      # aligned state
            
            S = self.S0 + 0.02 * seed

            nem_x, nem_y = np.cos(θ), np.sin(θ)
            self.Q[0,0,] = S * self.M * (nem_x**2 - 0.5)
            self.Q[0,1,] = S * self.M * (nem_x * nem_y)
            del nem_x, nem_y, θ

    def elastic_force(self, membrane_position):
        ''' CALCULATES THE ELASTIC FORCE IN THE LAGRANGIAN COORDINATES

        ARGUMENTS
        ---------
        membrane_position : numpy 2d.array [2, Nb] membrane position

        RETURNS
        -------
        total_force : numpy 2d.array [2, Nb] sum of all elastic forces '''

        # extract input data
        X = membrane_position

        stretch_force = self.Y * stretch(X, self.node, self.fore,
                                         self.back, self.ds, self.box)
        bending_force = self.B * bending(X, self.κ_0, self.node, self.fore,
                                         self.back, self.ds, self.box)
        pinning_force = - self.G * (X - self.X_0)

        total_force = stretch_force + bending_force + pinning_force

        return total_force

    def local_frame_measures(self, membrane_position: np.ndarray, axis: int): 
        ''' EXCHANGING DATA BETWEEEN EULERIAN AND LAGRANGIAN GRIDS

        ARGUMENTS
        ---------
        membrane_position: (np.ndarray) [2, Nb] membrane position
        axis : (int) x=0 (row) y=1 (col) axis for staggered grid

        RETURNS
        -------
        regularized_delt: (np.ndarray) [Nb,4,4] discrete delta function
        indices : (np.ndarray) [2,Nb] index of nearest grid point '''

        # extract input data
        X = membrane_position.copy()
        del membrane_position

        # shifted membrane position (for staggered grid)
        shift = [[0], [0]]
        shift[axis] = [self.dh/2]
        X -= shift

        # integer and normalized position
        indices = (X//self.dh).astype(int)
        r = X/self.dh - indices
        del X

        # calculate regularized delta functions
        regularized_delta = discrete_delta(r[0,], r[1,])
        del r

        return regularized_delta, indices

    def grid_exchange_operator(self, fluid_velocity: np.ndarray=None,
                               membrane_position: np.ndarray=None,
                               use: str='both'):
        ''' EXCHANGING DATA BETWEEEN EULERIAN AND LAGRANGIAN GRIDS

        ARGUMENTS
        ---------
        fluid_velocity : (np.ndarray) [Nx, Ny] fluid velocity in Eulerian frame
        membrane_position : (np.ndarray) [2, Nb] membrane in Lagrangian frame
        use : (str) decides the use case for the method

        RETURNS
        -------
        U : (np.ndarray) [2, Nb] fluid velocity in Lagrangian frame
        fx, fy : (np.ndarray) [Nx, Ny] elastic force in Eulerian frame
        nx, ny : (np.ndarray) [Nx, Ny] normal vector in Eulerian frame '''

        # rename input data
        X = membrane_position

        # neighbour stencil
        stencil = np.array([-1,0,1,2])

        # discrete delta and indices for staggered grids
        δ_stag_x, snap_stag_x = self.local_frame_measures(X, 0)
        δ_stag_y, snap_stag_y = self.local_frame_measures(X, 1)

        if use == 'spread':

            # calculate elastic force and normal vector
            F = self.elastic_force(X)
            ds, N, _ = interface_geometry(X, self.fore, self.box)

            # declare output arrays
            fx = np.zeros([self.n_x, self.n_y])
            fy = np.zeros([self.n_x, self.n_y])
            nx = np.zeros([self.n_x, self.n_y])
            ny = np.zeros([self.n_x, self.n_y])

            for k in range(self.n_b):

                ix_sx = np.mod((snap_stag_x[0, k] + stencil)[:,None], self.n_x)
                iy_sx = np.mod(snap_stag_x[1, k] + stencil, self.n_y)

                ix_sy = np.mod((snap_stag_y[0, k] + stencil)[:,None], self.n_x)
                iy_sy = np.mod(snap_stag_y[1, k] + stencil, self.n_y)

                fx[ix_sx, iy_sx] += ((self.ds*F[0,k]*δ_stag_x[k,])/(self.dh**2))
                fy[ix_sy, iy_sy] += ((self.ds*F[1,k]*δ_stag_y[k,])/(self.dh**2))

                nx[ix_sx, iy_sx] += (ds[k] * N[0, k] * δ_stag_x[k,])
                ny[ix_sy, iy_sy] += (ds[k] * N[1, k] * δ_stag_y[k,])

            del X, δ_stag_x, δ_stag_y, F, N

            return fx, fy, nx, ny

        if use == 'interpolate':

            # extract input data
            u = fluid_velocity

            # declare output arrays
            U = np.empty([2, self.n_b])

            for k in range(self.n_b):

                ix_sx = np.mod((snap_stag_x[0, k] + stencil)[:,None], self.n_x)
                iy_sx = np.mod(snap_stag_x[1, k] + stencil, self.n_y)

                ix_sy = np.mod((snap_stag_y[0, k] + stencil)[:,None], self.n_x)
                iy_sy = np.mod(snap_stag_y[1, k] + stencil, self.n_y)

                U[:, k] = (np.sum(δ_stag_x[k,] * u[0][ix_sx, iy_sx]),
                           np.sum(δ_stag_y[k,] * u[1][ix_sy, iy_sy]))

            del X, δ_stag_x, δ_stag_y, u
            
            return U

        if use == 'both':
            
            # extract input data
            u = fluid_velocity

            # calculate force and normal
            F = self.elastic_force(X)       
            ds, N, _ = interface_geometry(X, self.fore, self.box)

            # declare output arrays
            U = np.empty([2, self.n_b])
            fx = np.zeros([self.n_x, self.n_y])
            fy = np.zeros([self.n_x, self.n_y])
            nx = np.zeros([self.n_x, self.n_y])
            ny = np.zeros([self.n_x, self.n_y])

            for k in range(self.n_b):

                ix_sx = np.mod((snap_stag_x[0, k] + stencil)[:,None], self.n_x)
                iy_sx = np.mod(snap_stag_x[1, k] + stencil, self.n_y)

                ix_sy = np.mod((snap_stag_y[0, k] + stencil)[:,None], self.n_x)
                iy_sy = np.mod(snap_stag_y[1, k] + stencil, self.n_y)

                fx[ix_sx, iy_sx] += ((self.ds*F[0,k]*δ_stag_x[k,])/(self.dh**2))
                fy[ix_sy, iy_sy] += ((self.ds*F[1,k]*δ_stag_y[k,])/(self.dh**2))

                nx[ix_sx, iy_sx] += (ds[k] * N[0, k] * δ_stag_x[k,])
                ny[ix_sy, iy_sy] += (ds[k] * N[1, k] * δ_stag_y[k,])

                U[:, k] = (np.sum(δ_stag_x[k,] * u[0][ix_sx, iy_sx]),
                           np.sum(δ_stag_y[k,] * u[1][ix_sy, iy_sy]))

            del X, δ_stag_x, δ_stag_y, u, F, N
            
            return fx, fy, U, nx, ny

    def level_set_function(self, normal_x, normal_y):
        ''' OUTPUT A MASK OF ACTIVE REGION FROM THE MEMBRANE NORMAL VECTORS

        ARGUMENTS
        ---------
        normal_x : numpy 2d.array [Nx, Ny] x component as Eulerian field
        normal_y : numpy 2d.array [Nx, Ny] y component as Eulerian field

        RETURNS
        -------
        mask : numpy 2d.array [Nx, Ny] active region mask
        grad_mask: numpy 3d.array [2, Nx, Ny] ∇ of the active mask '''
        
        # declare output array
        grad_mask = np.empty([2, self.n_x, self.n_y])

        if self.Λ == 0:

            div_normal = np.fft.fft2(ddh(normal_x, self.dh, 0, bulk='backward') +
                                     ddh(normal_y, self.dh, 1, bulk='backward'))
            div_normal[0, 0] = 0
            
            mask = np.fft.ifft2(np.divide(div_normal, self.denom)).real
            mask -= mask.min()  
            mask /= mask.max()
    
            grad_mask[0,] = ddh(mask, self.dh, 0)
            grad_mask[1,] = ddh(mask, self.dh, 1)

            mask = np.ones((self.n_x, self.n_y))

        else:

            # self.Λ decides the active region
            div_normal = self.Λ * np.fft.fft2(ddh(normal_x, self.dh, 0,
                                                  bulk='backward') +
                                              ddh(normal_y, self.dh, 1,
                                                  bulk='backward'))
            div_normal[0, 0] = 0
            
            mask = np.fft.ifft2(np.divide(div_normal, self.denom)).real
            mask -= mask.min()
            mask /= mask.max()
    
            grad_mask[0,] = ddh(mask, self.dh, 0)
            grad_mask[1,] = ddh(mask, self.dh, 1)
        
        return mask, grad_mask

    def nematic_equation(self, U, Q, M, gradM, I=True):
        ''' DYNAMICS OF THE NEMATIC ORDER WITH FREE ENERGY AND FLOW ALIGNMENT

        ARGUMENTS
        ---------
        U : numpy 3d.array [2, Nx, Ny] fluid velocity at previous timestep
        Q : numpy 3d.array [2, Nx, Ny] nematic order at previous timestep
        M : numpy 2d.array [Nx, Ny] active region mask
        gradM : numpy 2d.array [Nx, Ny] active-passive interface

        RETURNS
        -------
        q_dot : numpy 3d.array [2, Nx, Ny] time derivative of nematic order '''

        # declare output array
        q_dot = np.empty([2, self.n_x, self.n_y])

        # components of strain rate tensor and rotation tensor
        E_xx = ddh(U[0,], self.dh, 0, bulk='backward')
        Tr_E = self.bulk_on * (ddh(U[0,], self.dh, 0, bulk='backward') +
                               ddh(U[1,], self.dh, 1, bulk='backward'))
        E_xy = 0.5 * (ddh(move(U[0,], 'stagger_x', 'centered'), self.dh, 1) +
                      ddh(move(U[1,], 'stagger_y', 'centered'), self.dh, 0))
        Ω_xy = 0.5 * (ddh(move(U[0,], 'stagger_x', 'centered'), self.dh, 1) -
                      ddh(move(U[1,], 'stagger_y', 'centered'), self.dh, 0))

        # value of local aligning nematic tensor
        ref_Q = np.empty([2, self.n_x, self.n_y])
        mag_gradM = gradM[0,]**2 + gradM[1,]**2
        ref_Q[0,] = gradM[0,]**2 - mag_gradM/2
        ref_Q[1,] = gradM[0,]*gradM[1,]
        del gradM

        # scalar order parameter
        S = 2 * np.sqrt(Q[0,]**2 + Q[1,]**2)

        # free energy
        if I:
            A = self.C
        else:
            A = self.C * (S*S*M - (2*M - 1))

        q_dot[0,] = ((self.λ * M * (E_xx - 0.5 * Tr_E))
                    + (2 * Ω_xy * Q[1,])
                    - (move(U[0,],'stagger_x','centered') * ddh(Q[0,],self.dh,0))
                    - (move(U[1,],'stagger_y','centered') * ddh(Q[0,],self.dh,1))
                    - Tr_E * Q[0,]
                    - A * Q[0,]
                    + (self.κ * M * nabla(Q[0,], self.dh))
                    - (self.W * np.sqrt(mag_gradM) * (Q[0,] + self.δ*ref_Q[0,])))

        q_dot[1,] = ((self.λ * M * E_xy)
                    - (2 * Ω_xy * Q[0,])
                    - (move(U[0,],'stagger_x','centered') * ddh(Q[1,],self.dh,0))
                    - (move(U[1,],'stagger_y','centered') * ddh(Q[1,],self.dh,1))
                    - Tr_E * Q[1,]
                    - A * Q[1,]
                    + (self.κ * M * nabla(Q[1,], self.dh))
                    - (self.W * np.sqrt(mag_gradM) * (Q[1,] + self.δ*ref_Q[1,])))
        del U, Q, M, E_xx, Tr_E, E_xy, Ω_xy, ref_Q, mag_gradM, S 

        return q_dot

    def stokes_equation(self, U, Q, Fx, Fy, M):
        ''' DYNAMICS OF THE FLUID VELOCITY

        ARGUMENTS
        ---------
        U : numpy 3d.array [2, Nx, Ny] fluid velocity at previous timestep
        Q : numpy 3d.array [2, Nx, Ny] nematic order at previous timestep
        F : numpy 2d.array [2, Nb] elastic forces at previous timestep
        M : numpy 2d.array [Nx, Ny] active region mask

        RETURNS
        -------
        v_dot : numpy 3d.array [2, Nx, Ny] time derivative of fluid velocity '''

        # declare output array
        v_dot = np.empty([2, self.n_x, self.n_y])

        Tr_E = (ddh(U[0,], self.dh, 0, bulk='backward') +
                ddh(U[1,], self.dh, 1, bulk='backward'))

        v_dot[0,] = (self.μ * nabla(U[0,], self.dh)
                     - self.Γ * U[0,]
                     + self.β * (ddh(Tr_E, self.dh, 0, bulk='forward'))
                     + self.α * (+ ddh(M*Q[0,], self.dh, 0, bulk='forward')
                                 + ddh(move(M*Q[1,], 'centered', 'stagger_x'),
                                       self.dh, 1))
                     + Fx)

        v_dot[1,] = (self.μ * nabla(U[1,], self.dh)
                     - self.Γ * U[1,]
                     + self.β * (ddh(Tr_E, self.dh, 1, bulk='forward'))
                     + self.α * (- ddh(M*Q[0,], self.dh, 1, bulk='forward')
                                 + ddh(move(M*Q[1,], 'centered', 'stagger_x'),
                                       self.dh, 0))
                     + Fy)
        del U, Q, Fx, Fy, M, Tr_E

        return v_dot

    def dynamics_solver(self, flag: int=0):

        ''' PREDICTOR HALF STEP '''

        # FORCE, NORMAL SPREADING, VELOCITY INTERPOLATION
        fx, fy, U, norm_x, norm_y = self.grid_exchange_operator(
            self.U[0,], self.X[0,], 'both')
        M, gradM = self.level_set_function(norm_x, norm_y)
        del norm_x, norm_y

        # VELOCITY CALCULATION
        u = np.empty([2, self.n_x, self.n_y])
        v_dot = np.fft.fft2(self.stokes_equation(self.U[0,],self.Q[0,],fx,fy,M))
        del fx, fy

        u[0,] = self.U[0,0,] + self.dt * np.fft.ifft2(
            v_dot[0]*self.P_xx + v_dot[1]*self.P_xy).real
        u[1,] = self.U[0,1,] + self.dt * np.fft.ifft2(
            v_dot[0]*self.P_yx + v_dot[1]*self.P_yy).real

        # NEMATIC ORDER CALCULATION
        q_dot = self.nematic_equation(self.U[0,], self.Q[0,], M, gradM, I=self.ISO)
        q = self.Q[0,] + self.dt * q_dot
        del M, gradM

        # MEMBRANE POSITION CALCULATION
        x = self.X[0,] + self.dt * U

        ''' CORRECTOR FULL STEP '''

        # FORCE, NORMAL SPREADING
        fx, fy, norm_x, norm_y = self.grid_exchange_operator(
            membrane_position=x, use='spread')
        self.M[0,], gradM = self.level_set_function(norm_x, norm_y)
        del norm_x, norm_y

        # VELOCITY CALCULATION
        v_dot += np.fft.fft2(self.stokes_equation(u, q, fx, fy, self.M[0,]))
        self.U[0,0,] += 0.5 * self.dt * np.fft.ifft2(
            v_dot[0]*self.P_xx + v_dot[1]*self.P_xy).real
        self.U[0,1,] += 0.5 * self.dt * np.fft.ifft2(
            v_dot[0]*self.P_yx + v_dot[1]*self.P_yy).real
        del fx, fy, v_dot

        # NEMATIC ORDER CALCULATION
        q_dot += self.nematic_equation(u, q, self.M[0,], gradM, I=self.ISO)
        self.Q[0,] += 0.5 * self.dt * q_dot
        del u, q, q_dot, gradM

        # MEMBRANE POSITION CALCULATION
        U += self.grid_exchange_operator(self.U[0,], x, 'interpolate')
        self.X[0,] += 0.5 * self.dt * U
        del x, U

    def solve(self):

        for time in pb.progressbar(range(self.n_t)):

            # initiate the files
            if time % self.BATCH == 0 and self.save:

                label = str(int((time/self.BATCH) + 1))

                velocity = h5py.File(
                    self.tag + '_{N}__velocity.h5'.format(N=label), 'a')
                velocity.create_dataset('dataset', data=self.U, compression="gzip",
                    chunks=True, maxshape=(None, 2, self.n_x, self.n_y))

                nematic = h5py.File(
                    self.tag + '_{N}__nematic.h5'.format(N=label), 'a')
                nematic.create_dataset('dataset', data=self.Q, compression="gzip",
                    chunks=True, maxshape=(None, 2, self.n_x, self.n_y))
                
                position = h5py.File(
                    self.tag + '_{N}__membrane.h5'.format(N=label), 'a')
                position.create_dataset('dataset', data=self.X, compression="gzip",
                    chunks=True, maxshape=(None, 2, self.n_b))

                mask = h5py.File(
                    self.tag + '_{N}__mask.h5'.format(N=label), 'a')
                mask.create_dataset('dataset', data=self.M, compression="gzip",
                    chunks=True, maxshape=(None, self.n_x, self.n_y))

            # save data at snapshots decided by self.SKIP
            if time % self.SKIP == 0 and time % self.BATCH != 0 and self.save:

                velocity['dataset'].resize((velocity['dataset'].shape[0] +
                                            self.U.shape[0]), axis=0)
                velocity['dataset'][-self.U.shape[0]:] = np.stack(
                    (self.U[0, 0, ], self.U[0, 1, ]))

                nematic['dataset'].resize((nematic['dataset'].shape[0] +
                                           self.Q.shape[0]), axis=0)
                nematic['dataset'][-self.Q.shape[0]:] = np.stack(
                    (self.Q[0, 0, ], self.Q[0, 1, ]))

                position['dataset'].resize((position['dataset'].shape[0] +
                                            self.X.shape[0]), axis=0)
                position['dataset'][-self.X.shape[0]:] = self.X

                mask['dataset'].resize((mask['dataset'].shape[0] +
                                        self.M.shape[0]), axis=0)
                mask['dataset'][-self.M.shape[0]:] = self.M

            # close the files
            if (time+1) % self.BATCH == 0 and self.save:

                velocity.close()
                nematic.close()
                position.close()
                mask.close()

            # solve the dynamics of the continuum fields
            self.dynamics_solver()

if __name__ == "__main__":

    import os
    import sys
    import datetime as dt
    from solver_fields import membraneField

    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # LOAD THE PARAMETER FILE
    assert os.path.isfile('solver_parameters.json'), 'cant find parameter file'
    with open('solver_parameters.json') as jsonFile:
        par = json.load(jsonFile)

    par['tag'] = (par['folder'] + dt.datetime.now().strftime('/%y%m%d_%H%M%S'))

    if par['save']:
        parameters_file = par['tag'] + '__parameters.json'
        with open(parameters_file, 'w') as jsonFile:
            json.dump(par, jsonFile, indent=4)

    membrane = membraneField(par)
    case = solverFluidStructure(par, membrane)
    case.solve()