import numpy as np
from basic_tools import roll

class membraneField(object):

    def __init__(self, parameters):

        # fetch parameters
        for key in parameters: setattr(self, key, parameters[key])

        lx = self.X_SIZE
        ly = self.Y_SIZE
        ds = self.SPACE_STEP/2
        G = self.PINNING_STIFFNESS

        self.count = self.MEMBRANE['count']
        shape = self.MEMBRANE['shape']
        radius = self.MEMBRANE['radius']
        width = self.MEMBRANE['width']
        stiff = self.MEMBRANE['stiff']
        slack = self.MEMBRANE['slack']
        ν = self.MEMBRANE['ratio']
        q = self.MEMBRANE['mode']
        off_center = self.MEMBRANE['off_center']/100

        a_q = 0                 # mode strength for deflated membrane
        k_q = (2*np.pi*q)/lx    # wavenumber for deflated membrane
        κ = 1/radius            # intrinsic curvature

        if self.count == 2 and shape == 'line' and slack == 'one':
            raise ValueError("slack 'one' not defined for a channel geometry.")

        if slack != 'one' and slack != 'two': ν = 0

        if self.count == 1 and stiff != 'all':
            raise ValueError("for self.count=1 set stiff='all' only.")

        # NUMBER OF MEMBRANE POINTS
    
        if self.count == 1:
        
            if shape == 'line': l = lx * (1 + ν)
            if shape == 'circle': l = (2 * np.pi * radius) * (1 + ν)
        
        if self.count == 2:

            if shape == 'line': l = lx + lx * (1 + ν)
            if shape == 'circle':
                l = 2 * np.pi * (radius + (radius + width))
                if slack == 'one':
                    l += 2 * np.pi * ν * (radius + width)
                else:
                    l += 2 * np.pi * ν * (radius)
  
        self.n_b = int(l/ds)

        # CONNECTIVITY INDEX LIST

        self.node = np.linspace(0, self.n_b, self.n_b, endpoint=False, dtype=int)
        
        if self.count == 1:

            self.fore = roll(self.node, -1, 0)
            self.back = roll(self.node, +1, 0)

        if self.count == 2:

            if shape == 'line':

                n_i = int((lx * (1 + ν))/ds)
                n_o = int(self.n_b - n_i)

            if shape == 'circle':
                
                if slack == 'one':
                    n_i = int((2 * np.pi * radius)/ds)
                else:
                    n_i = int((2 * np.pi * radius * (1 + ν))/ds)
                n_o = int(self.n_b - n_i)

            one = np.linspace(0, n_o, n_o, endpoint=False, dtype=int)
            two = np.linspace(0, n_i, n_i, endpoint=False, dtype=int)
            self.fore = np.concatenate(
                (roll(one,-1,0), roll(two,-1,0) + n_o), axis=0)
            self.back = np.concatenate(
                (roll(one,+1,0), roll(two,+1,0) + n_o), axis=0)
            
            self.n_in = n_i
            self.n_out = n_o

        # ELASTIC STIFFNESS ARRAY

        self.G = np.zeros(self.n_b)

        if stiff == 'one':
            self.G[:n_o] = G
        if stiff == 'two':
            self.G[n_o:] = G
        if stiff == 'all':
            self.G[:] = G

        # INTRINSIC CURVATURE

        self.κ = np.zeros(self.n_b)

        if shape == 'circle': 
            self.κ[:] = κ
            if self.count == 2:
                self.κ[:n_o] = 1/(radius+width)

        # MEMBRANE INITIALIZATION

        self.X = np.empty((2, self.n_b))

        if slack == 'one' or slack == 'two':
            
            if shape == 'line': a_q = np.sqrt(ν/k_q**2)
            if shape == 'circle': a_q = np.sqrt(ν/(q**2 -1))

        if self.count == 1:

            if shape == 'line':

                s = (self.node + 0.5) * ds
                
                self.X[0,] = s
                self.X[1,] = ly/2 + 2 * a_q * np.cos(k_q * s)

            if shape == 'circle':

                φ = np.linspace(0, 2*np.pi, self.n_b, endpoint=False)
                R = radius * (1 - a_q**2 + 2 * a_q * np.cos(q * φ))

                self.X[0,] = lx/2 + R * np.cos(φ)
                self.X[1,] = ly/2 + R * np.sin(φ)

        if self.count == 2:

            if shape == 'line':

                s_one = (one + 0.5) * ds
                s_two = (two + 0.5) * (ds / (1 + ν))

                self.X[0,:n_o] = s_one
                self.X[1,:n_o] = (ly - width)/2

                self.X[0,n_o:] = lx - s_two
                self.X[1,n_o:] = (ly + width)/2 + 2 * a_q * np.cos(k_q * (lx - s_two))

            if shape == 'circle':

                φ_i = - np.linspace(0, 2*np.pi, n_i, endpoint=False)
                φ_o = + np.linspace(0, 2*np.pi, n_o, endpoint=False)
                
                R_i = radius
                R_o = radius
                if slack == 'one':
                    R_o = radius * (1 - a_q**2 + 2 * a_q * np.cos(q * φ_o))
                if slack == 'two':
                    R_i = radius * (1 - a_q**2 + 2 * a_q * np.cos(q * φ_i))

                self.X[0,:n_o] = (lx/2 + (R_o + width) * np.cos(φ_o))
                self.X[1,:n_o] = (ly/2 + (R_o + width) * np.sin(φ_o))
                
                self.X[0,n_o:] = (lx/2 + off_center * width + R_i * np.cos(φ_i))
                self.X[1,n_o:] = (ly/2 + R_i * np.sin(φ_i))