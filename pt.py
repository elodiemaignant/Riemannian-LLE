import os 
os.environ['GEOMSTATS_BACKEND'] = 'pytorch'

import geomstats.backend as gs
from geomstats.geometry.pre_shape import KendallShapeMetric, PreShapeBundle, PreShapeMetric, PreShapeSpace
from geomstats.visualization import KendallDisk

def sylvester(S, C):
    #solve the sylvester equation SX + XS = C 
    #where S is a SPD matrix
    [L, P] = gs.linalg.eigh(S)
    Y = (gs.transpose(P) @ C @ P) / (L[:, None] + L[None, :] + gs.eye(len(L)))
    return P @ Y @ gs.transpose(P)
    
def frobenius_product(x, y):
    return gs.sum(x * y, (-1,-2))

def ode(gamma, dgamma):
    #return F
    def F(s, vs):
        is_vectorized = (gs.ndim(gs.array(vs)) == 3)
        axes = (0, 2, 1) if is_vectorized else (1, 0)
        
        S = gs.transpose(gamma(s)) @ gamma(s)
        C = gs.transpose(dgamma(s)) @ vs - gs.transpose(vs, axes) @ dgamma(s)
        A = sylvester(S, C)
        fprod = frobenius_product(dgamma(s), vs)
        return - gs.einsum('...,ij->...ij', fprod, gamma(s)) - gamma(s) @ A
    return F

def runge_kutta_4(F, v, n):
    #return v(s) and v(s_n) solving edo (E) with n steps
    s = gs.linspace(0., 1., n)
    transported_v = v
    pt = [transported_v]
    
    for k in range(n - 1):
        h = s[k+1] - s[k]
        k1 = F(s[k], transported_v)
        k2 = F(s[k] + h / 2., transported_v + h / 2. * k1)
        k3 = F(s[k] + h / 2., transported_v + h / 2. * k2)
        k4 = F(s[k] + h, transported_v + h * k3)
        transported_v = transported_v + (h / 6.) * (k1 + 2. * k2 + 2. * k3 + k4)
        pt.append(transported_v)
    
    return pt, transported_v
    
def geodesic(w, base_point):
    #return the geodesic exp(sw) and its derivative
    norm_w = gs.linalg.norm(w)
    gamma = lambda s: gs.cos(s * norm_w) * base_point + gs.sin(s * norm_w) * w / norm_w
    dgamma = lambda s: - norm_w * gs.sin(s * norm_w) * base_point + gs.cos(s * norm_w) * w
    return gamma, dgamma
    
def parallel_transport(w, v, base_point, n):
    #return the parallel transport of horizontal 
    #vector(s) v along the horizontal geodesic exp(sw)
    gamma, dgamma = geodesic(w, base_point)
    F = ode(gamma, dgamma)
    return runge_kutta_4(F, v, n)
