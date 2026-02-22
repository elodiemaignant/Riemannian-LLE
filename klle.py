import os 
os.environ['GEOMSTATS_BACKEND'] = 'pytorch'

import geomstats.backend as gs
from geomstats.geometry.pre_shape import KendallShapeMetric, PreShapeBundle, PreShapeMetric, PreShapeSpace
from geomstats.visualization import KendallDisk

from geomstats.algebra_utils import flip_determinant
from geomstats.learning.frechet_mean import FrechetMean
import torch
from torch import diag
from scipy.linalg import expm, logm
from scipy.optimize import minimize, NonlinearConstraint
from sklearn.neighbors import NearestNeighbors

from pt import parallel_transport

class KendallHorizontalBundle:
    """Class for the horizontal Bundle of Kendall shape spaces.
    
    It gathers some methods for linear algebra computations in 
    the horizontal bundle.
    
    Parameters
    ----------
    k_landmarks : int
        Number of landmarks
    m_ambient : int
        Number of coordinates of each landmark.
        
    Attributes
    ----------
    k_landmarks : int
        Number of landmarks
    m_ambient : int
        Number of coordinates of each landmark.
    manifold : manifold
        Manifold from which the data are sampled (here a pre-shape space).
    hor_metric : metric
        Metric of the horizontal bundle (here the pre-shape metric).
    align : fun
        Align a point (or points) onto a base point in rotation.
    vertical_projection : fun
        Projects a tangent vector (or tangent vectors) onto its (their) vertical component at a base point.
    """
    
    def __init__(self, k_landmarks, m_ambient):
        self.k_landmarks=k_landmarks
        self.m_ambient=m_ambient
        self.manifold=PreShapeSpace(k_landmarks, m_ambient)
        self.hor_metric=PreShapeMetric(self.manifold)
        self.ver_dim=m_ambient * (m_ambient - 1) // 2
        self.hor_dim=self.manifold.dim - self.ver_dim
        self.align=PreShapeBundle(self.manifold).align
        self.vertical_projection=PreShapeBundle(self.manifold).vertical_projection
        
    def random_horizontal_vector(self, base_point, n_samples=1):
        """Sample in the horizontal subspace.
        
        Parameters
        ----------
        base_point: array-like, shape=[k_landmarks, m_ambient]
            Point of the pre-shape space.
        n_samples: int
            Number of samples.
        
        Returns
        -------
        hor_vec : array-like, shape=[..., k_landmarks, m_ambient]
            Horizontal vector(s) at the base_point.
        """
        vec = self.manifold.random_point(n_samples)
        tangent_vec = self.manifold.to_tangent(vec, base_point)
        hor_vec = tangent_vec - self.vertical_projection(tangent_vec, base_point)
        return hor_vec
    
    def horizontal_basis(self, base_point):
        """Construct an orthonormal basis of the horizontal subspace.
        
        Parameters
        ----------
        base_point : array-like, shape=[k_landmarks, m_ambient]
            Point of the pre-shape space.
        
        Returns
        -------
        hor_basis : array-like, shape=[hor_dim, k_landmarks, m_ambient]
            Orthonormal basis of the horizontal subspace.
        """
        hor_vec = self.random_horizontal_vector(base_point)
        unit_vec = hor_vec / gs.linalg.norm(hor_vec)
        hor_basis = [unit_vec]
        
        while len(hor_basis)<self.hor_dim:
            hor_vec = self.random_horizontal_vector(base_point)
        
            for e in hor_basis:
                prod = self.hor_metric.inner_product(hor_vec, e)
                hor_vec = hor_vec -  prod * e
                unit_vec = hor_vec / self.hor_metric.norm(hor_vec)
                if prod < 0.: unit_vec = - unit_vec
        
            hor_basis.append(unit_vec)
        
        return(gs.array(hor_basis))
        
    def horizontal_vector(self, coords, hor_basis):
        """Contruct a horizontal vector given its coordinates in an 
        orthonormal basis of the horizontal subspace.
        
        Parameters
        ----------
        coords : array-like, shape=[hor_dim]
            Coordinates in the basis.
        hor_basis : array-like, shape=[hor_dim, k_landmarks, m_ambient]
            Orthonormal basis of the horizontal subspace.
         
        Returns
        -------
        hor_vec : array-like, shape=[k_landmarks, m_ambient]
            Corresponding horizontal vector.
        """
        hor_vec = sum([c * e for (c, e) in zip (coords, hor_basis)])
        return(hor_vec)
    
    def horizontal_coordinates(self, hor_vec, hor_basis):
        """Return the coordinates of a horizontal vector in a given 
        orthonormal basis of the horizontal subspace.
        
        Parameters
        ----------
        hor_vec : array-like, shape=[k_landmarks, m_ambient]
            Horizontal vector.
        hor_basis : array-like, shape=[d_ambient, k_landmarks, m_ambient]
            Orthonormal basis of the horizontal subspace.
         
        Returns
        -------
        coords : array-like, shape=[hor_dim]
            Coordinates of the vector in the basis.
        """
        coords = gs.array([self.hor_metric.inner_product(hor_vec, e) for e in hor_basis])
        return(coords)
    
    def align_matrix(self, point, base_point):
        """Compute the rotation aligning point on base_point.
        
        Parameters
        ----------
        point : array-like, shape=[k_landmarks, m_ambient]
            Point of the pre-shape space.
        base_point : array-like, shape=[k_landmarks, m_ambient]
            Point of the pre-shape space.
         
        Returns
        -------
        align_mat : array-like, shape=[m_ambient, m_ambient]
            Alignment rotation matrix.
        """
        mat = gs.transpose(point) @ base_point
        left, singular_values, right = gs.linalg.svd(mat)
        det = gs.linalg.det(left @ right)
        
        if det < 0.: 
            right_flipped = gs.copy(right)
            right_flipped[-1, :] = - right[-1, :]
            align_mat = left @ right_flipped
        else: align_mat = left @ right
            
        return align_mat
        
class KendallLocallyLinearEmbedding:
    """Class for Locally Linear Embedding in Kendall shape spaces.
    
    Extend Locally Linear Embedding (LLE) to manifold-valued data. 
    In this first version, the manifold is some Kendall shape space.
    
    Parameters
    ----------
    k_landmarks : int
        Number of landmarks
    m_ambient : int
        Number of coordinates of each landmark.
        
    Attributes
    ----------
    k_landmarks : int
        Number of landmarks
    m_ambient : int
        Number of coordinates of each landmark.
    manifold : manifold
        Manifold from which the data are sampled (here a pre-shape space).
    metric : metric
        Intrinsic metric of the manifold (here the shape metric).
    hor_bundle : class
        Linear algebra methods for the horizontal bundle.
    hor_metric : metric
        Metric of the horizontal bundle (here the pre-shape metric).
    hor_parallel_transport : method
        Method for computing the parallel transport on the horizontal bundle.
    ver_dim : int
        Dimension of the vertical bundle.
    hor_dim : int
        Dimension of the horizontal bundle.
    
    References
    ----------
    .. [RS2000] Roweis, S. T., & Saul, L. K. (2000). "Nonlinear 
    dimensionality reduction by locally linear embedding". 
    science, 290(5500), 2323-2326.
    https://doi.org/10.1126/science.290.5500.2323
    """
    
    def __init__(self, k_landmarks, m_ambient):
        self.k_landmarks=k_landmarks
        self.m_ambient=m_ambient
        self.manifold=PreShapeSpace(k_landmarks, m_ambient)
        self.manifold.equip_with_group_action("rotations")
        self.manifold.equip_with_quotient()
        self.metric=self.manifold.quotient.metric
        self.hor_bundle=KendallHorizontalBundle(k_landmarks, m_ambient)
        self.hor_metric=PreShapeMetric(self.manifold)
        self.hor_parallel_transport=parallel_transport
        self.ver_dim=m_ambient * (m_ambient - 1) // 2
        self.hor_dim=self.manifold.dim - self.ver_dim
                
    def nearest_neighbours(self, points, k_neighbours):
        """Compute the k nearest neighbours of each data point.
        
        Parameters
        ----------
        points : array-like, shape=[n_points, k_landmarks, m_ambient]
            Pairwise distance matrix of the data.
        k_neighbours : int
            Number of neighbours
        
        Returns
        -------
        neighbours : array-like, shape=[n_points, k_neighbours]
            Indices of the k-nearest neighbours of each datapoint.
        """
        dist = self.metric.dist_pairwise(points)
        knn = NearestNeighbors(n_neighbors=k_neighbours, algorithm='auto', 
                               metric='precomputed').fit(dist)
        return(knn.kneighbors(return_distance=False))
    
    
    def barycentric_projection(self, point, ref_points, tol=1E-6, cons_tol=1E-6, n_steps=10, max_it=50, w0='equal'):  
        """Compute the projection of a point onto the barycentric 
        subspace of some reference points.
        
        Parameters
        ----------
        point : array-like, shape=[k_landmarks, m_ambient]
            Point to project.
        ref_points : array-like, shape=[k_points, k_landmarks, m_ambient]
            Reference points.
        tol : float.
            Tolerance for termination.
        cons_tol : float.
            Constraint tolerance for termination.
        max_it : int
            Maximum number of iterations before termination.
        n_steps : int
            Number of steps in computing the parallel transport.
        
        Returns
        -------
        a : array-like, shape=[k_points, m_ambient]
            Skew matrices encoding the rotations aligning the pushforward preshape 
            Exp(Par(uj)) on the reference point xj.
        u : array-like, shape=[k_points, k_landmarks, m_ambient]
            Horizontal vectors at the point such that their parallel 
            transport to the projection Par(uj) shoot towards the reference points.
        v : array-like, shape=[k_points, m_ambient]
            Horizontal vector at the point shooting towards the projection.
        w : array-like, shape=[k_points, m_ambient]
            Barycentric weights of the projection.
        """
        hor_basis = self.hor_bundle.horizontal_basis(point)
        
        def skew_to_array(skew_mat):
            return(gs.triu_to_vec(skew_mat, k=1))
    
        def convert_auvw_into_z(a, u, v, w):
            a_flat = gs.flatten(gs.array([skew_to_array(aj) for aj in a]))
            u_flat = gs.flatten(gs.array([self.hor_bundle.horizontal_coordinates(uj, hor_basis) for uj in u]))
            v_flat = self.hor_bundle.horizontal_coordinates(v, hor_basis)
            return gs.hstack((a_flat, u_flat, v_flat, w))
        
        def array_to_skew(array):
            n = int(1 + gs.sqrt(1 + 8 * len(array)) / 2)
            skew_mat = gs.zeros((n, n))
            k = 0
            for i in range(n):
                for j in range(i+1, n):
                    skew_mat[i, j] = array[k]
                    skew_mat[j, i] = - array[k]
                    k += 1
            return(skew_mat)
    
        def convert_z_into_auvw(z):
            arrays = gs.reshape(z[:len(ref_points) * self.ver_dim], (len(ref_points), self.ver_dim))
            a = gs.array([array_to_skew(array) for array in arrays])
        
            coords = gs.reshape(z[len(ref_points) * self.ver_dim: len(ref_points) * self.manifold.dim], 
                                (len(ref_points),  self.hor_dim))
            u = gs.array([self.hor_bundle.horizontal_vector(c, hor_basis) for c in coords])
        
            coords = z[len(ref_points) * self.manifold.dim: -len(ref_points)]
            v = self.hor_bundle.horizontal_vector(coords, hor_basis)
        
            w = z[-len(ref_points):]
            return(a, u, v, w)
    
        def reconstruction_error(v):
            return(self.hor_metric.squared_norm(v))
    
        def weights_constraint(w):
            return((gs.sum(w) - 1.) ** 2)
    
        def barycentric_constraint(u, w):
            return(self.metric.squared_norm(gs.einsum('j,jkl->kl', w, u), point))
      
        def shooting_constraint(a, u, v):
            if self.hor_metric.squared_norm(v, point) == 0.: 
                transported_u = u
                projected = point
            
            else:
                pt, transported_u = self.hor_parallel_transport(v, u, point, n_steps)
                projected = self.hor_metric.exp(v, point)
                
            shoots = self.hor_metric.exp(transported_u, projected)
            return gs.sum(self.manifold.embedding_space.metric.squared_dist(ref_points, shoots @ gs.linalg.expm(a)))
    
        def fun(z):
            z_torch = torch.from_numpy(z)
            a, u, v, w = convert_z_into_auvw(z_torch)
            f = reconstruction_error(v)
            return(f.numpy()) 
    
        def jac(z):
            z_torch = torch.from_numpy(z)
            z_rg = z_torch.clone().requires_grad_(True)
            a_rg, u_rg, v_rg, w_rg = convert_z_into_auvw(z_rg)
            df = torch.autograd.grad(outputs=reconstruction_error(v_rg), inputs=z_rg, 
                                     create_graph=True)[0].detach().flatten().numpy().astype('float64')
            return(df)
    
        def cons_fun(z):
            z_torch = torch.from_numpy(z)
            a, u, v, w = convert_z_into_auvw(z_torch)
            h = weights_constraint(w) + barycentric_constraint(u, w) + shooting_constraint(a, u, v)
            return(h.numpy()) 
    
        def cons_jac(z):
            z_torch = torch.from_numpy(z)
            z_rg = z_torch.clone().requires_grad_(True)
            a_rg, u_rg, v_rg, w_rg = convert_z_into_auvw(z_rg)
            dh = torch.autograd.grad(outputs=weights_constraint(w_rg)+barycentric_constraint(u_rg, w_rg)
                                     +shooting_constraint(a_rg, u_rg, v_rg), inputs=z_rg, 
                                     create_graph=True)[0].detach().flatten().numpy().astype('float64')
            return(dh)
        
        #Define the constraint
        constraint = NonlinearConstraint(fun=cons_fun, lb=-cons_tol, ub=cons_tol, jac=cons_jac)
        #, keep_feasible=True)
        
        #Initialise w: equal weights
        if w0 == 'equal': 
            w0 = gs.ones(len(ref_points)) / len(ref_points)
        
        #Initialise w: random weights
        if w0 == 'random': 
            w0 = gs.random.rand(len(ref_points))
            w0 = w0 / gs.sum(w0)
        
        #Initialise w: LLE weights
        if w0 == 'lle': 
            aligned = self.hor_bundle.align(ref_points, point)
            centred = gs.array([gs.flatten(x) for x in aligned]) - gs.flatten(point)
            covariance_mat = centred @ gs.transpose(centred)
            #covariance_mat = covariance_mat + alpha * gs.trace(covariance_mat) * gs.eye(k_neighbours)
            w0 = gs.linalg.inv(covariance_mat) @ gs.ones(len(ref_points))
            w0 = w0 / gs.sum(w0)
            
        #Initialise w: critical weights
        if w0 == 'critical': 
            aligned = self.hor_bundle.align(ref_points, point)
            centred = gs.array([gs.flatten(self.metric.log(x, point)) for x in aligned])
            covariance_mat = centred @ gs.transpose(centred)
            [eigen_val, eigen_vec] = gs.linalg.eigh(covariance_mat)
            #print(eigen_val)
            w0 = eigen_vec[0] / gs.sum(eigen_vec[0])
        
        #Initialise the projection: Frechet mean
        FM = FrechetMean(self.manifold)
        hat_x0 = FM.fit(ref_points, weights=w0).estimate_
        hat_u0 = self.hor_metric.log(ref_points, hat_x0)
        
        #Initialise a: alignment rotations
        a0 = gs.array([logm(self.hor_bundle.align_matrix(exp, p)).real for (p, exp) in 
                       zip(ref_points, self.hor_metric.exp(hat_u0, hat_x0))])
        
        #Initialise u: shoot from estimate at neighbours
        pt0, u0 = self.hor_parallel_transport(self.hor_metric.log(point, hat_x0), hat_u0, hat_x0, n_steps)
        
        #Initialise v: shoot from point at estimate
        v0 = self.hor_metric.log(hat_x0, point)
        pt0, transported_u0 = self.hor_parallel_transport(v0, u0, point, n_steps)
        
        #Initialise the optimisation variable z
        z0 = convert_auvw_into_z(a0, u0, v0, w0)
        
        #print("initial reconstruction error =", reconstruction_error(v0))
        #print("constraint =", cons_fun(z0.numpy()))
        #print("weights constraint =",weights_constraint(w0))
        #print("barycentric constraint =", barycentric_constraint(u0, w0))
        #print("shooting constraint =", shooting_constraint(a0, u0, v0))
        
        #Optimisation
        result = minimize(fun=fun, x0=z0, method='SLSQP', jac=jac, 
                          constraints=constraint, tol=tol, options={'maxiter' : max_it})
    
        #print("final reconstruction error =", result.fun)
        #print("constraint =", cons_fun(result.x))
        
        return(convert_z_into_auvw(torch.from_numpy(result.x)))
    
    def embedding_coordinates(self, weights, d_embedding):
        """Compute embedding coordinates in R^d with respect to 
        the optimal weights found at the reconstruction step.
        
        Parameters
        ----------
        weights : array-like, shape=[n_points, n_points]
            Sparse matrix of reconstruction weights.
        d_embedding : int
            Embedding dimension.
        
        Returns
        -------
        embedding : array-like, shape=[n_points, d_embedding]
            Embedding coordinates.
        """
        n_points=len(weights)
        identity = gs.eye(n_points)
        sparse_mat = gs.transpose(identity - weights) @ (identity - weights)

        #Find bottom d+1 eigenvectors of s (corresponding to the d+1 smallest 
        #eigenvalues) 
        [eigen_val, eigen_vec] = gs.linalg.eigh(sparse_mat)

        #Set the qth column of y to be the q+1 smallest eigenvector (discard the 
        #bottom (?) eigenvector [1,1,1,1...] with eigenvalue zero)
        embedding = gs.zeros((n_points, d_embedding))
        unit_vec = gs.ones(n_points) / gs.sqrt(n_points)
        
        for q in range(d_embedding):
            embedding[:, q] = eigen_vec[:, q+1] - gs.dot(eigen_vec[:, q+1], unit_vec) * unit_vec
            embedding[:, q] = embedding[:, q] / gs.linalg.norm(embedding[:, q])
            
        return(embedding)
    
    def fit(self, points, k_neighbours, d_embedding):
        """Perform Locally Barycentric Embedding of the data and returns the 
        position of the points in the d-dimensional embedding space. It is
        decomposed in 3 steps. Step 1: Find neighbours in ambient space. Step 2: 
        Optimize for reconstruction weights. Step 3: Compute embedding coordinates 
        w.r.t the weights.
        
        Parameters
        ----------
        points : array-like, shape=[n_points, k_landmarks, m_ambient]
            Datapoints in Kendall preshape space.
        k_neighbours : int
            Number of neighbours chosen per datapoint.
        d_embedding : int
            Dimension of embedding coordinates.
        
        Returns
        -------
        embedding : array-like, shape=[n_points, d_embedding]
            Embedding coordinates.
        
        References
        ----------
        """
        #Step 1: k nearest neighours
        neighbours = self.nearest_neighbours(points, k_neighbours)
        
        #Step 2: reconstruction weights
        n_points = len(points)
        weights = gs.zeros((n_points, n_points))
        
        for i in range(n_points):
            point = points[i]
            ref_points = self.hor_bundle.align(points[neighbours[i]], point)
            a, u, v, w = self.barycentric_projection(point, ref_points)
            weights[i][neighbours[i]] = w
        
        #Step 3: embedding coordinates
        embedding = self.embedding_coordinates(weights, d_embedding)
        
        return(embedding)
