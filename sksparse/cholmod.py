# encoding: utf-8
# module sksparse.cholmod
# from /home/marialaura/anaconda3/envs/ioh/lib/python3.9/site-packages/sksparse/cholmod.cpython-39-x86_64-linux-gnu.so
# by generator 1.147
# no doc

# imports
import builtins as __builtins__ # <module 'builtins' (built-in)>
import warnings as warnings # /home/marialaura/anaconda3/envs/ioh/lib/python3.9/warnings.py
import numpy as np # /home/marialaura/anaconda3/envs/ioh/lib/python3.9/site-packages/numpy/__init__.py
import scipy.sparse as sparse # /home/marialaura/anaconda3/envs/ioh/lib/python3.9/site-packages/scipy/sparse/__init__.py
import scipy.sparse.base as __scipy_sparse_base


# functions

def analyze(*args, **kwargs): # real signature unknown
    """
    Computes the optimal fill-reducing permutation for the symmetric matrix
        A, but does *not* factor it (i.e., it performs a "symbolic Cholesky
        decomposition"). This function ignores the actual contents of the matrix
        A. All it cares about are (1) which entries are non-zero, and (2) whether
        A has real or complex type.
    
        :param A: The matrix to be analyzed.
    
        :param mode: Specifies which algorithm should be used to (eventually)
          compute the Cholesky decomposition -- one of "simplicial", "supernodal",
          or "auto". See the CHOLMOD documentation for details on how "auto" chooses
          the algorithm to be used.
    
        :param ordering_method: Specifies which ordering algorithm should be used to
          (eventually) order the matrix A -- one of "natural", "amd", "metis",
          "nesdis", "colamd", "default" and "best". "natural" means no permutation.
          See the CHOLMOD documentation for more details.
    
        :param use_long: Specifies if the long type (64 bit) or the int type
          (32 bit) should be used for the indices of the sparse matrices. If
          use_long is None try to estimate if long type is needed.
    
        :returns: A :class:`Factor` object representing the analysis. Many
          operations on this object will fail, because it does not yet hold a full
          decomposition. Use :meth:`Factor.cholesky_inplace` (or similar) to
          actually factor a matrix.
    """
    pass

def analyze_AAt(*args, **kwargs): # real signature unknown
    """
    Computes the optimal fill-reducing permutation for the symmetric matrix
        :math:`AA'`, but does *not* factor it (i.e., it performs a "symbolic
        Cholesky decomposition"). This function ignores the actual contents of the
        matrix A. All it cares about are (1) which entries are non-zero, and (2)
        whether A has real or complex type.
    
        :param A: The matrix to be analyzed.
    
        :param mode: Specifies which algorithm should be used to (eventually)
          compute the Cholesky decomposition -- one of "simplicial", "supernodal",
          or "auto". See the CHOLMOD documentation for details on how "auto" chooses
          the algorithm to be used.
    
        :param ordering_method: Specifies which ordering algorithm should be used to
          (eventually) order the matrix A -- one of "natural", "amd", "metis",
          "nesdis", "colamd", "default" and "best". "natural" means no permutation.
          See the CHOLMOD documentation for more details.
    
        :param use_long: Specifies if the long type (64 bit) or the int type
          (32 bit) should be used for the indices of the sparse matrices. If
          use_long is None try to estimate if long type is needed.
    
        :returns: A :class:`Factor` object representing the analysis. Many
          operations on this object will fail, because it does not yet hold a full
          decomposition. Use :meth:`Factor.cholesky_AAt_inplace` (or similar) to
          actually factor a matrix.
    """
    pass

def cholesky(*args, **kwargs): # real signature unknown
    """
    Computes the fill-reducing Cholesky decomposition of
    
          .. math:: A + \beta I
    
        where ``A`` is a sparse, symmetric, positive-definite matrix, preferably
        in CSC format, and ``beta`` is any real scalar (usually 0 or 1). (And
        :math:`I` denotes the identity matrix.)
    
        Only the lower triangular part of ``A`` is used.
    
        ``mode`` is passed to :func:`analyze`.
    
        ``ordering_method`` is passed to :func:`analyze`.
    
        ``use_long`` is passed to :func:`analyze`.
    
        :returns: A :class:`Factor` object represented the decomposition.
    """
    pass

def cholesky_AAt(*args, **kwargs): # real signature unknown
    """
    Computes the fill-reducing Cholesky decomposition of
    
          .. math:: AA' + \beta I
    
        where ``A`` is a sparse matrix, preferably in CSC format, and ``beta`` is
        any real scalar (usually 0 or 1). (And :math:`I` denotes the identity
        matrix.)
    
        Note that if you are solving a conventional least-squares problem, you
        will need to transpose your matrix before calling this function, and
        therefore it will be somewhat more efficient to construct your matrix in
        CSR format (so that its transpose will be in CSC format).
    
        ``mode`` is passed to :func:`analyze_AAt`.
    
        ``ordering_method`` is passed to :func:`analyze_AAt`.
    
        ``use_long`` is passed to :func:`analyze_AAt`.
    
        :returns: A :class:`Factor` object represented the decomposition.
    """
    pass

def _analyze(*args, **kwargs): # real signature unknown
    pass

def _check_for_csc(*args, **kwargs): # real signature unknown
    pass

def _cholesky(*args, **kwargs): # real signature unknown
    pass

def __reduce_cython__(*args, **kwargs): # real signature unknown
    pass

def __setstate_cython__(*args, **kwargs): # real signature unknown
    pass

# classes

class CholmodError(Exception):
    # no doc
    def __init__(self, *args, **kwargs): # real signature unknown
        pass

    __weakref__ = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """list of weak references to the object (if defined)"""



class CholmodGpuProblemError(CholmodError):
    # no doc
    def __init__(self, *args, **kwargs): # real signature unknown
        pass


class CholmodInvalidError(CholmodError):
    # no doc
    def __init__(self, *args, **kwargs): # real signature unknown
        pass


class CholmodNotInstalledError(CholmodError):
    # no doc
    def __init__(self, *args, **kwargs): # real signature unknown
        pass


class CholmodNotPositiveDefiniteError(CholmodError):
    # no doc
    def __init__(self, *args, **kwargs): # real signature unknown
        pass


class CholmodOutOfMemoryError(CholmodError):
    # no doc
    def __init__(self, *args, **kwargs): # real signature unknown
        pass


class CholmodTooLargeError(CholmodError):
    # no doc
    def __init__(self, *args, **kwargs): # real signature unknown
        pass


class CholmodWarning(UserWarning):
    # no doc
    def __init__(self, *args, **kwargs): # real signature unknown
        pass

    __weakref__ = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """list of weak references to the object (if defined)"""



class CholmodTypeConversionWarning(CholmodWarning, __scipy_sparse_base.SparseEfficiencyWarning):
    # no doc
    def __init__(self, *args, **kwargs): # real signature unknown
        pass


class Common(object):
    # no doc
    def _print(self, *args, **kwargs): # real signature unknown
        pass

    def _print_dense(self, *args, **kwargs): # real signature unknown
        pass

    def _print_sparse(self, *args, **kwargs): # real signature unknown
        pass

    def __init__(self, *args, **kwargs): # real signature unknown
        pass

    @staticmethod # known case of __new__
    def __new__(*args, **kwargs): # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    def __reduce__(self, *args, **kwargs): # real signature unknown
        pass

    def __setstate__(self, *args, **kwargs): # real signature unknown
        pass

    __pyx_vtable__ = None # (!) real value is '<capsule object NULL at 0x7f991fed6c00>'


class Factor(object):
    """
    This class represents a Cholesky decomposition with a particular
        fill-reducing permutation. It cannot be instantiated directly; see
        :func:`analyze` and :func:`cholesky`, both of which return objects of type
        Factor.
    """
    def apply_P(self, *args, **kwargs): # real signature unknown
        """ Returns :math:`x`, where :math:`x = Pb`. """
        pass

    def apply_Pt(self, *args, **kwargs): # real signature unknown
        """ Returns :math:`x`, where :math:`x = P'b`. """
        pass

    def cholesky(self, *args, **kwargs): # real signature unknown
        """
        The same as :meth:`cholesky_inplace` except that it first creates
                a copy of the current :class:`Factor` and modifes the copy.
        
                :returns: The new :class:`Factor` object.
        """
        pass

    def cholesky_AAt(self, *args, **kwargs): # real signature unknown
        """
        The same as :meth:`cholesky_AAt_inplace` except that it first
                creates a copy of the current :class:`Factor` and modifes the copy.
        
                :returns: The new :class:`Factor` object.
        """
        pass

    def cholesky_AAt_inplace(self, *args, **kwargs): # real signature unknown
        """
        The same as :meth:`cholesky_inplace`, except it factors :math:`AA'
                + \beta I` instead of :math:`A + \beta I`.
        """
        pass

    def cholesky_inplace(self, *args, **kwargs): # real signature unknown
        """
        Updates this Factor so that it represents the Cholesky
                decomposition of :math:`A + \beta I`, rather than whatever it
                contained before.
        
                :math:`A` must have the same pattern of non-zeros as the matrix used
                to create this factor originally.
        """
        pass

    def copy(self, *args, **kwargs): # real signature unknown
        """
        Copies the current :class:`Factor`.
        
                :returns: A new :class:`Factor` object.
        """
        pass

    def D(self, *args, **kwargs): # real signature unknown
        """
        Converts this factorization to the style
        
                  .. math:: LDL' = PAP'
        
                or
        
                  .. math:: LDL' = PAA'P'
        
                and then returns the diagonal matrix D *as a 1d vector*.
        
                  .. note:: This method uses an efficient implementation that extracts
                     the diagonal D directly from CHOLMOD's internal
                     representation. It never makes a copy of the factor matrices, or
                     actually converts a full `LL'` factorization into an `LDL'`
                     factorization just to extract `D`.
        """
        pass

    def det(self, *args, **kwargs): # real signature unknown
        """
        Computes the determinant of the matrix A.
        
                Consider using :meth:`logdet` instead, for improved numerical
                stability. (In particular, determinants are often prone to problems
                with underflow or overflow.)
        
                .. versionadded:: 0.2
        """
        pass

    def inv(self): # real signature unknown; restored from __doc__
        """
        Returns the inverse of the matrix A, as a sparse (CSC) matrix.
        
                  .. warning:: For most purposes, it is better to use :meth:`solve`
                     instead of computing the inverse explicitly. That is, the
                     following two pieces of code produce identical results::
        
                       x = f.solve(b)
                       x = f.inv() * b  # DON'T DO THIS!
        
                     But the first line is both faster and produces more accurate
                     results.
        
                Sometimes, though, you really do need the inverse explicitly (e.g.,
                for calculating standard errors in least squares regression), so if
                that's your situation, here you go.
        
                .. versionadded:: 0.2
        """
        pass

    def L(self, *args, **kwargs): # real signature unknown
        """
        If necessary, converts this factorization to the style
        
                  .. math:: LL' = PAP'
        
                or
        
                  .. math:: LL' = PAA'P'
        
                and then returns the sparse lower-triangular matrix L.
        
                .. warning:: The L matrix returned by this method and the one returned
                   by :meth:`L_D` are different!
        """
        pass

    def LD(self, *args, **kwargs): # real signature unknown
        """
        If necessary, converts this factorization to the style
        
                  .. math:: LDL' = PAP'
        
                or
        
                  .. math:: LDL' = PAA'P'
        
                and then returns a sparse lower-triangular matrix "LD", which contains
                the D matrix on its diagonal, plus the below-diagonal part of L (the
                actual diagonal of L is all-ones).
        
                See :meth:`L_D` for a more convenient interface.
        """
        pass

    def logdet(self): # real signature unknown; restored from __doc__
        """
        Computes the (natural) log of the determinant of the matrix A.
        
                If `f` is a factor, then `f.logdet()` is equivalent to
                `np.sum(np.log(f.D()))`.
        
                .. versionadded:: 0.2
        """
        pass

    def L_D(self, *args, **kwargs): # real signature unknown
        """
        If necessary, converts this factorization to the style
        
                  .. math:: LDL' = PAP'
        
                or
        
                  .. math:: LDL' = PAA'P'
        
                and then returns the pair (L, D) where L is a sparse lower-triangular
                matrix and D is a sparse diagonal matrix.
        
                .. warning:: The L matrix returned by this method and the one returned
                   by :meth:`L` are different!
        """
        pass

    def P(self, *args, **kwargs): # real signature unknown
        """
        Returns the fill-reducing permutation P, as a vector of indices.
        
                The decomposition :math:`LL'` or :math:`LDL'` is of::
        
                  A[P[:, np.newaxis], P[np.newaxis, :]]
        
                (or similar for AA').
        """
        pass

    def slogdet(self, *args, **kwargs): # real signature unknown
        """
        Computes the log-determinant of the matrix A, with the same API as
                :meth:`numpy.linalg.slogdet`.
        
                This returns a tuple `(sign, logdet)`, where `sign` is always the
                number 1.0 (because the determinant of a positive-definite matrix is
                always a positive real number), and `logdet` is the (natural)
                logarithm of the determinant of the matrix A.
        
                .. versionadded:: 0.2
        """
        pass

    def solve_A(self, *args, **kwargs): # real signature unknown
        """
        Solves a linear system.
        
                :param b: right-hand-side
        
                :returns: math:`x`, where :math:`Ax = b` (or :math:`AA'x = b`, if
                you used :func:`cholesky_AAt`).
        
                :meth:`__call__` is an alias for this function, i.e., you can simply
                call the :class:`Factor` object like a function to solve :math:`Ax =
                b`.
        """
        pass

    def solve_D(self, *args, **kwargs): # real signature unknown
        """ Returns :math:`x`, where :math:`Dx = b`. """
        pass

    def solve_DLt(self, *args, **kwargs): # real signature unknown
        """
        Solves a linear system.
        
                :param b: right-hand-side
        
                :returns: math:`x`, where :math:`DL'x = b`.
        """
        pass

    def solve_L(self, *args, **kwargs): # real signature unknown
        """
        Solves a linear system.
        
                :param b: right-hand-side
        
                :param use_LDLt_decomposition: If True, use the `L` of the `LDL'`
                  decomposition. If False, use the `L` of the `LL'` decomposition.
        
                :returns: math:`x`, where :math:`Lx = b`.
        """
        pass

    def solve_LD(self, *args, **kwargs): # real signature unknown
        """
        Solves a linear system.
        
                :param b: right-hand-side
        
                :returns: math:`x`, where :math:`LDx = b`.
        """
        pass

    def solve_LDLt(self, *args, **kwargs): # real signature unknown
        """
        Solves a linear system.
        
                :param b: right-hand-side
        
                :returns: math:`x`, where :math:`LDL'x = b`.
        
                (This is different from :meth:`solve_A` because it does not correct
                for the fill-reducing permutation.)
        """
        pass

    def solve_Lt(self, *args, **kwargs): # real signature unknown
        """
        Solves a linear system.
        
                :param b: right-hand-side
        
                :param use_LDLt_decomposition: If True, use the `L` of the `LDL'`
                  decomposition. If False, use the `L` of the `LL'` decomposition.
        
                :returns: math:`x`, where :math:`L'x = b`.
        """
        pass

    def update_inplace(self, *args, **kwargs): # real signature unknown
        """
        Incremental building of :math:`AA'` decompositions.
        
                Updates this factor so that instead of representing the decomposition
                of :math:`A` (:math:`AA'`), it computes the decomposition of
                :math:`A + CC'` (:math:`AA' + CC'`) for ``subtract=False`` which is the
                default, or :math:`A - CC'` (:math:`AA' - CC'`) for
                ``subtract=True``. This method does not require that the
                :class:`Factor` was created with :func:`cholesky_AAt`, though that
                is the common case.
        
                The usual use for this is to factor AA' when A has a large number of
                columns, or those columns become available incrementally. Instead of
                loading all of A into memory, one can load in 'strips' of columns and
                pass them to this method one at a time.
        
                Note that no fill-reduction analysis is done; whatever permutation was
                chosen by the initial call to :func:`analyze` will be used regardless
                of the pattern of non-zeros in C.
        """
        pass

    def _cholesky_inplace(self, *args, **kwargs): # real signature unknown
        pass

    def _ensure_L_or_LD_inplace(self, *args, **kwargs): # real signature unknown
        pass

    def _L_or_LD(self, *args, **kwargs): # real signature unknown
        pass

    def _print(self, *args, **kwargs): # real signature unknown
        pass

    def _solve(self, *args, **kwargs): # real signature unknown
        pass

    def _solve_dense(self, *args, **kwargs): # real signature unknown
        pass

    def _solve_sparse(self, *args, **kwargs): # real signature unknown
        pass

    def __call__(self, *args, **kwargs): # real signature unknown
        """ Alias for :meth:`solve_A`. """
        pass

    def __init__(self, *args, **kwargs): # real signature unknown
        pass

    @staticmethod # known case of __new__
    def __new__(*args, **kwargs): # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    def __reduce__(self, *args, **kwargs): # real signature unknown
        pass

    def __setstate__(self, *args, **kwargs): # real signature unknown
        pass

    _common = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default



class _CholmodDenseDestructor(object):
    """
    This is a destructor for NumPy arrays based on dense data of Cholmod.
        Use this only once for each Cholmod dense array. Otherwise memory will be
        freed multiple times.
    """
    def __init__(self, *args, **kwargs): # real signature unknown
        pass

    @staticmethod # known case of __new__
    def __new__(*args, **kwargs): # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    def __reduce__(self, *args, **kwargs): # real signature unknown
        pass

    def __setstate__(self, *args, **kwargs): # real signature unknown
        pass

    __pyx_vtable__ = None # (!) real value is '<capsule object NULL at 0x7f991fed6cf0>'


class _CholmodSparseDestructor(object):
    """
    This is a destructor for NumPy arrays based on sparse data of Cholmod.
        Use this only once for each Cholmod sparse array. Otherwise memory will be
        freed multiple times.
    """
    def __init__(self, *args, **kwargs): # real signature unknown
        pass

    @staticmethod # known case of __new__
    def __new__(*args, **kwargs): # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    def __reduce__(self, *args, **kwargs): # real signature unknown
        pass

    def __setstate__(self, *args, **kwargs): # real signature unknown
        pass

    __pyx_vtable__ = None # (!) real value is '<capsule object NULL at 0x7f991fed6c90>'


# variables with complex values

_modes = {
    'auto': 1,
    'simplicial': 0,
    'supernodal': 2,
}

_ordering_methods = {
    'amd': 2,
    'best': None,
    'colamd': 5,
    'default': None,
    'metis': 3,
    'natural': 0,
    'nesdis': 4,
}

__all__ = [
    'analyze',
    'analyze_AAt',
    'cholesky',
    'cholesky_AAt',
]

__loader__ = None # (!) real value is '<_frozen_importlib_external.ExtensionFileLoader object at 0x7f991ff4c640>'

__spec__ = None # (!) real value is "ModuleSpec(name='sksparse.cholmod', loader=<_frozen_importlib_external.ExtensionFileLoader object at 0x7f991ff4c640>, origin='/home/marialaura/anaconda3/envs/ioh/lib/python3.9/site-packages/sksparse/cholmod.cpython-39-x86_64-linux-gnu.so')"

__test__ = {}

