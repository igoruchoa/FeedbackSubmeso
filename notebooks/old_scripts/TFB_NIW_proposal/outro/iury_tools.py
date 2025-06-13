import numpy as np
import xarray as xr
from xhistogram.xarray import histogram

def azimuthal_integration(Fa, dim_names=("freq_xC", "freq_yC"), kmax=None, dk=0.002, chunk=dict(freq_time=10)):
    """
    Calculates the azimuthal integration of a 2D Fourier array (Fa).

    Args:
        Fa (xarray.DataArray): The 2D Fourier array to integrate.
        dim_names (tuple, optional): A tuple containing the names of the dimensions
            corresponding to the wavenumbers in the x and y directions. Defaults to ("freq_xC", "freq_yC").
        kmax (float, optional): The maximum value of the radial wavenumber (k).
            If None, it will be automatically determined from the data. Defaults to None.
        dk (float, optional): The bin size for the radial wavenumber (k) integration. Defaults to 0.002.
        chunk (dict, optional): A dictionary specifying chunking parameters for the xarray operations.
            Defaults to {'freq_time': 10}.

    Returns:
        xarray.DataArray: The azimuthally integrated spectrum (Pk). It will have a single dimension 'k'.
    """

    # Check for valid input dimensions
    if all(dim not in Fa.dims for dim in dim_names):
        raise ValueError(f"Input array (Fa) must be a 2D array with dimensions {dim_names}.")

    # Automatically determine kmax if not provided
    if kmax is None:
        kmax = np.sqrt((Fa[dim_names[0]]**2 + Fa[dim_names[1]]**2).max())

    # Create k bins
    kbins = np.arange(0, kmax + dk, dk)  # Ensure kmax is included

    # Calculate magnitude of k (radial wavenumber)
    K = np.sqrt(Fa[dim_names[0]]**2 + Fa[dim_names[1]]**2)

    # Calculate effective area element
    area = (np.diff(Fa[dim_names[0]]) * np.diff(Fa[dim_names[1]])).mean()

    # Calculate the scaling for the azimuthally-integrating power spectrum
    scaling = area / dk
    
    # Power spectrum
    P = ((np.abs(Fa)**2)).chunk(**chunk)

    # Broadcast K to match the shape of P and rename the dimension
    K = (K * xr.ones_like(P)).rename("K")

    # Perform histogram with weighting and appropriate dimensions
    kw = dict(bins=kbins, dim=dim_names, keep_coords=True)
    Pk = (histogram(K, weights=P, **kw)).load() * scaling

    return Pk