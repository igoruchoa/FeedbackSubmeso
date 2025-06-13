"""
This script performs spectral analysis on velocity data from a set of simulations. It calculates power spectra for the u, v, and w velocity components at various depths.
"""

# Standard Library Imports
import sys
from glob import glob

# Third-Party Imports
import dask
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xrft
from dask.distributed import Client, LocalCluster
from tqdm import tqdm
from xhistogram.xarray import histogram

# Local Module Imports
sys.path.append("../00-simulations") 
from run_all import simulations, simfolder
from tools import azimuthal_integration  # Custom tool for azimuthal integration

# Dask Cluster Setup for Parallel Processing
cluster = LocalCluster(n_workers=24, dashboard_address=":4242")  # Adjust workers if needed
client = Client(cluster)

# Constants and Parameters
TIME_SLICE = slice(0, 5 * 24)  # First 5 days of data
K_MAX = 0.07
FREQ_MAX = 4
FFT_KWARGS = dict(dim=["xC", "yC", "time"], true_phase=True, true_amplitude=True)

# Helper Functions
def integrate_dataset(ds):
    """Integrates a dataset over its dimensions."""
    return ds.sum() * np.product([np.diff(ds[dim]).mean() for dim in ds.dims])

def format_time(ds):
    """Converts time coordinate to hours."""
    return ds.assign_coords(time=(ds.time.astype("float") * 1e-9) / 3600)


for sim in simulations:
    print(sim["name"])  # Display current simulation

    # File Path and Simulation Name
    path = f"../../data/{simfolder.format(**sim)}"
    sim_name = f"Nxy{sim['Nxy']}Lxy{sim['Lxy']}_Nz{sim['Nz']}Lz{sim['Lz']}"

    # Load and Process Data
    ds = xr.open_dataset(f"{path}/vxyz_{sim_name}_mlwaves.nc").sortby("zC")
    ds = format_time(ds)

    check = True
    power_spectra = []
    for var in ["u", "v", "w"]:  # Iterate over velocity components
        print(f"\n{var}\n")

        spectra_at_levels = []
        for zi in tqdm(ds.zC.values):
            # Extract and preprocess data at a z-level
            da = ds[var].sel(zC=zi, method="nearest").squeeze()
            da = da.chunk(time=10).load()  # Load chunked data for parallel processing
            da = da - da.mean("time")  # Remove mean
            
            # Calculate Power Spectrum
            fft_result = xrft.fft(da, **FFT_KWARGS)
            dk = np.sqrt(fft_result.freq_xC.spacing*2 + fft_result.freq_yC.spacing*2)
            power_spectrum = azimuthal_integration(fft_result, dk=dk)
            
            if check:
                # Check Parseval's Theorem
                parseval_holds = np.allclose(
                    integrate_dataset(np.abs(fft_result) ** 2), 
                    integrate_dataset(power_spectrum),
                    atol = 0.1,
                )
                if not parseval_holds:
                    print("Warning: Parseval's theorem not satisfied.")
                else:
                    print("Parseval's theorem is satisfied.")
                    check = False
            
            power_spectrum = power_spectrum.sel(freq_time=slice(0, FREQ_MAX)) ** 2  # Scale spectrum
            
            # Select the relevant wavenumber range up to K_MAX for analysis
            power_spectrum = power_spectrum.sel(K_bin = slice(0, K_MAX))
            spectra_at_levels.append(power_spectrum)

        # Combine Spectra at Different Levels
        spectra_at_levels = xr.concat(spectra_at_levels, "zC").rename(var)
        power_spectra.append(spectra_at_levels)

    # Merge and Save Power Spectra
    power_spectra = xr.merge(power_spectra)
    power_spectra.to_netcdf(f"{path}/v_spectra_{sim_name}_mlwaves.nc")