import os
import sys
filepath = os.path.join('..',os.getcwd())
if filepath not in sys.path:
    sys.path.append(filepath)


import colour
import numpy as np
import math
import matplotlib.pyplot as plt
try:
    import colour
    HAS_COLOUR = True
except ImportError:
    HAS_COLOUR = False




def render_band_rgb(band, wavelength, illuminant_name="D65"):
    """
    Render one spectral band as a wavelength-colored image.
    """


    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    ill = colour.SDS_ILLUMINANTS[illuminant_name]

    # 1. Define a general spectral shape using your uneven wavelengths array
    # Setting interval=None explicitly accommodates non-evenly spaced arrays
    #target_shape = colour.SpectralShape(wavelengths[0], wavelengths[-1], 2.5)

    # 2. Use a Cubic Spline (or Linear) interpolator, which tolerates uneven spacing
    # SpragueInterpolator (the default) will crash with uneven intervals
    #interpolator = colour.CubicSplineInterpolator

    # 3. Interpolate the CMFs and extract individual components safely

    xbar = np.interp(wavelength, cmfs.wavelengths, cmfs.values[:, 0])
    ybar = np.interp(wavelength, cmfs.wavelengths, cmfs.values[:, 1])
    zbar = np.interp(wavelength, cmfs.wavelengths, cmfs.values[:, 2])

    #cmfs_interpolated = cmfs[wavelengths] #.copy().align(wavelengths, interpolator=interpolator)
    #print(cmfs_interpolated)
    #xbar = cmfs_interpolated[:, 0]
    #ybar = cmfs_interpolated[:, 1]
    #zbar = cmfs_interpolated[:, 2]

    E = np.interp(wavelength, ill.wavelengths, ill.values) 

    # trapezoidal integration weights
    dw = np.gradient(ill.wavelengths)
    idx = np.argmin(np.abs(ill.wavelengths - wavelength))

    wx = xbar * E * dw[idx]
    wy = ybar * E * dw[idx]
    wz = zbar * E * dw[idx]

    X = band * wx
    Y = band * wy
    Z = band * wz

    XYZ = np.stack([X, Y, Z], axis=-1)

    XYZ /= np.max(XYZ)

    rgb = colour.XYZ_to_sRGB(XYZ)

    rgb = np.clip(rgb, 0, 1)

    return rgb


def hsi_to_rgb(cube, wavelengths, illuminant_name="D65"):
    """
    Convert an HSI cube (H,W,L) into sRGB.

    Parameters
    ----------
    cube : ndarray
        (H,W,L)
    wavelengths : ndarray
        wavelengths in nm
    """

    cube = np.maximum(cube, 0)
    wavelengths = np.asarray(wavelengths)

    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    ill = colour.SDS_ILLUMINANTS[illuminant_name]

    # 1. Define a general spectral shape using your uneven wavelengths array
    # Setting interval=None explicitly accommodates non-evenly spaced arrays
    #target_shape = colour.SpectralShape(wavelengths[0], wavelengths[-1], 2.5)

    # 2. Use a Cubic Spline (or Linear) interpolator, which tolerates uneven spacing
    # SpragueInterpolator (the default) will crash with uneven intervals
    #interpolator = colour.CubicSplineInterpolator

    # 3. Interpolate the CMFs and extract individual components safely

    xbar = np.interp(wavelengths, cmfs.wavelengths, cmfs.values[:, 0])
    ybar = np.interp(wavelengths, cmfs.wavelengths, cmfs.values[:, 1])
    zbar = np.interp(wavelengths, cmfs.wavelengths, cmfs.values[:, 2])

    #cmfs_interpolated = cmfs[wavelengths] #.copy().align(wavelengths, interpolator=interpolator)
    #print(cmfs_interpolated)
    #xbar = cmfs_interpolated[:, 0]
    #ybar = cmfs_interpolated[:, 1]
    #zbar = cmfs_interpolated[:, 2]

    E = np.interp(wavelengths, ill.wavelengths, ill.values) 

    # trapezoidal integration weights
    dw = np.gradient(wavelengths)

    wx = xbar * E * dw
    wy = ybar * E * dw
    wz = zbar * E * dw

    X = np.tensordot(cube, wx, axes=([2], [0]))
    Y = np.tensordot(cube, wy, axes=([2], [0]))
    Z = np.tensordot(cube, wz, axes=([2], [0]))

    XYZ = np.stack([X, Y, Z], axis=-1)

    XYZ /= np.max(XYZ)

    rgb = colour.XYZ_to_sRGB(XYZ)

    rgb = np.clip(rgb, 0, 1)

    return rgb

def generate_mosaic_from_hsi(cube, wavelengths):
    """
    Display each hyperspectral band as a colorized image together with
    an RGB rendering of the full hyperspectral cube.

    Parameters
    ----------
    cube : ndarray (H, W, B)
        Hyperspectral image cube.
    wavelengths : ndarray (B,)
        Wavelength corresponding to each spectral band.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the mosaic.
    """
    wavelengths = np.asarray(wavelengths).flatten()
    nbands = cube.shape[2]

    # Create a roughly square grid for the band images, leaving one extra
    # column to display the RGB rendering.
    nrows = math.ceil(np.sqrt(nbands))
    ncols = nrows

    fig, ax = plt.subplots(
        nrows,
        ncols + 1,
        figsize=(2.2 * (ncols + 1), 2.2 * nrows),
        constrained_layout=True,
    )
    # TODO:
    # Revisit subplot indexing. The current implementation precomputes a mapping
    # from (row, col) -> flattened axis index so the plotting loop is independent
    # of the layout. Investigate whether this can be simplified by indexing the
    # 2D axes array directly without losing readability.

    # Flatten axes so they can be indexed with a single integer.
    ax = ax.ravel()

    # Compute flattened axis indices while skipping the final column,
    # which is reserved for the RGB image.
    band_indices = [
        np.ravel_multi_index((i, j), (nrows, ncols + 1), order="C")
        for j in range(ncols)
        for i in range(nrows)
    ]

    # Display each spectral band.
    for k in range(nbands):
        idx = band_indices[k]

        ax[idx].imshow(render_band_rgb(cube[:, :, k], wavelengths[k]))
        ax[idx].set_title(f"{wavelengths[k]:.0f} nm", fontsize=8, pad=2)
        ax[idx].axis("off")

    # Place the RGB rendering in the middle of the last column.
    rgb_row = nrows // 2
    rgb_idx = np.ravel_multi_index(
        (rgb_row, ncols),      # last column
        (nrows, ncols + 1),
        order="C",
    )

    ax[rgb_idx].imshow(hsi_to_rgb(cube, wavelengths))
    ax[rgb_idx].set_title("RGB", fontsize=11, fontweight="bold")
    ax[rgb_idx].axis("off")

    # Hide any unused axes.
    used = set(band_indices[:nbands] + [rgb_idx])
    for i, axis in enumerate(ax):
        if i not in used:
            axis.axis("off")

    plt.tight_layout()
    plt.show()

    return fig

if __name__=="__main__":

    
    from utils.datasets import load_dataset

    path_to_file = 'datasets/simulated_data_HSDC1_DB_Oct092019_5_OE.mat'

    dataset = load_dataset(path_to_file)
    print(dataset.keys())
    fig = generate_mosaic_from_hsi(dataset['X'],dataset['lambda_calib'])



