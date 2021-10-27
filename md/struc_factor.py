



def conserved_var_fourier(ds, Nstart, Nend, box_size, n, max_wavelength=1.):
    """
    Compute Fourier coefficients of conserved variables (mass and momentum density)
    from chunks of a trajectory (netCDF4 format)
    Parameters
    ----------
    ds : netCDF4.Dataset
        dataset containing the coordinates and velocities of the atoms
    Nstart : int
        first atom number in chunk
    Nend : int
        last atom number in chunk
    n : int
        multiple of the smallest discrete wave vector
    max_wavelength : float
        maximum wavelength, multiples of the corresponding box length (default = 1)
    Returns
    ----------
    numpy.ndarray
        time series of mass density Fourier coefficients
    numpy.ndarray
        time series of longitudinal momentum density Fourier coefficients
    numpy.ndarray
        time series of transverse momentum density Fourier coefficients
    """

    coords = np.array(ds.variables["coordinates"][:, Nstart:Nend, :]).astype(np.float32)
    vels = np.array(ds.variables["velocities"][:, Nstart:Nend, :]).astype(np.float32)

    Lx, Ly, Lz = box_size

    kx = 2. * n * np.pi / (max_wavelength * Lx)
    ky = 2. * n * np.pi / (max_wavelength * Ly)
    kz = 2. * n * np.pi / (max_wavelength * Lz)

    rho_tk = np.empty(coords.shape[::2], dtype=np.complex64)



    rho_tk[:, 0] = np.sum(np.exp(-1.j * kx * coords[:, :, 0]), axis=1) / (N_l)
    rho_tk[:, 1] = np.sum(np.exp(-1.j * ky * coords[:, :, 1]), axis=1) / (N_l)
    rho_tk[:, 2] = np.sum(np.exp(-1.j * kz * coords[:, :, 2]), axis=1) / (N_l)


    return rho_tk
