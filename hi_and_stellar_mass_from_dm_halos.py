import numpy as np

### HI-halo mass relation from Kamran et al. (2026) [https://arxiv.org/pdf/2607.20123]
'''
We use the paper's median total-HI relation (their Eq. 5), where $M_{\rm HI}$ is the HI mass summed over all galaxies belonging to the parent halo.  The fit is

$$
M_{\rm HI}(M_h)=M_h\left[
 a_1\left(\frac{M_h}{10^{10}\,h_{\rm paper}^{-1}M_\odot}\right)^\beta
 \exp\left(-\left(\frac{M_h}{M_{\rm break}}\right)^\alpha\right)
 + a_2
\right]
\exp\left[-\left(\frac{M_{\min}}{M_h}\right)^\gamma\right],
$$

with $M_{\min}=10^8\,M_\odot$ and $\gamma=0.5$.  Table 2 gives the redshift-dependent parameters.  The tabulated mass scales are in physical $M_\odot$ for the paper cosmology ($h_{\rm paper}=0.73$).

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Our halo masses are in $M_\odot/h$ for the simulation cosmology, so they are converted to physical $M_\odot$ before applying this relation. 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''
# Table 2 of Kamran et al. (2026), Eq. 5: total (central + satellite) HI mass from DM halo masss
HI_Z = np.array([0.0, 1.0, 2.1, 3.1, 3.9, 4.9])
HI_A1 = np.array([1.3e-3, 7.0e-4, 5.0e-4, 6.5e-4, 1.2e-3, 3.4e-3])
HI_A2 = np.array([5.5e-4, 1.1e-3, 1.5e-3, 1.4e-3, 1.1e-3, 0.0])
HI_ALPHA = np.array([0.63, 0.44, 0.40, 0.35, 0.31, 0.23])
HI_BETA = np.array([1.17, 1.32, 1.22, 1.05, 0.75, 0.41])
HI_LOG10_MBREAK = np.array([11.03, 10.68, 10.66, 10.50, 10.50, 10.50])

H_PAPER = 0.73
HI_MMIN_PHYS = 1e8       # Msun
HI_GAMMA = 0.5


def get_HI_fit_parameters(z):
    """Linearly interpolate the published Table-2 parameters in redshift."""

    if z < HI_Z[0] or z > HI_Z[-1]:
        raise ValueError(
            f"z={z} is outside the tabulated HI-fit range "
            f"[{HI_Z[0]}, {HI_Z[-1]}]"
        )

    a1 = np.interp(z, HI_Z, HI_A1)
    a2 = np.interp(z, HI_Z, HI_A2)
    alpha = np.interp(z, HI_Z, HI_ALPHA)
    beta = np.interp(z, HI_Z, HI_BETA)
    log10_Mbreak = np.interp(z, HI_Z, HI_LOG10_MBREAK)

    return a1, a2, alpha, beta, 10.0**log10_Mbreak

def HI_mass_from_halo_mass(Mh_h, z, h_sim=H_PAPER):
    """
    Median total HI mass for parent halos using Eq. 5 of Kamran et al. (2026).

    Parameters
    ----------
    Mh_h : array_like
        Halo masses in Msun/h, in the simulation/HMF convention.
    z : float
        Redshift.
    h_sim : float
        Simulation Hubble parameter h.

    Returns
    -------
    MHI : ndarray
        Total HI mass per halo in physical Msun.
    """

    Mh_h = np.asarray(Mh_h, dtype=float)

    # Simulation Msun/h -> physical Msun.
    Mh = Mh_h / h_sim

    a1, a2, alpha, beta, Mbreak = get_HI_fit_parameters(z)

    # 10^10 h_paper^-1 Msun expressed as a physical-Msun number.
    M0 = 1e10 / H_PAPER

    first_part = (
        a1
        * (Mh / M0)**beta
        * np.exp(-(Mh / Mbreak)**alpha)
    )

    second_part = np.exp(-(HI_MMIN_PHYS / Mh)**HI_GAMMA)

    return Mh * (first_part + a2) * second_part


### Stellar mass to halo mass relation
'''
From refernce (COSMOS2020, https://www.aanda.org/articles/aa/pdf/2022/08/aa43136-22.pdf)
'''

H_COSMOS2020 = 0.7

# Paper best-fit values from Table F.1
#STAR_Z_LO = np.array([0.2, 0.5, 0.8, 1.1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5])
STAR_Z_LO = np.array([0., 0.5, 0.8, 1.1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5])
STAR_Z_HI = np.array([0.5, 0.8, 1.1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 5.5])

STAR_LOGM1 = np.array([
    12.629, 12.793, 12.730, 12.673, 12.787,
    13.097, 12.627, 12.820, 13.638, 13.547,
])

STAR_LOGMSTAR0 = np.array([
    10.855, 10.927, 11.013, 10.967, 11.040,
    11.254, 10.920, 11.067, 12.222, 12.105,
])

STAR_BETA = np.array([
    0.487, 0.502, 0.454, 0.393, 0.410,
    0.495, 0.393, 0.465, 0.551, 0.567,
])

STAR_DELTA = np.array([
    0.935, 0.802, 1.109, 0.746, 0.716,
    0.668, 0.274, 0.354, 1.557, 1.427,
])

STAR_GAMMA = np.array([
    1.939, 3.132, 1.925, 0.335, 1.312,
    1.077, 0.446, 0.741, 3.149, 3.225,
])


def get_stellar_parameter_index(z):
    """Return the Shuntov+22 redshift-bin index appropriate for z."""
    z = float(z)
    idx = np.searchsorted(STAR_Z_HI, z, side='right')
    idx = min(idx, len(STAR_Z_LO) - 1)
    return int(idx)

def get_stellar_parameters(z):
    """Best-fit parameters for the chosen z-bin."""
    i = get_stellar_parameter_index(z)

    return {
        'index': i,
        'z_lo': STAR_Z_LO[i],
        'z_hi': STAR_Z_HI[i],
        'logM1': STAR_LOGM1[i],
        'logMstar0': STAR_LOGMSTAR0[i],
        'beta': STAR_BETA[i],
        'delta': STAR_DELTA[i],
        'gamma': STAR_GAMMA[i]
    }

############################################################
'''
The equation we refer to is Eq. 9, but it gives the halo mass
given the stellar mass, so we interpolate on a grid and then 
invert, since our goal is to estimate the stellar mass starting 
from halo mass.
'''
def shuntov_logmhalo_from_logmstar(log10_Mstar, pars):
    """Eq. 9 evaluated directly in log10"""
    log10_Mstar = np.asarray(log10_Mstar, dtype=float)
    logx = log10_Mstar - pars['logMstar0']
    x = 10.0**logx

    return (
        pars['logM1']
        + pars['beta'] * logx
        + x**pars['delta'] / (1.0 + x**(-pars['gamma']))
        - 0.5
    )

############################################################

H_DEMNUNI = 0.677
'''
We now extrapolate the stellar mass from halo mass on a grid
'''
def stellar_mass_from_halo_mass(Mh_h, z, h_sim=H_DEMNUNI):
    """Numerically invert Eq. 9 to obtain the central Mstar(Mh)."""
    Mh_phys = np.asarray(Mh_h / h_sim, dtype=float)
    pars = get_stellar_parameters(z)

    logMstar_inv = np.linspace(6.0, 13.5, 10_000)
    logMh_inv = shuntov_logmhalo_from_logmstar(logMstar_inv, pars)

    return 10.0**np.interp(
        np.log10(Mh_phys),
        logMh_inv,
        logMstar_inv,
        left=logMstar_inv[0],
        right=logMstar_inv[-1],
    )
