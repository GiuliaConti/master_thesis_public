import time
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import warnings
import multiprocessing
import os, datetime
from astropy import units as u

from lisatools.diagnostic import *
from lisatools.sensitivity import SensitivityMatrix, A1TDISens, E1TDISens, LISASens
from lisatools.utils.constants import *
from lisatools.detector import ESAOrbits, EqualArmlengthOrbits
from lisatools.datacontainer import DataResidualArray 
from lisatools.analysiscontainer import AnalysisContainer

from few.waveform import GenerateEMRIWaveform
from few.utils.constants import *
from few.trajectory.inspiral import EMRIInspiral
from few.utils.utility import get_p_at_t
from few.utils.fdutils import *

from eryn.ensemble import EnsembleSampler
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.state import State
from scipy.signal.windows import tukey, hann, boxcar, nuttall, blackman
from eryn.backends import HDFBackend

from fastlisaresponse import pyResponseTDI, ResponseWrapper



# gpu
use_gpu = True



# metric
metric = "FastKerrEccentricEquatorialFlux"   # Kerr
traj = "KerrEccEqFlux"

#metric = "FastSchwarzschildEccentricFlux"    # Schw
#traj = "SchwarzEccFlux"



# Observation parameters
Tobs = 1  # [years]
dt = 50.0  # [s]
eps = 1e-5  # mode content

emri_waveform_kwargs = dict(T=Tobs, dt=dt, eps=eps)



# Waveform parameters
M = 1e6  # central object mass
mu = 10  # secondary object mass
a = 0.5  # spin (will be ignored in Schwarzschild waveform)
p0 = 8.2  # initial semi-latus rectum
e0 = 0.5  # eccentricity
x0 = 1.0  # cosine of inclination 
dist = 1.0  # distance

qK = np.pi / 6  # polar spin angle (theta)
phiK = np.pi / 3  # azimuthal viewing angle
qS = np.pi / 6  # polar sky angle
phiS = np.pi / 3  # azimuthal viewing angle

Phi_phi0 = np.pi / 3
Phi_theta0 = np.pi / 6
Phi_r0 = np.pi / 3



emri_waveform_args = [
    M,
    mu,
    a,
    p0,
    e0,
    x0,
    dist,
    qS,
    phiS,
    qK,
    phiK,
    Phi_phi0,
    Phi_theta0,
    Phi_r0,
]



# TDI
tdi_chan="AE"
tdi_labels=["A", "E"]

#tdi_chan="AET"
#tdi_labels=["A", "E", "T"]

orbit_file_esa = "equalarmlength-trailing-fit.h5"
#orbit_file_esa = "esa-trailing-orbits.h5"

orbit_kwargs_esa = dict(orbit_file=orbit_file_esa)

order = 25 # order of the langrangian interpolation (for strain and orbits?)

tdi_gen="1st generation"
#tdi_gen="2nd generation"


response_kwargs = dict(
        Tobs=Tobs,
        dt=dt,
        t0 = 100000.0,  # time at which signal starts (chops off data at start and end of waveform)
        order = order, # order of the langrangian interpolation (for strain and orbits?)
        index_beta = 7,   # Sky location parameters: theta --> qS
        index_lambda = 8, #                          phi --> phiS
        tdi=tdi_gen, 
        tdi_chan=tdi_chan,
        orbit_kwargs=orbit_kwargs_esa,
    )



# Initialise generator
td_gen = GenerateEMRIWaveform(
        metric,
        sum_kwargs=dict(pad_output=True, odd_len=True),
        return_list=False,
        use_gpu=use_gpu,
    )



lisa_response = ResponseWrapper(waveform_gen=td_gen,
                                flip_hx=True,
                                use_gpu=use_gpu,
                                remove_sky_coords=False,
                                is_ecliptic_latitude=False,
                                remove_garbage=True,
                                **response_kwargs)




def fastlisaresponse(*params, emri_waveform_kwargs=None):
    return lisa_response(*params, **(emri_waveform_kwargs or {}))


# Generate a waveform
start = time.time()
chans = fastlisaresponse(
    *emri_waveform_args,
    emri_waveform_kwargs=emri_waveform_kwargs,
)
print(f"Waveform generation took {time.time()-start:.2f} s")



# Compute f_min
r_apo = p0 * M * MTSUN_SI / (1 - e0)  # meters
r_per = p0 * M * MTSUN_SI / (1 + e0)  # meters
r0 = (r_apo+r_per)/2

f_orb_0 = np.sqrt(M * MTSUN_SI / r0**3) / (2 * np.pi)  # rad/s

f_min = 2 * f_orb_0



# conversions
day  = u.day        # Quantity unit (1 day)
hour = u.hour       # Quantity unit (1 hour)
year = 365.25 * u.day   # Julian year as a Quantity

years_to_s = (year).to(u.s).value
days_to_s = (day).to(u.s)
hours_to_s = (hour).to(u.s)





# Generate TD noise
def generate_time_domain_noise_lisa(pos_psd, dt, N_obs):

    sigma = 0.5 * (pos_psd / (1.0 / (N_obs * dt))) ** 0.5

    noise_real = np.random.normal(0.0, sigma) + 1j * np.random.normal(0.0, sigma)

    noise_time_domain = np.fft.irfft(noise_real, n=N_obs) / dt

    return noise_time_domain


def add_TDI_noise(chans, dt, f_min):
    N_obs = len(chans[0])
    freq = np.fft.fftshift(np.fft.fftfreq(N_obs , dt))
    positive_frequency_mask = (freq>f_min)

    # generate psd
    pos_psd_A = get_sensitivity(freq[positive_frequency_mask], sens_fn='A1TDISens', return_type="PSD")
    pos_psd_E = get_sensitivity(freq[positive_frequency_mask], sens_fn='E1TDISens', return_type="PSD")

    # generate two realizations for the noise (A and E)
    noise_td_A = cp.asarray(generate_time_domain_noise_lisa(pos_psd_A, dt, N_obs))
    noise_td_E = cp.asarray(generate_time_domain_noise_lisa(pos_psd_E, dt, N_obs))

    # add TD noise to TD signal
    chans_noise = chans.copy()
    chans_noise[0] = chans[0] + noise_td_A
    chans_noise[1] = chans[1] + noise_td_E

    return chans_noise


    
chans_noise = add_TDI_noise(chans, dt, f_min)


# Add gaps
def gen_periodic_gaps(chans, dt, w_total, gap_s, period_s, seed=None):

    rng = cp.random.default_rng(seed)

    N_tot = len(chans[0])

    # conversion of gaps and their period to samples
    gap_samples = int(round(float(gap_s) / dt))
    period_samples = int(round(float(period_s) / dt))

    # random start within one period
    start_idx = int(rng.integers(0, period_samples))
    
    # create mask
    w_local = cp.ones(N_tot, dtype=int)
    starts = cp.arange(start_idx, N_tot, period_samples)
    for s in starts:
        e = min(s + gap_samples, N_tot)  # within the end of the signal
        w_local[s:e] = 0

    # apply mask
    chans_gap = chans.copy()
    chans_gap[0] = chans[0] * w_local
    chans_gap[1] = chans[1] * w_local
    w_final = w_total * w_local

    return chans_gap, w_final, starts


def gen_planned_gaps(chans, dt, seed=None):
    N_tot = len(chans[0])
    w_total = cp.ones(N_tot, dtype=int)
    
    # Antenna re-pointing    
    gap_ant = 3 * hours_to_s
    period_ant = 14 * days_to_s

    chans_gap_ant, w_final_ant, starts_ant = gen_periodic_gaps(chans, dt, w_total, gap_ant, period_ant, seed=seed)

    
    # Tilt-to-length coupling constant estimation
    gap_ttl = 2 * days_to_s
    period_ttl = 0.25 * years_to_s # 4 times a year

    chans_gap_ttl, w_final_ttl, starts_ttl = gen_periodic_gaps(chans_gap_ant, dt, w_final_ant, gap_ttl, period_ttl, seed=seed)
    

    # point-ahead angle mechanism (PAAM) adjustments
    gap_PAAM = 100
    period_PAAM = (1/3) * days_to_s

    chans_gap_PAAM, w_final_PAAM, starts_PAAM = gen_periodic_gaps(chans_gap_ttl, dt, w_final_ttl, gap_PAAM, period_PAAM, seed=seed)


    return chans_gap_PAAM, w_final_PAAM, starts_ant, starts_ttl, starts_PAAM


chans_gap, w_final, starts_ant, starts_ttl, starts_PAAM = gen_planned_gaps(chans_noise, dt, seed=13)




# Visualise the signal

time_array = np.arange(0,len(chans_gap[0].get()))*dt

plt.plot(time_array, chans[0].get(),label='h')
plt.plot(time_array, chans_gap[0].get(),label='h + n (gaps)', alpha=0.2, linewidth=0.5)
plt.ylabel(r'$h_{+}(t)$')
plt.xlabel(r'$t$ [s]')

t0 = time_array[-1]*0.7
space_t = 1.3*10e3
plt.legend()
plt.tight_layout()

plt.savefig("whittle_likelihood_signal_plot.pdf",
            dpi=600,
            bbox_inches="tight")
plt.show()




# Cut the data
complete_data = DataResidualArray(chans_gap, dt=dt)

freq_threshold = f_min
freqs = complete_data.f_arr
mask = freqs > freq_threshold

data_filtered = complete_data[:,mask]
freq_filtered = freqs[mask]

data = DataResidualArray(data_filtered, f_arr=freq_filtered)

sens_mat = SensitivityMatrix(data.f_arr, [A1TDISens, E1TDISens])



# Define signal (no noise)
def signal_gen(*emri_waveform_args):
    complete_data = DataResidualArray(fastlisaresponse(*emri_waveform_args, emri_waveform_kwargs=emri_waveform_kwargs), dt=dt)

    freqs = complete_data.f_arr
    mask = freqs > freq_threshold

    return DataResidualArray(complete_data[:, mask], f_arr=freqs[mask])


analysis = AnalysisContainer(data, sens_mat, signal_gen=signal_gen)



print(analysis.calculate_signal_likelihood(
    *emri_waveform_args,
    source_only=True
    )
)



# MCMC
# Define likelihood
def wrapper_likelihood(x, fixed_parameters):
    # Only M and mu vary, others are fixed
    params = [x[0], x[1], x[2], x[3], x[4]]  + [fixed_parameters[0]] + [x[5], np.arccos(x[6]), x[7], np.arccos(x[8]), x[9], x[10], x[11], x[12]]
    return analysis.calculate_signal_likelihood(*params, source_only=True)



priors = {'emri': ProbDistContainer({
    0: uniform_dist(9e5, 1.1e6), #M
    1: uniform_dist(9, 11),      #mu
    2: uniform_dist(0.45, 0.55), #a
    3: uniform_dist(8, 8.5),     #p0
    4: uniform_dist(0.45, 0.55), #e0
    5: uniform_dist(0.9, 1.1),   #dist
    6: uniform_dist(-1, 1),      #qS
    7: uniform_dist(0, 2*np.pi), #phiS
    8: uniform_dist(-1, 1),      #qK
    9: uniform_dist(0, 2*np.pi), #phiK 
    10: uniform_dist(0, 2*np.pi),#Phi_phi0 
    11: uniform_dist(0, 2*np.pi),#Phi_r0 
    12: uniform_dist(0, 2*np.pi),#Phi_r0
}, return_gpu=use_gpu)}



fixed_parameters = np.array([
    x0,
])



fname = "whittle_gaps.h5"
if os.path.exists(fname):
    ts = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
    os.rename(fname, fname.replace(".h5", ts + ".h5"))
backend = HDFBackend(fname)


# In[16]:


sampler = EnsembleSampler(
    nwalkers=32,
    ndims={'emri': 13},
    log_like_fn=wrapper_likelihood,
    priors=priors,
    args=(fixed_parameters,),
    branch_names=['emri'],
    tempering_kwargs=dict(ntemps=10),
    backend=HDFBackend(fname),
)


# Starting positions
injection = np.array([M, mu, a, p0, e0, dist, np.cos(qS), phiS, np.cos(qK), phiK, Phi_phi0, Phi_theta0, Phi_r0])
start = injection * (1 + 1e-7 * np.random.randn(10, 32, 1, 13))
start_state = State({'emri': start})


# Run MCMC
start_time = time.time()

sampler.compute_log_prior(start_state.branches_coords)
sampler.run_mcmc(start_state, nsteps=1000, progress=True)

print(f"MCMC completed! It took {time.time()-start_time:.2f} s")




