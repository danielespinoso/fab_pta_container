#!/usr/bin/env python
# coding: utf-8

# ## Optimal statistic with discovery

import sys
import os
import glob

import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as pp

import jax
jax.config.update('jax_enable_x64', True)

import jax.random
import jax.numpy as jnp

import discovery as ds


# Read nanograv pulsars
if not os.path.exists("../data/"):
    print("\nError: 'data' folder not found in parent directory!!\n")
    exit(0)
allpsrs = [ds.Pulsar.read_feather(psrfile) for psrfile in sorted(glob.glob('../data/*-[JB]*.feather'))]

# Run with fewer pulsars
psrs = allpsrs[:5]


# Set up GlobalLikelihood object. The GP named 'gw' will be used to build
# the optimal statistic; everything else will be included in individual
# pulsar noise. This GW object should be identical for every pulsar,
# and have common parameters only.

Tspan = ds.getspan(psrs)
t0 = ds.getstart(psrs)

gbl = ds.GlobalLikelihood([ds.PulsarLikelihood([psr.residuals,
                                                ds.makenoise_measurement(psr, psr.noisedict),
                                                ds.makegp_ecorr(psr, psr.noisedict),
                                                ds.makegp_timing(psr, svd=True, variable=True),
                                                ds.makegp_fftcov(psr, ds.powerlaw, 61, oversample=6, T=Tspan, name='red_noise'),
                                                ds.makegp_fftcov(psr, ds.brokenpowerlaw, 61, oversample=6, T=Tspan, t0=t0,
                                                                 common=['gw_log10_A', 'gw_gamma', 'gw_log10_fb'], name='gw')
                                                ]) for psr in psrs])


# ### Basic OS

# Make OS object

# Get a random set of parameters, force gamma to 13/3
p0 = ds.sample_uniform(os.params, priordict={'gw_(.*_)?gamma': [13/3,13/3]})


# Compute the OS for the data. You get also the corresponding sigma, SNR, and the amplitude of the GW.
print(os.os(p0))


# ### Alternative ORFs

# The ORF takes only one parameter (the product `z = dot(pos1, pos2)`).
# Discovery predefines `hd_orfa`, `monopole_orfa`, `dipole_orfa`.
print(os.os(p0, ds.monopole_orfa))
print(os.os(p0, ds.dipole_orfa))


# ### "Marginalized" OS 

# Create a population of parameters (normally we'd get them from an MCMC run).
p0s = ds.sample_uniform(os.params, priordict={'gw_(.*_)?gamma': [13/3,13/3]}, n=5)

print(p0s['B1855+09_red_noise_gamma'])


# Then we `jax.vmap` over parameter sets

os_vpar = jax.vmap(os.os)
print("Jax was run on the parameter set. Results : ",os_vpar(p0s))


# ### Scrambles

# `os.scramble` takes as a second argument an array (or list)
# of pulsar positions of dimension `(npsr, 3)`
print("We 'scrambled' the pulsars positions : ",jnp.array(os.pos))
print(os.scramble(p0, jnp.array(os.pos)))


# To build a background you'll want a random array of positions
# of dimension `(nscramble, npsr, 3)`
key = ds.matrix.jnpkey(42)
rpos = jax.random.normal(key, (15, len(psrs), 3))
npos = rpos / jnp.linalg.norm(rpos, axis=2)[:,:,None] # normalize vectors

# Then we `jax.vmap` over positions (the second argument).
scramble_vpos = jax.vmap(os.scramble, (None,0))
print(scramble_vpos(p0, npos))


# ### Phase shifts

# These require a diagonal Fourier GW object.
gbl = ds.GlobalLikelihood([ds.PulsarLikelihood([psr.residuals,
                                                ds.makenoise_measurement(psr, psr.noisedict),
                                                ds.makegp_ecorr(psr, psr.noisedict),
                                                ds.makegp_timing(psr, svd=True),
                                                ds.makegp_fourier(psr, ds.brokenpowerlaw, 30, T=Tspan, name='red_noise'),
                                                ds.makegp_fourier(psr, ds.brokenpowerlaw, 14, T=Tspan,
                                                                  common=['gw_log10_A', 'gw_gamma', 'gw_log10_fb'], name='gw')
                                                ]) for psr in psrs])

os = ds.OS(gbl)
print("Computing phase shifts now... Parameters : ",os.os(p0))


# `os.shift` takes as a second argument an array (or list) of pulsar positions of dimension `(npsr, nfreq)`.

# Sanity check...

zero = jnp.zeros((len(psrs),14))
print("Sanity check : ",os.shift(p0, zero))

key = ds.matrix.jnpkey(42)
phases = 2.0 * jnp.pi * jax.random.uniform(key, shape=(20,10,14))


# Then we `jax.vmap` over phases (the second argument).

shift_vphase = jax.vmap(os.shift, (None,0))

print("Jax finished the computation of phase shifts : ",shift_vphase(p0, phases))


# ### Plotting correlation coefficients

# Build a NANOGrav 15-yr model using all the pulsars.

import importlib
import discovery.ostat as ostat
importlib.reload(ds.signals)
importlib.reload(ds)
importlib.reload(ostat)

Tspan = ds.getspan(allpsrs)

gbl = ds.GlobalLikelihood([ds.PulsarLikelihood([psr.residuals,
                                                ds.makenoise_measurement(psr, psr.noisedict),
                                                ds.makegp_ecorr(psr, psr.noisedict),
                                                ds.makegp_timing(psr, svd=True),
                                                ds.makegp_fourier(psr, ds.powerlaw, 30, T=Tspan, name='red_noise'),
                                                ds.makegp_fourier(psr, ds.powerlaw, 14, T=Tspan,
                                                                  common=['gw_log10_A', 'gw_gamma'], name='gw')
                                                ]) for psr in allpsrs])


# Get the MAP parameter set from an actual MCMC run on 15-yr NANOGrav data
if not os.path.exists("../data/"):
    print("\nError: 'data' folder not found in parent directory!!\n")
    exit(0)
df = pd.read_feather('../data/NG15yr-m2a-chain.feather')

p0 = df.iloc[df['logp'].argmax()].to_dict()


# Sanity check
os = ds.OS(gbl)
print("Sanity check : ",os.os(p0))


# Simple minded correlation plot: get angles and bin the data
def oscorr(os, p0, nbins=10):
    iota = np.arccos(jnp.array(os.angles)) * (180 / np.pi)

    bins = np.linspace(0, 180, nbins + 1)
    indices = np.digitize(iota, bins) - 1
    masks = [np.where(indices == i)[0] for i in range(nbins)]

    # orfs = ds.hd_orfa(np.array(os.angles))
    orfs = np.ones_like(os.angles)

    rhos, sigmas = os.os_rhosigma(p0)
    gwnorm = 10**(2.0 * p0[os.gwpar])
    rhos, sigmas = gwnorm * rhos, gwnorm * sigmas
    
    iotas = [np.mean(iota[mask]) for mask in masks]
    oses = [np.sum(rhos[mask] * orfs[mask] / sigmas[mask]**2) / np.sum(orfs[mask]**2 / sigmas[mask]**2)
            for mask in masks]
    osigs = [1 / np.sqrt(np.sum(orfs[mask]**2 / sigmas[mask]**2))
           for mask in masks]

    return iotas, oses, osigs

iotas, oses, osigs = oscorr(os, p0)

pp.errorbar(iotas, oses, yerr=osigs, fmt='.')

a = np.linspace(1e-6, 180)
hd = ost.os(p0)['os'] * ds.hd_orfa(np.cos(a * (np.pi/180.0)))

pp.plot(a, hd)
pp.xlabel('iota')
pp.ylabel(r'$A_\mathrm{gw}^2$')


# Plot the distribution of SNRs on a subchain

chain = df.sample(1000).astype(np.float64)
p0s = {var: jnp.array(chain[var]) for var in df.columns}

vos = jax.jit(jax.vmap(os.os))

pp.hist(oses['snr'], histtype='step', bins=20, density=True, label='fourier')
# pp.hist(ostes['snr'], histtype='step', bins=20, density=True, label='fftcov')
pp.axvline(np.mean(oses['snr']), color='C0', ls=':')
# pp.axvline(np.mean(ostes['snr']), color='C1', ls=':')
pp.legend()


# ### Rapid sim

import importlib
import discovery.os as ostat

importlib.reload(ostat)
Tspan = ds.getspan(allpsrs[:4])

sim = ds.GlobalLikelihood([ds.PulsarLikelihood([ds.makedelay(psr, ds.getresiduals, name='sim'), 
                                                ds.makenoise_measurement(psr, psr.noisedict),
                                                # ds.makegp_ecorr(psr, psr.noisedict), # cannot do fixed GP at this time 
                                                ds.makegp_timing(psr, variance=1e-12, variable=True),
                                                ds.makegp_fourier(psr, ds.powerlaw, 30, T=Tspan, name='red_noise'),
                                                ds.makegp_fourier(psr, ds.powerlaw, 14, T=Tspan,
                                                                  common=['gw_log10_A', 'gw_gamma'], name='gw')
                                                ]) for psr in allpsrs[:4]])

os = ds.os.OS(sim)

df = pd.read_feather('../data/NG15yr-m2a-chain.feather')
p0 = df.iloc[df['logp'].argmax()].to_dict()
print("Doing a rapid SIM : ", (os.os({**p0, **{f'{psr.name}_sim_y': psr.residuals for psr in allpsrs[:4]}})))

oss = os.sample_rhosigma(p0)

voss = jax.jit(jax.vmap(oss))

n = 10000

snrs = []
for i in range(10):
 snrs.append(voss(ds.matrix.jnpnormal(key, (n, oss.cnt))).block_until_ready())
snrs = jnp.concatenate(snrs)

joss, doss, voss = jax.jit(oss), jax.jit(jax.grad(oss)), jax.jit(jax.vmap(oss))
Q = lambda x: 0.5 * doss(x)

x = ds.matrix.jnpnormal(ds.matrix.jnpkey(42), oss.cnt)
print(joss(x))
print(x.T @ Q(x))

# Estimate $\lambda_\mathrm{max}$

def estimate_spectral_norm(Q, n, num_iters=20, seed=42):
    v = jax.random.normal(ds.matrix.jnpkey(seed), n)
    
    v = v / jnp.linalg.norm(v)
    for _ in range(num_iters):
        v = Q(v)
        norm_v = jnp.linalg.norm(v)
        if norm_v == 0:
            return 0.0
        v = v / norm_v

    return jnp.abs(jnp.dot(v, Q(v)))

Qnorm = estimate_spectral_norm(Q, oss.cnt, 40)

# This means the maximum $\theta$ is $1 / (2 \lambda_\mathrm{max})$.
Qnorm, 1 / (2 * Qnorm)


# Solve $\mathrm{tr} (Q (I - 2 \theta Q)^{-1}) = s$, where $s$ is the desired tail value.
def rademacher(key, shape):
    return 2.0 * jax.random.randint(key, shape=shape, minval=0, maxval=2) - 1.0

def get_trace(theta, Q, n, m=20, seed=42):
    keys = ds.matrix.jnpsplit(jax.random.PRNGKey(seed), m)
    
    matvec = lambda v: v - 2 * theta * Q(v)

    return np.mean([jnp.dot(z, Q(jax.scipy.sparse.linalg.cg(matvec, z)[0]))
                    for z in rademacher(jax.random.PRNGKey(seed), (m, n))])

theta0 = 1.6

tr0 = get_trace(theta0, Q, oss.cnt, 42)
print("Trace : ",tr0)


# Sample from tilted distribution: $x \sim \mathcal{N}(0, I)$, solve $(I - 2 \theta Q) y = x$.
def sample_sigma(theta, Q, n, m=1, seed=42):
    x = ds.matrix.jnpnormal(jax.random.PRNGKey(seed), (m, oss.cnt))
    matvec = lambda v: v - 2 * theta * Q(v)
    
    return jax.vmap(lambda v: jax.scipy.sparse.linalg.cg(matvec, v)[0])(x)

# Evaluate OS and weights $\exp -\theta x^T Q x$ for these samples. Note weights should be normalized by $\sqrt{\Sigma}$ with $\Sigma = (I - 2\theta Q)^{-1}$.
o = voss(y)
w = jnp.exp(-theta0 * o)

pp.hist(o, weights=w, bins=40)

def makecdf(array, weights=None, xmin=None, xcdf=0):
    if weights is None:
        return np.sort(array), np.arange(1, len(array) + 1) / len(array)
    else:
        if xmin is not None:
            weights = weights[array > xmin]
            array = array[array > xmin]
        
        sort_idx = np.argsort(array)
        return array[sort_idx], xcdf + (1 - xcdf) * np.cumsum(weights[sort_idx]) / np.sum(weights)

oval, cval = makecdf(o, w)

pp.figure(figsize=(4,3))
pp.semilogy(oval, 1 - cval)

ooval, ccval = makecdf(oo)


# Patch the two distributions together.
omatch = 1 * tr0
offset = np.interp(omatch, ooval, ccval) 
oval, cval = makecdf(o, w, omatch, offset)

pp.figure(figsize=(4,3))
pp.semilogy(ooval, 1 - ccval)
pp.semilogy(oval, 1 - cval)
pp.axvline(omatch, ls=':')
pp.axhline(1 - offset, ls=':')
pp.axis(xmin=0, xmax=15, ymin=1e-8, ymax=1e-1)

Qnorm = estimate_spectral_norm(matvec, oss.cnt, 100)

def estimate_frobenius_norm(Q_func, n, num_samples=50, seed=42):
    trace_estimate = 0.0

    for i in range(num_samples):
        z = 2.0 * jax.random.randint(ds.matrix.jnpkey(seed + i), shape=(n,), minval=0, maxval=2) - 1.0
        Qz = Q_func(z)
        trace_estimate += jnp.dot(Qz, Qz)

    trace_estimate /= num_samples
    frobenius_norm = jnp.sqrt(trace_estimate)
    return frobenius_norm

QFnorm = estimate_frobenius_norm(matvec, oss.cnt, 100, 42)
survival1 = 2 * jnp.exp(-(1/8) * s**2 / QFnorm**2)
survival2 = 2 * jnp.exp(-(1/8) * s / Qnorm)

s, c = makecdf(snrs)

pp.semilogy(s, 1 - c)

pp.semilogy(s, survival)
pp.semilogy(s, survival2)

pp.axis(xmin=2, xmax=5, ymin=1e-4, ymax=1e-1)


# ### Tilted tail sampling

def matvec(v):
    return v - 2 * theta * 0.5 * doss(v)

def rademacher(key, n):
    return 2.0 * jax.random.randint(key, shape=(n,), minval=0, maxval=2) - 1.0

def get_trace(theta, m=20, key=jax.random.PRNGKey(42)):
    trace_est = 0.0
    for _ in range(m):
        key, subkey = ds.matrix.jnpsplit(key)
        z = rademacher(subkey, oss.cnt)
        trace_est += np.dot(z, doss(jax.scipy.sparse.linalg.cg(lambda v: v - theta * doss(v), z)[0]))
    return trace_est / m

print("Tilted tail sampling - Trace 0.1 : ", get_trace(0.1, key=jax.random.PRNGKey(43)))
print("Tilted tail sampling - Trace 0.2 : ", get_trace(0.2, key=jax.random.PRNGKey(43)))
print("Tilted tail sampling - Trace 0.3 : ", get_trace(0.3, key=jax.random.PRNGKey(43)))
print("Tilted tail sampling - Trace 1.5 : ", get_trace(1.5, key=jax.random.PRNGKey(43)))

def sample_sigma(theta, key=jax.random.PRNGKey(42)):
    key, subkey = ds.matrix.jnpsplit(key)
    x = ds.matrix.jnpnormal(subkey, oss.cnt)
    return key, jax.scipy.sparse.linalg.cg(lambda v: v - theta * doss(v), x)[0]

vo, vw = [], []
for i in tqdm.tqdm(range(1000)):
    key, x1 = sample_sigma(1.5, key)
    o = oss(x1)
    w = jnp.exp(-1.5 * o)
    vo.append(o)
    vw.append(w)

def lanczos(matvec, v, m):
    """
    Run m steps of Lanczos on (I - 2*theta*Q) using starting vector v.
    Returns:
        - alphas: diagonal entries of T
        - betas: off-diagonal entries of T
    """
    n = v.shape[0]
    
    beta_prev = 0.0
    q_prev = jnp.zeros_like(v)
    q = v / jnp.linalg.norm(v)
    
    alphas = []
    betas = []
    
    for j in range(m):
        w = matvec(q) - beta_prev * q_prev
        alpha = jnp.dot(q, w)
        w = w - alpha * q
        beta = jnp.linalg.norm(w)
        
        alphas.append(alpha)
        betas.append(beta)
        
        q_prev = q
        q = jnp.where(beta > 1e-10, w / beta, q)   # avoid divide by zero
        beta_prev = beta
    
    betas = jnp.array(betas[:-1])  # last beta unused
    alphas = jnp.array(alphas)
    return alphas, betas


def log_trace_from_tridiagonal(alphas, betas):
    """
    Compute e1^T log(T) e1 from T defined by alphas, betas.
    """
    T = jnp.diag(alphas)
    if betas.shape[0] > 0:
        T = T.at[jnp.arange(len(betas)), jnp.arange(1, len(betas)+1)].set(betas)
        T = T.at[jnp.arange(1, len(betas)+1), jnp.arange(len(betas))].set(betas)

    eigvals, eigvecs = jnp.linalg.eigh(T)
    log_eigs = jnp.log(eigvals)
    weights = eigvecs[0, :] ** 2
    return jnp.sum(weights * log_eigs)

def estimate_logdet(matvec, n, num_samples=30, lanczos_steps=10, key=jax.random.PRNGKey(42)):
    keys = jax.random.split(key, num_samples)
    log_traces = []
    
    for k in keys:
        v = jax.random.choice(k, jnp.array([-1.0, 1.0]), shape=(n,))
        v = v / jnp.linalg.norm(v)
        
        alphas, betas = lanczos(matvec, v, lanczos_steps)
        log_trace = log_trace_from_tridiagonal(alphas, betas)
        log_traces.append(log_trace)
    
    logdet_est = jnp.mean(jnp.stack(log_traces))
    return logdet_est

print("Logarithm of Determinant : ",estimate_logdet(matvec, oss.cnt, num_samples=100, lanczos_steps=6))

xs = ds.matrix.jnpnormal(key, oss.cnt)
print(0.5 * xs.T @ jax.grad(oss)(xs))

print(oss(xs))

print("Matrix : ",oss(ds.matrix.jnpnormal(key, (100, oss.cnt))))

Tspan = ds.getspan(allpsrs)

sim = ds.GlobalLikelihood([ds.PulsarLikelihood([ds.makedelay(psr, ds.getresiduals, name='sim'), 
                                                ds.makenoise_measurement(psr, psr.noisedict),
                                                # ds.makegp_ecorr(psr, psr.noisedict), # cannot do fixed GP at this time 
                                                ds.makegp_timing(psr, variance=1e-12, variable=True),
                                                ds.makegp_fourier(psr, ds.powerlaw, 30, T=Tspan, name='red_noise'),
                                                ds.makegp_fourier(psr, ds.powerlaw, 14, T=Tspan,
                                                                  common=['gw_log10_A', 'gw_gamma'], name='gw')
                                                ]) for psr in allpsrs])


os = ds.os.OS(sim)

df = pd.read_feather('../data/NG15yr-m2a-chain.feather')
p0 = df.iloc[df['logp'].argmax()].to_dict()

os.os({**p0, **{f'{psr.name}_sim_y': psr.residuals for psr in allpsrs}})

key = ds.jnpkey(42)

def makecdf(array):
    return np.sort(array), np.arange(1, len(array) + 1) / len(array)

s, c = makecdf(snrs - jnp.mean(snrs))
pp.semilogy(s, 1 - c, lw=4, alpha=0.5)

mask = (s > 3) & (s < 5)
pp.plot(s, jnp.exp(jnp.polyval(jnp.polyfit(s[mask], jnp.log(1 - c[mask]), 2), s)))

mask = (s > 4) & (s < 5)
pp.plot(s, jnp.exp(jnp.polyval(jnp.polyfit(s[mask], jnp.log(1 - c[mask]), 1), s)))

pp.axis(xmin=1, xmax=6, ymin=1e-5, ymax=1)

s, c = makecdf(lsnrs - jnp.mean(lsnrs))
pp.semilogy(s, 1 - c, lw=4, alpha=0.5)

mask = (s > 3) & (s < 5)
pp.plot(s, jnp.exp(jnp.polyval(jnp.polyfit(s[mask], jnp.log(1 - c[mask]), 2), s)))

mask = (s > 4) & (s < 5)
pp.plot(s, jnp.exp(jnp.polyval(jnp.polyfit(s[mask], jnp.log(1 - c[mask]), 1), s)))

pp.axis(xmin=1, xmax=6, ymin=1e-5, ymax=1)

jsample, jos = jax.jit(sim.sample), jax.jit(os.os)

key = ds.matrix.jnpkey(42)

oslist = []
for i in tqdm.tqdm(range(1000)):
    key, ys = jsample(key, p0)
    oslist.append(jos({**p0, **{f'{psr.name}_sim_y': y for psr, y in zip(allpsrs, ys)}}))
ssnrs = np.array([o['snr'] for o in oslist])

# The tail goes from Gaussian to subexponential. The bound is 2 exp(-min(t^2/|Q_F|^2, t/|Q|)).

s, c = makecdf(snrs - jnp.mean(snrs))
pp.semilogy(s, 1 - c, lw=4, alpha=0.5)

mask = (s > 0) & (s < 4)
pp.plot(s[mask], jnp.exp(jnp.polyval(jnp.polyfit(s[mask], jnp.log(1 - c[mask]), 2), s[mask])))

mask = (s > 2) & (s < 4.5)
pp.plot(s[mask], jnp.exp(jnp.polyval(jnp.polyfit(s[mask], jnp.log(1 - c[mask]), 1), s[mask])))

pp.plot(*makecdf(snrs), label='direct')
pp.plot(*makecdf(ssnrs),label='sims')
pp.axis(xmin=1, xmax=3, ymin=0.8)
pp.legend()

pp.hist(snrs, histtype='step', density=True, bins=30)
pp.hist(ssnrs, histtype='step', density=True, bins=30)


# ### GX2 distribution

import importlib

importlib.reload(ds.os)
importlib.reload(ds.matrix)
importlib.reload(ds.signals)
importlib.reload(ds.likelihood)
importlib.reload(ds)

import discovery.ostat as ostat
importlib.reload(ostat)


# We will compute the distribution of OS over a set of simulated datasets, and compare with the GP-only GX2 distribution.

Tspan = ds.getspan(allpsrs)

sim = ds.GlobalLikelihood([ds.PulsarLikelihood([ds.makedelay(psr, ds.getresiduals, name='sim'), 
                                                ds.makenoise_measurement(psr, psr.noisedict),
                                                ds.makegp_ecorr(psr, psr.noisedict),
                                                ds.makegp_timing(psr, variance=1e-12),
                                                ds.makegp_fourier(psr, ds.powerlaw, 30, T=Tspan, name='red_noise'),
                                                ds.makegp_fourier(psr, ds.powerlaw, 14, T=Tspan,
                                                                  common=['gw_log10_A', 'gw_gamma'], name='gw')
                                                ]) for psr in allpsrs])

df = pd.read_feather('../data/NG15yr-m2a-chain.feather')
p0 = df.iloc[df['logp'].argmax()].to_dict()

os = ostat.OS(sim)


# Evaluate the OS for the dataset residuals.
print("OS parameters : ",os.os({**p0, **{f'{psr.name}_sim_y': psr.residuals for psr in allpsrs}}))

key = ds.matrix.jnpkey(42)

oslist = []
for i in tqdm.tqdm(range(1000)):
    key, ys = sim.sample(key, p0)
    oslist.append(os.os({**p0, **{f'{psr.name}_sim_y': y for psr, y in zip(allpsrs, ys)}}))

snrs = np.array([o['snr'] for o in oslist])

print(os.gx2cdf(p0, [1.5], cutoff=1e-6, limit=200, epsabs=1e-6))

print(os.gx2cdf(p0, [1.5], cutoff=1e-8, limit=200, epsabs=1e-6))

pp.hist(snrs, histtype='step', bins=20, density=True)
pp.plot(0.5*(xs[1:] + xs[:-1]), np.diff(cs) / (xs[1] - xs[0]))

ssnrs = np.sort(snrs)
cdf = np.arange(1, len(ssnrs) + 1) / len(ssnrs)
pp.plot(ssnrs, cdf)

# pp.plot(xs, cs)
pp.plot(1.25 * xs, cs)

Tspan = ds.getspan(allpsrs)

sim2 = ds.GlobalLikelihood([ds.PulsarLikelihood([ds.makedelay(psr, ds.getresiduals, name='sim'), 
                                                 ds.makenoise_measurement(psr, psr.noisedict),
                                                 ds.makegp_fourier(psr, ds.powerlaw, 30, T=Tspan, name='red_noise'),
                                                 ds.makegp_fourier(psr, ds.powerlaw, 14, T=Tspan,
                                                                   common=['gw_log10_A', 'gw_gamma'], name='gw')
                                                 ]) for psr in allpsrs])


print("SIM2 : ",sim2.psls[0].N.N.make_sqrt()({}))

os2 = ds.OS(sim2)

print("Params : ", os2.os({**p0, **{f'{psr.name}_sim_y': psr.residuals for psr in allpsrs}}))

jsim, jos = jax.jit(sim2.sample), jax.jit(os2.os)

key = ds.matrix.jnpkey(42)

oslist2 = []
for i in tqdm.tqdm(range(10000)):
    key, ys = jsim(key, p0)
    oslist2.append(jos({**p0, **{f'{psr.name}_sim_y': y for psr, y in zip(allpsrs, ys)}}))
snrs2 = np.array([o['snr'] for o in oslist2])

# Try a direct calculation

def makecdf(array):
    return np.sort(array), np.arange(1, len(array) + 1) / len(array)


pp.plot(*makecdf(snrs2), label='sim')
pp.plot(*makecdf(dsnr), label='direct gx2')
pp.plot(xs2, cs2, label='imhof gx2')
pp.legend()

pp.plot(*makecdf(snrs2), label='sim')
pp.plot(*makecdf(dsnr), label='direct gx2')
pp.plot(xs2, cs2, label='imhof gx2')
pp.axis(xmin=2, ymin=0.96, ymax=1.02)
pp.legend()

pp.hist(snrs2, histtype='step', bins=40, density=True, label='sim')
pp.hist(dsnr, histtype='step', bins=40, density=True, label='direct gx2')
pp.plot(0.5*(xs2[1:] + xs2[:-1]), np.diff(cs2) / np.diff(xs2)[0], label='imhof gx2')
pp.legend()

pp.plot(*makecdf(snrs2), label='sim')
pp.plot(*makecdf(dsnr), label='direct gx2')
pp.plot(xs2, cs2, label='imhof gx2')
pp.legend()

def lownoise(noisedict):
    return {par: (val / 10) if 'efac' in par else val - 1 if 'equad' in par else val 
            for par, val in noisedict.items()}

sim2lw = ds.GlobalLikelihood([ds.PulsarLikelihood([ds.makedelay(psr, ds.getresiduals, name='sim'), 
                                                   ds.makenoise_measurement(psr, lownoise(psr.noisedict)),
                                                   ds.makegp_fourier(psr, ds.powerlaw, 30, T=Tspan, name='red_noise'),
                                                   ds.makegp_fourier(psr, ds.powerlaw, 14, T=Tspan,
                                                                     common=['gw_log10_A', 'gw_gamma'], name='gw')
                                                   ]) for psr in allpsrs])

os2lw = ds.OS(sim2lw)

joslw, jsimlw = jax.jit(os2lw.os), jax.jit(sim2lw.sample)

key = ds.matrix.jnpkey(42)

oslist2lw = []
for i in tqdm.tqdm(range(10000)):
    key, ys = jsimlw(key, p0)
    oslist2lw.append(joslw({**p0, **{f'{psr.name}_sim_y': y for psr, y in zip(allpsrs, ys)}}))
snrs2lw = np.array([o['snr'] for o in oslist2lw])

pp.figure(figsize=(12,4))

pp.subplot(1,2,1)
pp.plot(*makecdf(snrs2), label='sim')
pp.plot(*makecdf(dsnr), label='direct gx2')
pp.legend()

pp.subplot(1,2,2)
pp.plot(*makecdf(snrs2lw), label='sim LW')
pp.plot(*makecdf(dsnrlw), label='direct gx2 LW')
pp.legend()

pp.figure(figsize=(12,4))

pp.subplot(1,2,1)
pp.plot(*makecdf(snrs2), label='sim')
pp.plot(xs2, cs2, label='imhof gx2')
pp.plot(*makecdf(dsnr), label='direct gx2')
pp.axis(xmin=2, ymin=0.96, ymax=1.02)
pp.legend()

pp.subplot(1,2,2)
pp.plot(*makecdf(snrs2lw), label='sim LN')
pp.plot(xs2lw, cs2lw, label='imhof gx2 LN')
pp.plot(*makecdf(dsnrlw), label='direct gx2 LN')
pp.axis(xmin=2, ymin=0.96, ymax=1.02)
pp.legend()

def lownoise(noisedict, factor=10):
    return {par: (val / factor) if 'efac' in par else val - np.log10(factor) if 'equad' in par else val 
            for par, val in noisedict.items()}

sim2vlw = ds.GlobalLikelihood([ds.PulsarLikelihood([ds.makedelay(psr, ds.getresiduals, name='sim'), 
                                                    ds.makenoise_measurement(psr, lownoise(psr.noisedict, 100)),
                                                    ds.makegp_fourier(psr, ds.powerlaw, 30, T=Tspan, name='red_noise'),
                                                    ds.makegp_fourier(psr, ds.powerlaw, 14, T=Tspan,
                                                                      common=['gw_log10_A', 'gw_gamma'], name='gw')
                                                    ]) for psr in allpsrs])

os2vlw = ds.OS(sim2vlw)

josvlw, jsimvlw = jax.jit(os2vlw.os), jax.jit(sim2vlw.sample)

key = ds.matrix.jnpkey(42)

oslist2vlw = []
for i in tqdm.tqdm(range(10000)):
    key, ys = jsimvlw(key, p0)
    oslist2vlw.append(josvlw({**p0, **{f'{psr.name}_sim_y': y for psr, y in zip(allpsrs, ys)}}))
snrs2vlw = np.array([o['snr'] for o in oslist2vlw])


pp.figure(figsize=(12,3))

pp.subplot(1,3,1)
pp.plot(*makecdf(snrs2), label='sim')
pp.plot(xs2, cs2, label='imhof gx2')
pp.legend()

pp.subplot(1,3,2)
pp.plot(*makecdf(snrs2lw), label='sim LN')
pp.plot(xs2lw, cs2lw, label='imhof gx2 LN')
pp.legend()

pp.subplot(1,3,3)
pp.plot(*makecdf(snrs2vlw), label='sim VLN')
pp.plot(xs2vlw, cs2vlw, label='imhof gx2 VLN')
pp.legend()


pp.figure(figsize=(12,3))

pp.subplot(1,3,1)
pp.plot(*makecdf(snrs2), label='sim')
pp.plot(xs2, cs2, label='imhof gx2')
pp.plot(*makecdf(dsnr), label='direct gx2')
pp.axis(xmin=2, ymin=0.97, ymax=1.005)
pp.legend()

pp.subplot(1,3,2)
pp.plot(*makecdf(snrs2lw), label='sim LN')
pp.plot(xs2lw, cs2lw, label='imhof gx2 LN')
pp.plot(*makecdf(dsnrlw), label='direct gx2 LN')
pp.axis(xmin=2, ymin=0.97, ymax=1.005)
pp.legend()

pp.subplot(1,3,3)
pp.plot(*makecdf(snrs2vlw), label='sim VLN')
pp.plot(xs2vlw, cs2vlw, label='imhof gx2 VLN')
pp.plot(*makecdf(dsnrvlw), label='direct gx2 VLN')
pp.axis(xmin=2, ymin=0.97, ymax=1.005)
pp.legend()

pp.tight_layout()

def matvec(x):
    return amat @ x

def Lanczos( A, v, m=100 ):
    n = len(v)
    if m>n: m = n
    # from here https://en.wikipedia.org/wiki/Lanczos_algorithm
    V = np.zeros( (m,n) )
    T = np.zeros( (m,m) )
    vo   = np.zeros(n)
    beta = 0
    for j in range( m-1 ):
        w    = np.dot( A, v )
        alfa = np.dot( w, v )
        w    = w - alfa * v - beta * vo
        beta = np.sqrt( np.dot( w, w ) ) 
        vo   = v
        v    = w / beta 
        T[j,j  ] = alfa 
        T[j,j+1] = beta
        T[j+1,j] = beta
        V[j,:]   = v
    w    = np.dot( A,  v )
    alfa = np.dot( w, v )
    w    = w - alfa * v - beta * vo
    T[m-1,m-1] = np.dot( w, v )
    V[m-1]     = w / np.sqrt( np.dot( w, w ) ) 
    return T, V

def estimate_eigenvalues(A, m=30, num_probes=20, seed=0):
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    all_eigs = []

    for _ in range(num_probes):
        v = rng.normal(size=n)
        v = v / np.linalg.norm(v)
        T, _ = Lanczos(A, v, m)
        eigs = np.linalg.eigvalsh(T)
        all_eigs.append(eigs)

    return np.array(all_eigs)  # shape: (num_probes, m)

pp.plot(np.sort(np.mean(estimate_eigenvalues(amat, m=60, num_probes=200), axis=0))[::-1], 'x')
pp.plot(np.sort(np.abs(esA))[::-1][:len(esT)])

v0 = np.random.rand(amat.shape[0])
v0 /= np.sqrt(np.dot(v0,v0))

T, V = Lanczos(amat, v0, m=35)
esT, vsT = np.linalg.eig(T)

esA, vsA = np.linalg.eig(amat)

pp.plot(np.sort(np.abs(esT))[::-1], 'x')
pp.plot(np.sort(np.abs(esA))[::-1][:len(esT)])
