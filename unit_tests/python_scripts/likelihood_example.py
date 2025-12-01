#!/usr/bin/env python
# coding: utf-8

# ## Building basic likelihoods

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
import discovery.models.nanograv as ds_nanograv
import discovery.samplers.numpyro as ds_numpyro


# Read nanograv pulsars
if not os.path.exists("../data/"):
    print("\nError: 'data' folder not found in parent directory!!\n")
    exit(0)
allpsrs = [ds.Pulsar.read_feather(psrfile) for psrfile in sorted(glob.glob('../data/*-[JB]*.feather'))]

psr = allpsrs[0]


#---- This code builds a single pulsar likelihood

# #### Measurement noise only, no backends

m = ds.PulsarLikelihood([psr.residuals,
                         ds.makenoise_measurement_simple(psr)])

# What are the active parameters?
print("Active parameters : ", m.logL.params)

# Sample random values from their priors
p0 = ds.sample_uniform(m.logL.params)
print("Random values sampled from their priors : ", p0)


# Evaluate the likelihood
print("Likelihood evaluated : ",m.logL(p0))


# Try compiled version, grad
print("Compiled version, grad : ", jax.jit(m.logL)(p0), jax.grad(m.logL)(p0))


# #### Measurement noise only, nanograv backends, free parameters
m = ds.PulsarLikelihood([psr.residuals,
                         ds.makenoise_measurement(psr)])

print("Noise-only, Nanograv backends parameters : ", m.logL.params)


# #### Measurement noise only, nanograv backends, parameters from noisedict
m.logL(ds.sample_uniform(m.logL.params))

print("Noise-only, Nanograv backends parameters from noise-dictionary: ", psr.noisedict)

m = ds.PulsarLikelihood([psr.residuals,
                         ds.makenoise_measurement(psr, psr.noisedict)])
m.logL.params
m.logL(ds.sample_uniform(m.logL.params))


# #### Add ECORR noise (GP), free params

m = ds.PulsarLikelihood([psr.residuals,
                         ds.makenoise_measurement(psr),
                         ds.makegp_ecorr(psr)])

print("Added Ecorr noise (GP), parameters : ", m.logL.params)

m.logL(ds.sample_uniform(m.logL.params))


# #### Add ECORR noise (GP), noisedict params

m = ds.PulsarLikelihood([psr.residuals,
                         ds.makenoise_measurement(psr, psr.noisedict),
                         ds.makegp_ecorr(psr, psr.noisedict)])

print("Added Ecorr noise (GP), parameters from noise-dictionary: ", m.logL.params)

m.logL(ds.sample_uniform(m.logL.params))


# #### Add timing model

m = ds.PulsarLikelihood([psr.residuals,
                         ds.makenoise_measurement(psr, psr.noisedict),
                         ds.makegp_ecorr(psr, psr.noisedict),
                         ds.makegp_timing(psr, svd=True)])

print("Added timing model : ",m.logL.params)

m.logL(ds.sample_uniform(m.logL.params))


# #### Add red noise (powerlaw)

m = ds.PulsarLikelihood([psr.residuals,
                         ds.makenoise_measurement(psr, psr.noisedict),
                         ds.makegp_ecorr(psr, psr.noisedict),
                         ds.makegp_timing(psr, svd=True),
                         ds.makegp_fourier(psr, ds.powerlaw, components=30, name='rednoise')])

print("Added red noise : ", m.logL.params)

m.logL(ds.sample_uniform(m.logL.params))


# #### Add red noise (powerlaw, fixed gamma)

m = ds.PulsarLikelihood([psr.residuals,
                         ds.makenoise_measurement(psr, psr.noisedict),
                         ds.makegp_ecorr(psr, psr.noisedict),
                         ds.makegp_timing(psr, svd=True),
                         ds.makegp_fourier(psr, ds.partial(ds.powerlaw, gamma=4.33), components=30, name='rednoise')])

print("Added red noise, powerlaw with fixed gamma : ", m.logL.params)
print(m.logL.params)

m.logL(ds.sample_uniform(m.logL.params))


# #### Add red noise (free spectrum)

m = ds.PulsarLikelihood([psr.residuals,
                         ds.makenoise_measurement(psr, psr.noisedict),
                         ds.makegp_ecorr(psr, psr.noisedict),
                         ds.makegp_timing(psr, svd=True),
                         ds.makegp_fourier(psr, ds.freespectrum, components=30, name='rednoise')])

print("Added red noise, free spectrum : ", m.logL.params)

print("B1855+09_rednoise_log10_rho(30)",m.logL({'B1855+09_rednoise_log10_rho(30)': 1e-6 * np.random.randn(30)}))


# ### Multiple pulsars

psrs = allpsrs[:3]


# #### Combined likelihood

m = ds.ArrayLikelihood([ds.PulsarLikelihood([psr.residuals,
                                             ds.makenoise_measurement(psr, psr.noisedict),
                                             ds.makegp_ecorr(psr, psr.noisedict),
                                             ds.makegp_timing(psr, svd=True),
                                             ds.makegp_fourier(psr, ds.powerlaw, components=30, name='rednoise')])
                        for psr in psrs])

print("Combined likelihood of multiple pulars, parameters : ",m.logL.params)

print(m.logL(ds.sample_uniform(m.logL.params)))


# #### Add common noise

# Indicating parameters under common shares them among pulsars

T = ds.getspan(psrs)

m = ds.ArrayLikelihood([ds.PulsarLikelihood([psr.residuals,
                                             ds.makenoise_measurement(psr, psr.noisedict),
                                             ds.makegp_ecorr(psr, psr.noisedict),
                                             ds.makegp_timing(psr, svd=True),
                                             ds.makegp_fourier(psr, ds.powerlaw, components=30, T=T, name='rednoise'),
                                             ds.makegp_fourier(psr, ds.powerlaw, components=14, T=T, name='crn',
                                                               common=['crn_log10_A', 'crn_gamma'])])
                        for psr in psrs])


print("Common noise added : ", m.logL.params)

p0 = ds.sample_uniform(m.logL.params)
print(m.logL(p0))


# #### Parallelize red components

# Coordinated timespan is required
m = ds.ArrayLikelihood([ds.PulsarLikelihood([psr.residuals,
                                             ds.makenoise_measurement(psr, psr.noisedict),
                                             ds.makegp_ecorr(psr, psr.noisedict),
                                             ds.makegp_timing(psr, svd=True)]) for psr in psrs],
                       commongp = [ds.makecommongp_fourier(psrs, ds.powerlaw, components=30, T=T, name='rednoise'),
                                   ds.makecommongp_fourier(psrs, ds.powerlaw, components=14, T=T, name='crn',
                                                           common=['crn_log10_A', 'crn_gamma'])])

print("Red noise components : ",m.logL.params)
print(m.logL(p0))


# #### Reuse Fourier vectors

# `ds.makepowerlaw_crn` yields the sum of two powerlaws,
# with possibly different number of components.

m = ds.ArrayLikelihood([ds.PulsarLikelihood([psr.residuals,
                                             ds.makenoise_measurement(psr, psr.noisedict),
                                             ds.makegp_ecorr(psr, psr.noisedict),
                                             ds.makegp_timing(psr, svd=True)]) for psr in psrs],
                       commongp = ds.makecommongp_fourier(psrs, ds.makepowerlaw_crn(components=14), components=30, T=T, name='rednoise',
                                                          common=['crn_log10_A', 'crn_gamma']))

print("Red noise components re-using Fourier vectors : ",m.logL.params)
print(m.logL(p0))


# #### Add global spatially correlated process

# Note `ds.makeglobalgp_fourier` requires the ORF, but
# not the `common` specification, which is automatic.

m = ds.ArrayLikelihood([ds.PulsarLikelihood([psr.residuals,
                                             ds.makenoise_measurement(psr, psr.noisedict),
                                             ds.makegp_ecorr(psr, psr.noisedict),
                                             ds.makegp_timing(psr, svd=True)]) for psr in psrs],
                       commongp = ds.makecommongp_fourier(psrs, ds.powerlaw, components=30, T=T, name='rednoise'),
                       globalgp = ds.makeglobalgp_fourier(psrs, ds.powerlaw, ds.hd_orf, components=14, T=T, name='gw'))

print("Added global spatially correlated processes : ",m.logL.params)
p0 = ds.sample_uniform(m.logL.params)
print(m.logL(p0))


# #### Another way of doing this (useful if variable GPs differ among pulsars)

m = ds.GlobalLikelihood([ds.PulsarLikelihood([psr.residuals,
                                             ds.makenoise_measurement(psr, psr.noisedict),
                                             ds.makegp_ecorr(psr, psr.noisedict),
                                             ds.makegp_timing(psr, svd=True),
                                             ds.makegp_fourier(psr, ds.powerlaw, components=30, name='rednoise')]) for psr in psrs],
                        globalgp = ds.makeglobalgp_fourier(psrs, ds.powerlaw, ds.hd_orf, components=14, T=T, name='gw'))


print(m.logL.params)
print(m.logL(p0))

