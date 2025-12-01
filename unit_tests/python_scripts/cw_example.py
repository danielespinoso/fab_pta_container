#!/usr/bin/env python
# coding: utf-8

# ## CW parameter estimation with discovery and numpyro

# Obviously, `discovery.samplers.numpyro` requires `numpyro`.

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

import importlib
importlib.reload(ds.signals)
importlib.reload(ds.pulsar)
importlib.reload(ds.deterministic)
importlib.reload(ds.matrix)
importlib.reload(ds)



# Load EPTA DR2 data

if not os.path.exists("../data/"):
    print("\nError: 'data' folder not found in parent directory!!\n")
    exit(0)
allpsrs = [ds.Pulsar.read_feather(psrfile) for psrfile in sorted(glob.glob('../data/epta_dr2/*.feather'))]

# We'll use only five pulsars for speed.
psrs = allpsrs[0:5]


# Currently `ds.delay_binary` takes `dec` and `inc`,
# but `sindec` and `cosinc` respectively have uniform
# priors and are easier to sample without writing
# special code. So we make a factory to make a JAX
# delay function with those.

# Note also that the CW code is beta quality so
# it is not guaranteed that all the conventions
# are right.

# The standard EPTA model can be created with functions in
# `discovery.models.epta`, but here we use a nanograv-style
# model so we can add the delay. Enable `globalgp` to get HD GW.

import importlib
importlib.reload(ds.signals)
importlib.reload(ds.pulsar)
importlib.reload(ds.deterministic)
importlib.reload(ds.matrix)
importlib.reload(ds.likelihood)
importlib.reload(ds)

fourdelay = ds.makefourier_binary(pulsarterm=True)
cwcommon = ['cw_sindec', 'cw_cosinc', 'cw_log10_f0', 'cw_log10_h0', 'cw_phi_earth', 'cw_psi', 'cw_ra']

T = ds.getspan(psrs)
fml = ds.ArrayLikelihood([ds.PulsarLikelihood([psr.residuals,
                                               ds.makenoise_measurement(psr, noisedict=psr.noisedict),
                                               # ds.makegp_ecorr(psr, noisedict=psr.noisedict),
                                               ds.makegp_timing(psr, svd=True),
                                              ]) for psr in psrs],
                         commongp = ds.makecommongp_fourier(psrs, ds.makepowerlaw_crn(14), 30, T, means=fourdelay,
                                                            common=['crn_gamma', 'crn_log10_A'] + cwcommon, name='rednoise', meansname='cw'))


# printing out some parameters, just to check them
logl = fml.logL
print("parameters collection: ",logl.params)

p0 = ds.sample_uniform(logl.params)
print("logl(p0) : ", logl(p0))


timedelay = ds.makedelay_binary(pulsarterm=True)
fourdelay = ds.makefourier_binary(pulsarterm=True)

cwcommon = ['cw_sindec', 'cw_cosinc', 'cw_log10_f0', 'cw_log10_h0', 'cw_phi_earth', 'cw_psi', 'cw_ra']

T = ds.getspan(psrs)

tml = ds.ArrayLikelihood([ds.PulsarLikelihood([psr.residuals,
                                               ds.makenoise_measurement(psr, noisedict=psr.noisedict),
                                               # ds.makegp_ecorr(psr, noisedict=psr.noisedict),
                                               ds.makegp_timing(psr, svd=True),
                                               ds.makedelay(psr, timedelay, common=cwcommon, name='cw')
                                              ]) for psr in psrs],
                          commongp = ds.makecommongp_fourier(psrs, ds.powerlaw, 30, T, name='rednoise'),
                          globalgp = ds.makeglobalgp_fourier(psrs, ds.powerlaw, ds.hd_orf, 14, T, name='gw'))

fml = ds.ArrayLikelihood([ds.PulsarLikelihood([psr.residuals,
                                               ds.makenoise_measurement(psr, noisedict=psr.noisedict),
                                               # ds.makegp_ecorr(psr, noisedict=psr.noisedict),
                                               ds.makegp_timing(psr, svd=True),
                                              ]) for psr in psrs],
                          commongp = ds.makecommongp_fourier(psrs, ds.powerlaw, 30, T, name='rednoise'),
                          globalgp = ds.makeglobalgp_fourier(psrs, ds.powerlaw, ds.hd_orf, 14, T, means=fourdelay, common=cwcommon, name='gw', meansname='cw'))


# Random parameters.
p0 = ds.sample_uniform(tml.logL.params)

# Sanity check---the likelihood runs fine.
print("Sanity check---the likelihood runs fine : ", tml.logL(p0) )
print("Sanity check---the likelihood runs fine : ", fml.logL(p0) )


# Let's see what the CW is doing in the first pulsar.
pp.plot(psrs[0].toas, fml.globalgp.Fs[0] @ fml.globalgp.means(p0)[:28])
pp.plot(psrs[0].toas, ds.makedelay(psrs[0], timedelay, name='cw', common=cwcommon)(p0))


# Prepare transformed likelihood and model for numpyro.
flogl   = ds_numpyro.makemodel_transformed(fml.logL)
sampler = ds_numpyro.makesampler_nuts(flogl)

# Run!
sampler.run(jax.random.PRNGKey(42))

# Let's see a plot. Not very significant of course
# with few pulsars and no HD correlations.
chain = sampler.to_df()

import corner

fig = corner.corner(chain.values, labels=chain.columns)


