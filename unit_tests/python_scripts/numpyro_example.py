#!/usr/bin/env python
# coding: utf-8

# ## Parameter estimation with discovery and numpyro

# Creating MCMC chains with discovery likelihoods and `numpyro`'s [NUTS sampler](https://num.pyro.ai/en/latest/mcmc.html).


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

# Obviously, `discovery.samplers.numpyro` requires `numpyro`.

import discovery as ds
import discovery.models.nanograv as ds_nanograv
import discovery.samplers.numpyro as ds_numpyro

if not os.path.exists("../data/"):
    print("\nError: 'data' folder not found in parent directory!!\n")
    exit(0)

allpsrs = [ds.Pulsar.read_feather(psrfile) for psrfile in sorted(glob.glob('../data/*-[JB]*.feather'))]


# Run with fewer pulsars to converge on GPU
allpsrs = allpsrs[:5]


# Set up a NANOGrav CURN model. Possible options to `makemodel_curn` are:
#   `rn_components`
#   `crn_components`
#   `gamma=<fixed_value>`
# For an HD model, use: `ds_nanograv.makemodel_hd`.

model = ds_nanograv.makemodel_curn(allpsrs)

# Obtain the likelihood
logl = model.logL


# These are the parameters.
print("These are the likelihood parameters : ",logl.params)


# Sample parameter values from their default uniform priors.
p0 = ds.sample_uniform(logl.params)


# Try likelihood, compiled version, likelihood gradient.
print("sampled parameter : ",logl(p0))
print("compiled version : ",jax.jit(logl)(p0))
print("gradient : ", jax.grad(logl)(p0))


# Make a numpyro model, transforming likelihood to standard `[-inf, inf]` parameter ranges.
npmodel = ds_numpyro.makemodel_transformed(logl)


# Make a numpyro NUTS sampler object. Numypro arguments for `infer.MCMC`
# and `infer.NUTS` are supported.
npsampler = ds_numpyro.makesampler_nuts(npmodel)


# Run with a set random seed.
npsampler.run(jax.random.PRNGKey(42))


# `discovery` enhances the `numpyro` sampler with a method `to_df()`
# that returns the sampler chain as a pandas `DataFrame`.
chain = npsampler.to_df()
print("Chain : ",chain)

pp.hist(chain['crn_log10_A'], bins=20, histtype='step', density=True);




# ## Experimenting with FFTint
if not os.path.exists("../data/"):
    print("\nError: 'data' folder not found in parent directory!!\n")
    exit(0)

allpsrs = [ds.Pulsar.read_feather(psrfile) for psrfile in sorted(glob.glob('../data/*-[JB]*.feather'))]

tspan = ds.getspan(allpsrs)

onesm = lambda psr: ds.PulsarLikelihood([psr.residuals,
                                         ds.makenoise_measurement(psr, psr.noisedict),
                                         ds.makegp_ecorr(psr, psr.noisedict),
                                         ds.makegp_timing(psr, svd=True),
                                         ds.makegp_fftcov(psr, ds.powerlaw, 61, tspan, oversample=6, name='rednoise')])

onegm = lambda allpsrs: ds.GlobalLikelihood([onesm(psr) for psr in allpsrs],
                                            globalgp=ds.makeglobalgp_fftcov(allpsrs, ds.powerlaw, ds.hd_orf, 61,
                                                                            T=tspan, oversample=6, name='gw'))

p0 = ds.sample_uniform(logl.params)

jlogl = jax.jit(jax.value_and_grad(logl))


import importlib

importlib.reload(ds.matrix)
importlib.reload(ds.signals)
importlib.reload(ds.likelihood)
importlib.reload(ds)

tspan = ds.getspan(allpsrs)

onesm = lambda psr: ds.PulsarLikelihood([psr.residuals,
                                         ds.makenoise_measurement(psr, psr.noisedict),
                                         ds.makegp_ecorr(psr, psr.noisedict),
                                         ds.makegp_timing(psr, svd=True)])

onegm = lambda allpsrs: ds.ArrayLikelihood([onesm(psr) for psr in allpsrs],
                                           commongp=ds.makecommongp_fftcov(allpsrs, ds.powerlaw, 61,
                                                                           T=tspan, oversample=6, name='rednoise'), 
                                           globalgp=ds.makeglobalgp_fftcov(allpsrs, ds.powerlaw, ds.hd_orf, 61,
                                                                           T=tspan, oversample=6, name='gw'))


clogl1 = model.cglogL(100, 5, 200, make_logdet='CG-Woodbury')
clogl2 = model.cglogL(100, 5, 200, make_logdet='CG-MDL')
clogl3 = model.cglogL(100, 5, 200, make_logdet='G-series')
clogl4 = model.cglogL(100, 5, 200, make_logdet='D-series')

jlogl = jax.jit(model.logL)
clogl1 = jax.jit(model.cglogL(100, detmatvecs=5, detsamples=200, clip=1e-6, make_logdet='CG-Woodbury'))
clogl2 = jax.jit(model.cglogL(100, detmatvecs=5, detsamples=200, clip=1e-6, make_logdet='CG-MDL'))
clogl3 = jax.jit(model.cglogL(100, detmatvecs=5, detsamples=200, make_logdet='G-series'))
clogl4 = jax.jit(model.cglogL(100, detmatvecs=5, detsamples=200, make_logdet='D-series'))

p0s, cls = [], []
for i in tqdm.tqdm(range(1000)):
    p0 = ds.sample_uniform(model.logL.params)
    p0s.append(p0)
    cls.append([jlogl(p0), clogl1(p0), clogl2(p0), clogl3(p0), clogl4(p0)])

cls2 = []
for i in tqdm.tqdm(range(1000)):
    cls2.append([jlogl(p0), clogl1(p0), clogl2(p0), clogl3(p0), clogl4(p0)])

cls = jnp.array(cls)

pp.subplot(2,2,1); pp.plot(cls[:,0], cls[:,1] - cls[:,0], '.')
pp.subplot(2,2,2); pp.plot(cls[:,0], cls[:,2] - cls[:,0], '.')

mask = jnp.abs(cls[:,3] - cls[:,0]) < 2000
pp.subplot(2,2,3); pp.plot(cls[mask,0], cls[mask,3] - cls[mask,0], '.')
pp.plot(cls[~mask,0], jnp.zeros(jnp.sum(~mask)), '.')

pp.subplot(2,2,4)
# pp.plot(cls[:,0], cls[:,4] - cls[:,0], '.')
mask = jnp.abs(cls[:,3] - cls[:,0]) < 10
pp.plot(cls[mask,0], cls[mask,4] - cls[mask,0], '.')

jnp.isnan(cls[:,3])


glogl1 = jax.jit(jax.grad(model.cglogL(100, 5, 200, make_logdet='CG-Woodbury')))
glogl2 = jax.jit(jax.grad(model.cglogL(100, 5, 200, make_logdet='CG-MDL')))
glogl3 = jax.jit(jax.grad(model.cglogL(100, 5, 200, make_logdet='G-series')))
glogl4 = jax.jit(jax.grad(model.cglogL(100, 5, 200, make_logdet='D-series')))


jnpa, jspa = jax.numpy.linalg, jax.scipy.linalg


# orfcf, phicf, FtNmF = clogl(p0)

orfmat = orfcf[0].T @ orfcf[0]
phimat = phicf[0].T @ phicf[0]
phiinv = jspa.cho_solve(phicf, jnp.eye(61))
orfinv = jspa.cho_solve(orfcf, jax.numpy.eye(5))


# Try CG approach
Y = np.random.randn(5,61)
AY = jnp.einsum('akl,al->ak', FtNmF, jnp.einsum('ab,bc,cl->al',
                                                orfcf[0].T, orfcf[0], jnp.einsum('li,ij,aj->al',
                                                                                 phicf[0].T, phicf[0], Y))) + Y


# Do diagonal part separately
orfinv = jspa.cho_solve(orfcf, jax.numpy.eye(5))
orfinv = np.diag(np.diag(orfinv)) + 1.0 * (orfinv - np.diag(np.diag(orfinv)))
Sigma = jnp.block([[jnp.make2d(val * phiinv) for val in row] for row in orfinv]) + jax.scipy.linalg.block_diag(*FtNmF)

print("Solved Diagonal part separately. Determinant : ", 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(jspa.cho_factor(Sigma)[0])))))

i1, i2 = jnp.diag_indices(61, ndim=2)
cfD = jspa.cho_factor(jnp.diag(orfinv)[:,None,None] * phiinv[None,:,:] + FtNmF)
logD = 2.0 * jnp.sum(jnp.log(jnp.abs(cfD[0][:, i1, i2])))
print("Determinant : ",logD)

E = jax.vmap(lambda c, m: jspa.cho_solve((c, False), m), in_axes=(0,None))(cfD[0], phiinv)
traces = jnp.einsum('nij,mji->nm', E, E)
gamma_prod = orfinv * orfinv.T
off_diag_mask = ~jnp.eye(5, dtype=bool)
print(logD -0.5 * jnp.sum(gamma_prod * traces * off_diag_mask))

traces3 = jnp.einsum('aij,bjk,ckl->abc', E, E, E)
gamma_prod3 = jnp.einsum('ij,jk,ki->ijk', orfinv, orfinv, orfinv)
N = 5
i_idx, j_idx, k_idx = jnp.meshgrid(jnp.arange(N), jnp.arange(N), jnp.arange(N), indexing="ij")
off_diag_mask3 = (i_idx != j_idx) & (j_idx != k_idx) & (k_idx != i_idx)
print(logD -0.5 * jnp.sum(gamma_prod * traces * off_diag_mask) + (1/3.0) * jnp.sum(gamma_prod3 * traces3 * off_diag_mask3))

print("Printing matrix...\n",gamma_prod * off_diag_mask)
print("Printing matrix...\n",gamma_prod3 * off_diag_mask3)

traces = jnp.einsum('nij,nkj->nk', E, E)  # shape (N, N), Tr(E_i @ E_j)

# Elementwise product of Gamma * Gamma^T (outer product of Gamma)
gamma_prod = Gamma * Gamma.T  # shape (N, N)

# Mask out the diagonal (i ≠ j)
off_diag_mask = ~jnp.eye(N, dtype=bool)
correction_matrix = gamma_prod * traces * off_diag_mask
correction = -0.5 * jnp.sum(correction_matrix)

print("Matrix shape : ",cfD[0].shape)


# Expand in G
phiG = phimat @ FtNmF
phiG = phicf[0].T @ (phicf[0] @ FtNmF)
print(jax.numpy.diag(orfmat) @ jax.numpy.trace(phiG, axis1=1, axis2=2))

print(-0.5 * jax.numpy.diag(orfmat)**2 @ jax.numpy.trace(phiG @ phiG, axis1=1, axis2=2))

print(+(1/3) * jax.numpy.diag(orfmat)**3 @ jax.numpy.trace(phiG @ phiG @ phiG, axis1=1, axis2=2))

clogl = jax.jit(model.cglogL(100, 40, 2000))

glogl = jax.jit(jax.grad(model.cglogL(100, 40, 1000)))

glogl = jax.jit(jax.grad(model.logL))


clogls = [model.cglogL(1000, n, 1000)(p0) for n in range(10,110,10)]

clogls2 = [model.cglogL(1000, 50, l)(p0) for l in range(100,1100,100)]

# plotting
pp.plot(range(100,1100,100), clogls2)
pp.axhline(logl(p0))

pp.plot(range(10,110,10), clogls)
pp.axhline(logl(p0))

logl = jax.jit(model.logL)
print(logl(p0))

p0 = ds.sample_uniform(logl.params)

clogl = model.cglogL(1000)
print(clogl(p0))


import matfree
from matfree import decomp, funm, stochtrace

num_matvecs = 3
tridiag_sym = decomp.tridiag_sym(num_matvecs)
problem = funm.integrand_funm_sym_logdet(tridiag_sym)
x_like = jnp.ones((nrows,), dtype=float)
sampler = stochtrace.sampler_normal(x_like, num=1_000)
estimator = stochtrace.estimator(problem, sampler=sampler)
logdet = estimator(matvec, jax.random.PRNGKey(1))
print(logdet)

matfree.slq

from matfree.trace_estimation import slq


clogl = model.cglogL(1000)
print(clogl(p0))

clogl = jax.jit(model.cglogL(1000))
print(clogl(p0))

jlogl = jax.jit(jax.value_and_grad(logl))

Phimat = model.globalgp.Phi.getN(p0)

Phiinv, detPhi = model.globalgp.Phi_inv(p0)

orfmat = ds.matrix.jnparray([[ds.hd_orf(p1.pos, p2.pos) for p1 in allpsrs[:5]] for p2 in allpsrs[:5]])
invorf = jnp.linalg.inv(orfmat)

phi = Phimat[:61,:61]
invphi = jnp.linalg.inv(phi)
invPhi = jnp.block([[jnp.make2d(val * invphi) for val in row] for row in invorf])

Phiinv = jnp.linalg.inv(Phimat[:61*5,:61*5])

jnp.allclose(Phiinv, invPhi)

phicf = jax.scipy.linalg.cho_factor(phi)

orfcf = jax.scipy.linalg.cho_factor(orfmat)

def matvec3(orfcf, phicf, FtNmF):
    def apply(FtNmy):
        term1 = jax.scipy.linalg.cho_solve(orfcf, jax.scipy.linalg.cho_solve(phicf, FtNmy.T).T)
        term2 = jnp.einsum('kij,kj->ki', FtNmF, FtNmy)
        return term1 + term2

    return apply

FtNmy = np.random.randn(5*61).reshape(5,61)
FtNmF = np.random.randn(5,61,61)
FtNmF = 0.5 * (FtNmF + np.swapaxes(FtNmF, 1, 2))
# invphi = 0.5 * (invphi + invphi.T)
# invorf = 0.5 * (invorf + invorf.T)

def matvec(invorf, invphi, FtNmF):
    def apply(FtNmy):
        # return jnp.einsum('lk,ki->li', invorf, jnp.einsum('ij,kj->ki', invphi, FtNmy))
        # return invorf @ jnp.einsum('ij,kj->ki', invphi, FtNmy)
        # return jnp.einsum('lk,ij,kj->li', invorf, invphi, FtNmy)
        return jnp.einsum('kl,ij,lj->ki', invorf, invphi, FtNmy) + jnp.einsum('kij,kj->ki', FtNmF, FtNmy)

    return apply

def matvec2(invorf, invphi, FtNmF):
    @jax.jit
    def apply(FtNmy):
        return invorf @ FtNmy @ invphi.T + jnp.einsum('kij,kj->ki', FtNmF, FtNmy)
    
    return apply

print("matvec3 : ",matvec3(orfcf, phicf, FtNmF)(FtNmy)[1,:10])
print("matvec2 : ",matvec2(invorf, invphi, FtNmF)(FtNmy)[1,:10])
print("matvec  : ",matvec(invorf, invphi, FtNmF)(FtNmy)[1,:10])

sol = jaxopt.linear_solve.solve_cg(matvec3(orfcf, phicf, FtNmF), FtNmy, maxiter=20000)
print(jnp.sum(sol * FtNmy))


sigma = invPhi + jax.scipy.linalg.block_diag(*FtNmF)

Y = FtNmy.reshape(5 * 61)

print(sigma @ Y - matvec2(invorf, invphi, FtNmF)(FtNmy).reshape(5 * 61))

m1 = matvec(invorf, invphi, FtNmF)
m2 = matvec2(invorf, invphi, FtNmF)
print(m2(FtNmy))

import jaxopt
jaxopt.linear_solve.solve_normal_cg
from jaxopt.linear_solve import solve_cg

print(jnp.unique((matvec(invorf, invphi, FtNmF)(FtNmy) - matvec2(invorf, invphi, FtNmF)(FtNmy)).reshape(30*61)))

matvec(invorf, invphi, FtNmF)(FtNmy).reshape(30 * 61) - (invPhi + jax.scipy.linalg.block_diag(*FtNmF)) @ FtNmy.reshape(30 * 61)


print((invPhi + jax.scipy.linalg.block_diag(*FtNmF)) @ FtNmy.reshape(30 * 61))

jnp.einsum('ij,ki->kj', invphi, FtNmy)
