#adapted from bayesian for hackers cam davidson book

import pymc as pm

#parameters are bounds of uniform
p = pm.Uniform("p", lower=0, upper=1)

#set constants
p_true = 0.05
N = 1500

#sample N Bernoulli random variables from Ber(0.05)
#each random var has a chance of being a 1
occurrences = pm.rbernoulli(p_true, N)

print (occurrences)
print (occurrences.sum())

print ("What is the observed frequency in Group A? %.4f" % occurrences.mean())
print ("Does this equal the true frequency? %s" % (occurrences.mean() == p_true))

#include observations which are Bernoulli
obs = pm.Bernoulli("obs", p, value=occurrences, observed=True)

mcmc = pm.MCMC([p, obs])
mcmc.sample(18000, 1000)

figsize(12.5, 4)
plt.title("Posterior distribution of $p_A$, the true effectiveness of site A")
plt.vlines(p_true, 0, 90, linestyle="--", label="true $p_A$ (unknown)")
plt.hist(mcmc.trace("p")[:], bins=25, histtype="stepfilled", normed=True)
plt.legend()


#finding the difference between b and a
figsize(12, 4)

#two quantities unknown to us
true_p_A = 0.05
true_p_B = 0.04

#unequal sample sizes are fine in Bayesian analysis
N_A = 1500
N_B = 750

#generate some observations
observations_A = pm.rbernoulli(true_p_A, N_A)
observations_B = pm.rbernoulli(true_p_B, N_B)
print ("Obs from Site A: ", observations_A[:30].astype(int), "...")
print ("Obs from Site B: ", observations_B[:30].astype(int), "...")

print (observations_A.mean())
print (observations_B.mean())


#set up pymc model, assuming uniform priors for p_A and p_B
p_A = pm.Uniform("p_A", 0, 1)
p_B = pm.Uniform("p_B", 0, 1)

#define deterministic delta function (unknown of interest)
@pm.deterministic
def delta(p_A=p_A, p_B=p_B):
    return p_A - p_B

#set of observations, in this case two observation datasets
obs_A = pm.Bernoulli("obs_A", p_A, value=observations_A, observed=True)
obs_B = pm.Bernoulli("obs_B", p_B, value=observations_B, observed=True)

mcmc = pm.MCMC([p_A, p_B, delta, obs_A, obs_B])
mcmc.sample(20000, 1000)

#plot posterior distributions of three unknowns
p_A_samples = mcmc.trace("p_A")[:]
p_B_samples = mcmc.trace("p_B")[:]
delta_samples = mcmc.trace("delta")[:]

figsize(12.5, 10)

#histogram of posteriors
ax = plt.subplot(311)

plt.xlim(0, .1)
plt.hist(p_A_samples, histtype="stepfilled", bins=25, alpha=0.85,
         label="posterior of $p_A$", color="#A60628", normed=True)
plt.vlines(true_p_A, 0, 80, linestyle="--", label="true $p_A$ (unknown)")
plt.legend(loc = "upper_right")
plt.title("Posterior distributions of $p_A$, $p_B$, and delta unknowns")

ax = plt.subplot(312)

plt.xlim(0, .1)
plt.hist(p_B_samples, histtype="stepfilled", bins=25, alpha=0.85,
         label="posterior of $p_B$", color="#467821", normed=True)
plt.vlines(true_p_B, 0, 80, linestyle="--", label="true $p_B$ (unknown)")
plt.legend(loc="upper right")

ax = plt.subplot(313)
plt.hist(delta_samples, histtype="stepfilled", bins=30, alpha=0.85,
         label="posterior of delta", color="#7A68A6", normed=True)
plt.vlines(true_p_A - true_p_B, 0, 60, linestyle="--",
           label="true delta (unknown)")
plt.vlines(0, 0, 60, color="black", alpha=0.2)
plt.legend(loc="upper right");

#count number of samples less than 0, i.e area under curve
#before 0, represent probability that site A worse than site B
print ("Probability site A is WORSE than site B: %.3f" % \
       (delta_samples < 0).mean())

print ("Probability site A is BETTER than site B: %.3f" % \
       (delta_samples > 0).mean())

#
