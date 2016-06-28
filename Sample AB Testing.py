import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import pymc

#website A had 1055 clicks and 28 signups
values_A = np.hstack(([0]*(1055-28),[1]*28))
#Website B had 1057 clicks and 45 signups
values_B = np.hstack(([0]*(1057-45),[1]*45))

#Create uniform prior for probabilities p_a and p_b
p_A = pymc.Uniform("p_A", 0, 1)
p_B = pymc.Uniform("p_B", 0, 1)

#Creates a posterior distribution of B - A
@pymc.deterministic
def delta(p_A =  p_A, p_B = p_B):
    return p_B - p_A

#Create Bernoulli variables for observation
obs_A = pymc.Bernoulli("obs_A", p_A, value = values_A, observed = True)
obs_B = pymc.Bernoulli("obs_B", p_B, value = values_B, observed = True)

#Create model and run sampling
model = pymc.Model([p_A, p_B, delta, values_A, values_B])
mcmc = pymc.MCMC(model)
#Sample 1,000,000 million points and throw out first 500,000
mcmc.sample(1000000, 500000)

delta_distribution = mcmc.trace("delta")[:]

sns.kdeplot(delta_distribution, shade = True)
plt.axvline(0.00, color = "black")


print ("Probability that website A gets MORE sign-ups than site B: %0.3f" %
       (delta_distribution < 0).mean())
print ("Probability that website gets LESS sign-ups than site B: %0.3f" %
       (delta_distribution > 0).mean())


@pymc.stochastic(dtype=np.float64)
def beta_priors(value=[1.0, 1.0]):
    a, b = value
    if a <= 0 or b <= 0:
        return -np.inf
    else:
        return np.log(np.power((a+b), -2.5))

a = beta_priors[0]
b = beta_priors[1]


#hidden true rate for each website
true_rates = pymc.Beta("true_rates", a, b, size=5)

#observed values
trials = np.array([1055, 1057, 1065, 1039, 1046])
successes = np.array([28, 45, 69, 58, 60])
observed_values = pymc.Binomial("observed_values", trials, true_rates, observed=True,
                                value=successes)

model = pymc.Model([a, b, true_rates, observed_values])
mcmc = pymc.MCMC(model)

#Generate 1,000,000 samples and throw out first 500,000
mcmc.sample(1000000, 500000)


diff_CA = mcmc.trace("true_rates")[:][:,2] - mcmc.trace("true_rates")[:][:,0]
sns.kdeplot(diff_CA, shade=True, label="Difference site C - site A")
plt.axvline(0.0, color="black")

print ("Probability that website A gets MORE sign-ups than website C: %0.3f" %
       (diff_CA < 0).mean())
print ("Probability that website A gets LESS sign-ups than website C: %0.3f" %
       (diff_CA > 0).mean())

sns.kdeplot(siteA_distribution, shade= True, label="Bernoulli Model")
sns.kdeplot(mcmc.trace("true_rates")[:][:,0], shade=True, label="Hierachical Beta")
plt.axvline(0.032, color="black")