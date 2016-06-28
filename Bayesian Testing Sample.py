def generate_data(no_samples, treatment_proportion=0.1, treatment_mu=1.2, control_mu=1.0, sigma=0.4):
    """
    Generate sample data from experiment
    """
    rnd = np.random.RandomState(seed=12345)
    treatment = rnd.bionmial(1, treatment_proportion, size=no_samples)
    treatment_outcome = rnd.normal(treatment_mu, sigma, size=no_samples)
    control_outcome = rnd.normal(control_mu, sigma, size=no_samples)
    observed_outcome = (treatment * treatment_outcome + (1 - treatment) * control_outcome)
    return pd.DataFrame({"treatment": treatment,
                         "outcome": observed_outcome})


    def fit_uniform_priors(data):
    """
    Fit data with uniform priors on mu
    """
    treatment = data["treatment"].values
    with pm.Model() as model:
        #specify priors for mean
        treatment_mean = pm.Uniform("Treatment mean",
                                    lower=0,
                                    upper=100)
        control_mean = pm.Uniform("Control mean",
                                  lower=0,
                                  upper=100)
        #Estimate treatment effect
        treatment_effect = pm.Deterministic("Treatment effect",
                                            treatment_mean - control_mean)
        
        #Specify prior for sigma
        sigma = pm.InverseGamme("Sigma", 1.1, 1.1)
        
        #Data model
        outcome = pm.Normal("Outcome",
                            treatment * treatment_mean
                            + (1 - treatment) * control_mean,
                            sd=sigma, observed=data["outcomme"])
        
        #Fit
        samples = 5000
        start = pm.find_MAP()
        step = pm.Metropolis()
        trace = pm.sample(samples, step, start, njobs=3)
        
        #Discard burn in
        trace = trace[int(samples * 0.5):]
        
        return pm.trace_to_dataframe(trace)

treatment_mean = pm.Normal("Treatment mean",
                           prior_mu,
                           sd=prior_sigma)
control_mean = pm.Normal("Control mean",
                         prior_mu,
                         sd=prior_sigma)