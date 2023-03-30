"""
================================
Gaussian Mixture Model Selection
================================

Model selection performed with Gaussian Mixture Models on one dimensional data.
Model selection concerns both the covariance type and the number of components in the model.
Unlike Bayesian procedures, such inferences are prior-free.

"""

import numpy as np
import itertools

#from scipy import linalg
import matplotlib.pyplot as plt
#import matplotlib as mpl

from sklearn import mixture

print(__doc__)


number_of_Gaussians = 10 # Parameter of the code = max number of Gaussians considered


#Create array from file
file = open("Data-Flood.txt", 'r')
list = []
for line in file:
    list.append([line.strip()])
X = np.array(list, dtype = float)
print(list)
#print(X)
print("Input shape: ", X.shape)

# Generate best GMM using AIC
lowest_aic = np.infty
aic = []

n_components_range = range(1, number_of_Gaussians+1)
cv_types = ['spherical', 'tied', 'diag', 'full']
for cv_type in cv_types:
    for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(n_components=n_components,
                                          covariance_type=cv_type)
            gmm.fit(X)
            aic.append(gmm.aic(X))
            if aic[-1] < lowest_aic:
                lowest_aic = aic[-1]
                aic_best_gmm = gmm

aic = np.array(aic)
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                              'darkorange'])
bars = []

# Plot the AIC scores
plt.figure(figsize=(8, 6))
spl = plt.subplot(2, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + .2 * (i - 2)
    bars.append(plt.bar(xpos, aic[i * len(n_components_range):
                                    (i + 1) * len(n_components_range)],
                        width=.2, color=color))
plt.xticks(n_components_range)
plt.ylim([aic.min() * 1.01 - .01 * aic.max(), aic.max()])
plt.title('AIC score per model')
xpos = np.mod(aic.argmin(), len(n_components_range)) + .65 +\
        .2 * np.floor(aic.argmin() / len(n_components_range))
plt.text(xpos, aic.min() * 0.97 + .03 * aic.max(), '*', fontsize=14)
spl.set_xlabel('Number of components')
plt.legend([b[0] for b in bars], cv_types)

print("")
print("GMM model selection using the AIC score")
plt.show()
print("")
print("Lowest AIC score:", lowest_aic)
print("Model: ", aic_best_gmm)
print("Parameters: ", aic_best_gmm.get_params())

# Generate best GMM using BIC
lowest_bic = np.infty
bic = []

n_components_range = range(1, number_of_Gaussians+1)
cv_types = ['spherical', 'tied', 'diag', 'full']
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type)
        gmm.fit(X)
        bic.append(gmm.bic(X))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            bic_best_gmm = gmm

bic = np.array(bic)
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                              'darkorange'])
bars = []

# Plot the BIC scores
plt.figure(figsize=(8, 6))
spl = plt.subplot(2, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + .2 * (i - 2)
    bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))
plt.xticks(n_components_range)
plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
plt.title('BIC score per model')
xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
    .2 * np.floor(bic.argmin() / len(n_components_range))
plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
spl.set_xlabel('Number of components')
spl.legend([b[0] for b in bars], cv_types)

print("")
print("GMM model selection using the BIC score")
plt.show()
print("")
print("Lowest BIC score:", lowest_bic)
print("Model: ", bic_best_gmm)
print("Parameters: ", bic_best_gmm.get_params())


# Generate random samples from the fitted Gaussian distribution
n_gen_samples = 100 # max number of generated samples

print("")
print("BIC scores for generated samples using the best BIC GMM")
for i in range(100,1000,100):
    Y,Z = bic_best_gmm.sample(n_gen_samples) #Y is dataset, Z are cluster labels
    print("For ", i, " new samples, ", "BIC is ", bic_best_gmm.bic(Y))












