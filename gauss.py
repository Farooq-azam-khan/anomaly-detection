import numpy as np


def gauss_anom(dataset):
    examples, features = dataset.shape
    mu = dataset.mean(axis=0)
    assert len(mu) == features
    var = np.zeros(features)
    for i in range(features):
        var[i] = np.sum((dataset[:, i] - mu[i])**2) / examples
    return mu, var


dataset = np.array([
    [1  ,2 ,3   ],
    [1.1, 2, 3.5],
    [1.1, 2, 3.5],
    [1.1, 2, 3.5],
    [1.1, 2, 3.5],
    [1.1, 2, 3.5],
    [1.1, 2, 3.5],
    [1.1, 2, 3.5],
    [1.1, 2, 3.5],
    [1.1, 2, 3.5],
    [1.1, 2, 3.5],
    [1.1, 2, 3.5],
    [1.1, 2, 3.5],
    [1.1, 2, 3.5],
    [1.1, 2, 3.5],
    [1.1, 2, 3.5],
    [1.1, 2, 3.5],
    [1.1, 2, 3.5],
    [1.1, 2, 3.5],
    [1.1, 2, 3.5],
    [1.15, 2.25, 3],
    [1.15, 2.25, 3],
    [1.15, 2.25, 3],
    [1.15, 2.25, 3],
    [1.15, 2.25, 3],
    [1.001, 2.002, 3.001]])
mu, var = gauss_anom(dataset)
print('mu=')
print(mu)
print('var=')
print(var)

new_datapoint = np.array([10, 20, 3])


def compute_p(new_datapoint, mu, var):
    assert len(new_datapoint) == len(mu)
    assert len(mu) == len(var)
    prob = 1
    for x, m, s in zip(new_datapoint, mu, var):
        coef = 1/(np.sqrt(2*np.pi)*s)
        prob *= coef * np.exp((-1*(x-m)**2)/(2*s*s))

    coeff1 = 1/(np.sqrt(2*np.pi)*var)
    coeff2 = np.exp(-1 * (new_datapoint-mu)**2 / (2*var*var))
    print(coeff1*coeff2)
    return prob


threashold = 0.001
prob = compute_p(new_datapoint, mu, var)


def is_anomaly(threashold, prob):
    print(f'prob(point) = {prob}')
    if prob > threashold:
        print(f'Not an Anomaly')
    else:
        print('Anomaly')


is_anomaly(threashold, prob)
prob = compute_p(np.array([1.1, 2, 3.5]), mu, var)
is_anomaly(threashold, prob)
