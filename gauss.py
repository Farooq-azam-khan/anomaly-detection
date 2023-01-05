import numpy as np 

def gauss_anom(dataset): 
    examples, features = dataset.shape 
    mu = dataset.mean(axis=0) 
    assert len(mu) == features 
    var = np.zeros(features) 
    for i in range(features): 
        var[i] = np.sum((dataset[:, i] - mu[i])**2) / examples 
    return mu, var 

dataset = np.array([[1,2,3], [1.1, 2, 3.5], [1.15, 2.25, 3], [1.001, 2.002, 3.001]])
print(dataset)
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
    return prob 
threashold = 0.01
prob = compute_p(new_datapoint, mu, var) 
print(f'prob(new_datapoint) = {prob}')
if prob > threashold:  
    print(f'Not an Anomaly')
else: 
    print('Anomaly')
