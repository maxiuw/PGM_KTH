from http.client import PARTIAL_CONTENT
from termios import PARENB
from turtle import color
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from matplotlib import cm

def gen_x_t(theta, t, x, n):
    # generating gaussian model at x_t 
    if t == 0:
        return stats.norm(loc=theta['x']['mean'], scale=theta['x']['cov']).rvs(n)
    v = stats.norm(loc=theta['v']['mean'], scale=theta['v']['cov']).rvs(n)

    return 0.5 * x + 25*(x/(1+x**2)) + 8 * np.math.cos(1.2*t) + v

def gen_y_t(theta, xt, n):
    # generating gaussian model at y_t 
    w = stats.norm(loc=theta['w']['mean'], scale=theta['w']['cov']).rvs(n)
    return (xt**2 /20) + w

def calculate_weights(x,y, theta):
    # calculate normalzied weights 
    weights = stats.norm.pdf(y, (x**2)/20, theta['w']['cov'])
    return weights/np.sum(weights)

def run(theta, MAX_STEPS,PARTICLES):
    
    # initializing the arrays
    x = np.zeros((MAX_STEPS,PARTICLES))
    y = np.zeros((MAX_STEPS,PARTICLES))
    x = np.zeros((MAX_STEPS,PARTICLES))
    weights = np.zeros((MAX_STEPS,PARTICLES))
    for t in range(MAX_STEPS):    
        # calculate x, different x at time step 0   
        x[t] = gen_x_t(theta,t,x[t-1], PARTICLES) 
        # calc y
        y[t] = gen_y_t(theta, x[t], PARTICLES)
        # calculate weights selected samples
        weights[t] = calculate_weights(x[t],y[t], theta)
        # select values with the probability corresponding to the weight
        selection = np.random.choice(x[t], p = weights[t], size = PARTICLES)
        x[t] = selection
    return x

def plot(x,k):
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    time = 200 
    n = 25
    grid = np.linspace(-25, 25, n)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel("values")
    ax.set_ylabel("timestep")
    ax.set_zlabel("density")
    for i in range(0,time,20):
        t = i * np.ones(n) 
        density = np.histogram(x[i], bins=n, density=True)[0]
        print(cm.hot(density).shape)

        # vals = sns.kdeplot(x[i],gridsize=n).get_lines()[0].get_data()
        # plotting pdf at timestep x, y, z 
        ax.plot(grid , t, density, c=plt.get_cmap('inferno'))
        ax.cmap()
    ax.set_title(f"PDF with {k} particles")
    plt.draw()

if __name__ == "__main__":
    theta = {
        'x': {
        "mean": 0,
        "cov": np.sqrt(10)
        },
        'v': {
            "mean": 0,
            "cov": np.sqrt(10)
        },
        'w': {
            "mean": 0,
            "cov": 1
        }
    }
    MAX_STEPS = 200
    # for PARTICLES in [10,100,1000,10000,100000,1000000]:
    PARTICLES = 10  
    # PARTICLES = 10
    x = run(theta, MAX_STEPS,PARTICLES)
    plot(x, PARTICLES)
    plt.show()