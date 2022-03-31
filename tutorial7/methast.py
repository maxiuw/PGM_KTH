from pstats import Stats
from scipy import stats 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
'''
source https://www.youtube.com/watch?v=0lpT-yveuIA
1. Sample from makrov chain
2. picking random initial values 
3. iteratively accepting/rejecting candidates drawn from the easy-to-sample distribution
p(th) -> g(th) -> we know that p(th) is proportional to g(th)

Steps 
1. Initial values for theta th_0 (params)
2. for i in range(m):
    a) draw a candiadate th* (from a proposal distribution) th*~q(th|th_i-1)
    b) alpha = g(th*) * q(th_i-1|th*)/q(th*|th_i-1)*g(i_th)
    if (alpha >= 1)
        accept th* and th_i <- th*
    elif (0 < alpha <= 1)  
        accept th* and th_i <- th* p = alpha
        reject th*, th_i <- th_i-1 p = 1 - alpha

q - candidate generating distribution 
    -> q does not depend on the previous values of th --> q(th*) is always the same distribution --> it should be similar to p(th)
    -> q depends on the previous values of th (random walk M-H) -> proposal distribution is centered on the previous iteration 
        e.g. normal distribution using mean th_i-1 and conts var, it causes q(th_i-1|th*)=q(th*|th_i-1) and canceles each othe out 
th* - candiadte 
''' 
def init_distribution(theta1, theta2, x):
    '''
    returns normal distribution, given parameters
    '''
    return theta1['a'] * stats.norm.pdf(x, theta1['mean'], theta1['cov']) + theta2['a'] * stats.norm.pdf(x, theta2['mean'],theta2['cov'])

    # return np.random.multivariate_normal(theta['mean'],theta['cov'],n)

def accept(p_i, p_0, q_i, q_0):
    '''
    calcualting acceptance rate for the current x 
    '''
    return np.random.uniform(size = 1) < (p_i * q_0)/(p_0 * q_i)
    

def run(theta1, theta2, iterations = 1000, scale = 100):
    x_0 = 0 #np.random.random_integers(size = 1) # we should be able do ajdust the size 
    accepted = []
    for _ in range(iterations):
        # sample new rv xi from the norm
        x_i = stats.norm(x_0 , 100).rvs(1)
        # calculate p0 (pi) and pi (p* o pi+1)
        p_i = init_distribution(theta1, theta2, x_i)
        p_0 = init_distribution(theta1, theta2, x_0)
        # loc is the mean, calculate q(x*|xi) and q(xi|x*) 
        q_i = stats.norm.pdf(x_0, loc = x_i, scale = scale)
        q_0 = stats.norm.pdf(x_i, loc = x_0, scale = scale)
        A =  accept(p_i, p_0, q_i, q_0)
        # if acceptance rate is True, accept new sample      
        if A:            
            x_0 = x_i
        accepted.append(x_0)

    # results 
    x = np.linspace(start = min(accepted), stop = max(accepted), num = len(accepted))
    fig, ax = plt.subplots(ncols=2)
    ax[0].plot(range(len(accepted)),accepted)
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("Sample's value")

    # ax[1].plot(range(len(accepted)),accepted, label='predicted distribution')
    sns.distplot(accepted, kde = True, ax = ax[1], label = "predicted distribution")
    # sns.lineplot(x, np.array(init_distribution(theta1, theta2, x)), ax = ax[1], label = "distribution")

    # sns.distplot(init_distribution(theta1, theta2, x), norm_hist=True, kde=True, ax = ax[1], label = "true distribution")
    sns.lineplot(x.reshape(iterations),init_distribution(theta1, theta2, x).reshape((iterations)), ax=ax[1], label = "true distribution")
    # ax[1].plot(range(len(x)), np.array(init_distribution(theta1, theta2, x)), label = 'true distribution')
    ax[0].set_title("Accepted samples")
    ax[1].set_title(f"GT and predicted pdf's with sampling variance {scale}")
    plt.legend()
    plt.draw()
if __name__ == "__main__":
    theta1 = {
        "a": 0.5,
        "cov" : 1,
        "mean" : 0
    }
    theta2 = {
        "a": 0.5,
        "cov" : 0.5,
        "mean" : 3
    }
    iterations = 10000
    # trying different scale (variants)
    for scale in [0.1, 1, 10, 100]:
    # scale = 1 
        run(theta1, theta2, iterations, scale)
    plt.show()

