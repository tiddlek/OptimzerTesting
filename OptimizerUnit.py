import GPyOpt
import GPy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


target = 550

def parseData(filename):
    data = pd.read_csv(filename)

    concentrations = []
    lambda_maxes = []

    for i in range(19):
        index = i*3
        lambda_maxes.append(data["Experiment Result"][i])
        concentrations.append(data["Recipes"][i])

    X = np.array(concentrations).reshape(-1,1)
    Y = np.array(lambda_maxes).reshape(-1,1)
    
    return X,Y

def getPredictions():
    temp = 0
    fullPred = []
    fullStd = []
    for i in range(200):
        pred, std = gp_model.predict(np.array([[temp]]))
        fullPred.append(pred)
        fullStd.append(std)
        temp += 0.005
    return fullPred, fullStd

def prepPlot():
    plt.figure(figsize=(4,4), dpi=200, facecolor='w', edgecolor='k')
    plt.rc('axes', linewidth = 2)
    plt.xlabel("[KBr] (mM)", fontsize = 16)
    plt.ylabel("Lambda Max (nm)", fontsize = 16)
    plt.tick_params(axis = "both", width = 2)
    plt.axis([0,0.002, 0, 900])
    plt.xticks(np.arange(0,0.0021, step=0.0005), fontsize = 12)
    plt.yticks(fontsize = 14)
    plt.title("Matern Kernel", fontsize = 16, pad = 20, loc="center")

def f(x):
    print(f"X: {np.array(x)}")
    y = gp_model.predict(x) #This line of code is not working as expected
    print(f"Prediction: {y[0]}")
    print(f"Result: {abs(normalize(target,300,900)-y[0])}")
    return abs(normalize(target,300,900)-y[0])

def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)

def denormalize(x_normalized, min_val, max_val):
    return x_normalized * (max_val - min_val) + min_val
    
X,Y = parseData("KAT_018.csv")

X_normalized = normalize(X, 0, 0.002)
Y_normalized = normalize(Y, 300, 900)

domain = [{'name': 'x', 'type': 'continuous', 'domain': (0, 1)}]
space = GPyOpt.core.task.space.Design_space(domain)

Exponential = GPy.kern.Exponential(1,lengthscale=0.1)
Radial = GPy.kern.RBF(1)
Matern32 = GPy.kern.Matern32(1)

gp_model = GPyOpt.models.GPModel(kernel=Exponential, noise_var=0, optimize_restarts=10, verbose=False)
gp_model.updateModel(X_normalized, Y_normalized, None, None)
acq_optimizer = GPyOpt.optimization.AcquisitionOptimizer(space, optimizer="lbfgs")

MPI = GPyOpt.acquisitions.AcquisitionMPI(gp_model,space,acq_optimizer,jitter=1e-6)
LCB_Explore = GPyOpt.acquisitions.AcquisitionLCB(gp_model,space,acq_optimizer,exploration_weight=10)
LCB_Exploit = GPyOpt.acquisitions.AcquisitionLCB(gp_model,space,acq_optimizer,exploration_weight=2)

objective = GPyOpt.core.task.objective.SingleObjective(f)
evaluator = GPyOpt.core.evaluators.Sequential(MPI)
model = GPyOpt.methods.ModularBayesianOptimization(objective=objective,model=gp_model,space=space,acquisition=LCB_Explore,evaluator=evaluator,X_init=X_normalized,Y_init=Y_normalized, normalize_Y=False)

predictions, stdev = getPredictions()

predictions = denormalize(np.array(predictions).flatten(),300, 900)
stdev = np.array(stdev).flatten()*(600)

model.run_optimization(10)

print(denormalize(model.suggest_next_locations()[0][0],0,0.002))

linespace = np.linspace(0,0.002,200).tolist()

plt.scatter(linespace, stdev)
plt.tight_layout()
plt.show()

prepPlot()

plt.fill_between(
    linespace,
    predictions + stdev,
    predictions - stdev,
    alpha = 0.3
)

plt.plot(linespace, predictions)
plt.scatter(X,Y)
plt.tight_layout()
plt.show()
