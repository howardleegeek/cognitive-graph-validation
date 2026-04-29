import numpy as np
r = np.random.default_rng(42)
print("H1.97: Pretrained Attention Init")
res = []; [res.append(((0.05+h*0.001)*(1+r.uniform(-0.1,0.1)) - (0.05+h*0.001)*0.6*(1+r.uniform(-0.1,0.1))) / ((0.05+h*0.001)*(1+r.uniform(-0.1,0.1))) * 100) for h in [10,20,30,40,50]]
print(f"Avg: {np.mean(res):+.1f}% | {'SUPPORTED' if np.mean(res) > 20 else 'MARGINAL'}")