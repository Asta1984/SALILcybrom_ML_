import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse

#steps 1: Create dummy data

np.random.seed(42)
cluster1 = np.random.randn(100, 2)*1.5+[4,4]
cluster2 = np.random.randn(100, 2)*1.5 + [4,4]

