from kan import KAN
import torch

def build_kan(input_dim: int) -> KAN:
    # [input_dim, hidden, 1] — small network, interpretability > capacity
    model = KAN(width=[input_dim, 8, 1], grid=5, k=3)
    model.speed()   # CRITICAL: disables symbolic branch, 10x faster
    return model

# One KAN per regime — train each on its subset of data
def train_regime_kan(kan, X_regime, y_regime, steps=200):
    dataset = {"train_input": X_regime, "train_label": y_regime,
               "test_input": X_regime, "test_label": y_regime}
    kan.fit(dataset, opt="LBFGS", steps=steps, lamb=0.01)  # lamb = L1
    return kan