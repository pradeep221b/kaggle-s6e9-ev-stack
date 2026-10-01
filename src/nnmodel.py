"""RealMLP-TD fed in-fold TE(exact income). Leak-free: TargetEncoder(cv=5) cross-fits."""
import logging, warnings
warnings.filterwarnings('ignore')
for _n in ('pytorch_lightning', 'lightning', 'lightning.pytorch'):
    logging.getLogger(_n).setLevel(logging.ERROR)
from pytabkit import RealMLP_TD_Classifier
from sklearn.preprocessing import TargetEncoder
from cv import SEED

INC = 'Annual_Income_USD'


class TEMLP:
    def __init__(self, n_epochs=24, seed=SEED):
        self.n_epochs, self.seed = n_epochs, seed

    def _f(self, X, y=None):
        X = X.copy()
        key = X[INC].astype(str).to_frame()
        if y is not None:
            self.te = TargetEncoder(target_type='binary', smooth=20, cv=5, shuffle=True, random_state=SEED)
            X['income_te'] = self.te.fit_transform(key, y).ravel()
        else:
            X['income_te'] = self.te.transform(key).ravel()
        return X

    def fit(self, X, y):
        self.m = RealMLP_TD_Classifier(device='cuda', random_state=self.seed, n_epochs=self.n_epochs,
                                       batch_size=1024, verbosity=0).fit(self._f(X, y), y)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(self._f(X))
