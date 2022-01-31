Code for paper [Margin-aware Adversarial Domain Adaptation with Optimal Transport](http://proceedings.mlr.press/v119/dhouib20b.html)

# Dependencies:
* Numpy >= 1.18.1
* POT >= 0.6.0
* CVXPY >= 1.0.25
* MOSEK >= 9.1.9
* Scikit-Learn >= 0.22.1

# Scripts for experiments:
* Moons:
	* cross validation: `cross_valid.py` (might take up to 90 minutes to run)
	* testing: `postprocessing_cross_valid.py`
* Amazon:
	* Data download [link](http://researchers.lille.inria.fr/pgermain/data/amazon_tfidf_svmlight.tgz)
	* testing: `postprocessing_cross_valid_amazon.py` (with **fixed hyperparameters** indicated in the main paper)

# Scripts for figures
* Moons: `postprocessing_cross_valid.py`
* Loss function <img src="https://render.githubusercontent.com/render/math?math=l^{\rho, \beta}">: `loss_funcs.py`
* Smooth proxies (supplementary): `proxies.py` (end of script, to decomment)

# Other scripts:
* Main class for our algorithm: `madaot.py`
* Cross validation (supports parallelism): `myDA.py`
* Algorithm computing the transport plan at each step (decribed in [Blankenship and Falk, 1976](https://link.springer.com/article/10.1007/BF00934096)): `advEmd.py`
