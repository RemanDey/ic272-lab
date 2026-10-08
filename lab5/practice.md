# Regularised Regression: An Olympiad-Calibre Treatment

**Scope:** feature scaling and conditioning, Ridge (Tikhonov) regression, the bias–variance trade-off with cross-validation, unit recovery (H1), coefficient paths (H2–H3), and Lasso vs Ridge geometry (H4–H5).

**How to use this guide.** Each section gives (i) the intuition, (ii) a rigorous derivation, (iii) the pitfalls the short notes gloss over. Section 8 has proof-style problems with solutions. Section 9 collects *corrections and sharpenings* of statements in the original notes: read it before an exam or viva.

---

## 0. Setup and notation

- Data: $n$ samples, $p$ raw features. Raw feature matrix $X\in\mathbb{R}^{n\times p}$, target $y\in\mathbb{R}^n$ (here: rent). In the lab, $n=40$, $p=13$.
- Model: $y = \theta_0\mathbf 1 + Z\beta + \varepsilon$, where $\varepsilon\sim(0,s^2 I)$ (noise variance $s^2$).
- Column means $\mu_j=\frac1n\sum_i X_{ij}$, standard deviations $\sigma_j=\sqrt{\frac1n\sum_i (X_{ij}-\mu_j)^2}$.
- Standardised matrix: $Z_{ij}=\dfrac{X_{ij}-\mu_j}{\sigma_j}$, so every column of $Z$ has mean $0$ and $\frac1n\|Z_{\cdot j}\|^2=1$.
- Singular value decomposition: $Z=U\Sigma V^\top$, with singular values $d_1\ge\dots\ge d_p>0$, right singular vectors $v_i$ (principal axes), left singular vectors $u_i$.
- Ridge penalty weight $\lambda\ge0$. I write $\beta$ for the *penalised* coefficients and reserve $\theta=(\theta_0,\beta)$ for the full vector in the scaled coordinate system.

---

## 1. The core pathology: anisotropy and ill-conditioning

### 1.1 The Hessian and the condition number

For the squared-error loss $L(\theta)=\tfrac12\|y-A\theta\|^2$ (with $A=[\mathbf 1\ \ X]$ or $[\mathbf 1\ \ Z]$):

$$\nabla L(\theta)=A^\top(A\theta-y),\qquad \nabla^2L(\theta)=A^\top A .$$

The Hessian is constant. Its eigenvalues $\lambda_{\max}\ge\dots\ge\lambda_{\min}>0$ are the squares of the singular values of $A$. Level sets of $L$ are ellipsoids

$$L(\theta)=L(\hat\theta_{\rm OLS})+\tfrac12(\theta-\hat\theta_{\rm OLS})^\top A^\top A(\theta-\hat\theta_{\rm OLS}),$$

with semi-axes of length proportional to $1/\sqrt{\lambda_i}$ along eigenvector $v_i$. The **condition number**

$$\kappa(A^\top A)=\frac{\lambda_{\max}}{\lambda_{\min}}=\kappa(A)^2$$

is exactly the squared aspect ratio of the longest to shortest axis of that ellipsoid.

**Why this matters (three separate reasons).**

1. **Numerical.** Solving $A^\top A\,\theta=A^\top y$ loses roughly $\log_{10}\kappa(A^\top A)$ digits of accuracy. Forming the Gram matrix *squares* the condition number, which is why QR/SVD solvers are preferred to the normal equations.
2. **Optimisation.** Gradient descent with the optimal fixed step $2/(\lambda_{\max}+\lambda_{\min})$ contracts the error by the factor
$$\rho=\frac{\kappa-1}{\kappa+1}\quad\text{per step},$$
so the iteration count to reach accuracy $\epsilon$ scales like $\tfrac{\kappa}{2}\ln\tfrac1\epsilon$. For $\kappa=10^{10}$ GD is hopeless.
3. **Statistical.** $\operatorname{Cov}(\hat\theta_{\rm OLS})=s^2(A^\top A)^{-1}$ has eigenvalues $s^2/\lambda_i$. The direction with the smallest $\lambda_i$ carries variance $s^2/\lambda_{\min}$: tiny data variation along $v_{\min}$ means the data barely constrains that direction.

### 1.2 How raw units create anisotropy

Suppose two uncorrelated, centred features: area in square feet (typical spread $\sim10^{2}$–$10^{3}$) and, say, a ratio in $[0,1]$ (spread $\sim10^{-1}$). Their Gram diagonal is about $n\sigma_j^2$, so with $\sigma_1=10^{3},\ \sigma_2=10^{-1}$,

$$\kappa\approx\frac{n\cdot10^{6}}{n\cdot10^{-2}}=10^{8}.$$

The ellipsoid is a needle of aspect ratio $10^4$, purely because of the *choice of units*. Nothing about the underlying relationship changed.

**General principle.** If feature $j$ is rescaled $x_j\mapsto c_jx_j$, i.e. $A\mapsto AD$ with $D=\operatorname{diag}(1,c_1,\dots,c_p)$, then

$$A^\top A\ \mapsto\ D\,(A^\top A)\,D .$$

The condition number can be changed arbitrarily by choosing $D$, but the *predictive content* of the model cannot (OLS fitted values are invariant to $D$). So the ill-conditioning is an artefact of coordinates, and we are free to choose better ones.

### 1.3 Why Ridge is *not* invariant (and why that is the real disaster)

OLS is **scale-equivariant**: if $x_j\mapsto c\,x_j$ then $\hat\beta_j\mapsto\hat\beta_j/c$ and predictions are unchanged. Ridge is not:

$$\min_\beta\ \|y-X\beta\|^2+\lambda\sum_j\beta_j^2 .$$

Replace column $j$ by $c\,x_j$ and reparametrise $\beta_j=\tilde\beta_j/c$ (same function). The loss term is identical, but the penalty becomes $\lambda\tilde\beta_j^2/c^2$ in terms of the new coefficient. In other words:

> A feature expressed in units $c$ times larger pays a penalty $c^{2}$ times *smaller* for exactly the same predictive effect.

So with raw units, Ridge silently favours whichever features happen to be measured in big numbers, and crushes those measured in small ones. The "shrinkage" would reflect your unit choice, not the data.

### 1.4 Standardisation: what it does and what it does *not* do

After standardising, $\frac1nZ^\top Z=R$, the **correlation matrix**:

- $R_{jj}=1$, so $\operatorname{tr}R=p$ and the *average* eigenvalue is exactly $1$;
- $\kappa(R)\ge1$, with equality iff the features are pairwise uncorrelated;
- each coefficient $\beta_j$ now means "change in target per one standard deviation of feature $j$", so all coefficients live on a **common, dimensionless-feature scale**, and $\lambda\sum\beta_j^2$ penalises them symmetrically.

**Precision point.** The notes call this an "affine whitening". Strictly:

- *Standardisation* = centring + **diagonal** rescaling. It equalises the marginal variances (the diagonal of the covariance) but leaves the **correlations** intact.
- *Whitening* = multiplying by $\Sigma_X^{-1/2}$, which makes the covariance exactly $I$ (so $\kappa=1$) but mixes features, so coefficients lose their meaning.

Standardisation is "equilibration" (also called Jacobi/diagonal preconditioning). A classical theorem (**van der Sluis, 1969**) says that scaling columns to *unit Euclidean norm* is near-optimal among all diagonal scalings:

$$\kappa_2(AD_{\rm unit})\ \le\ \sqrt{p}\ \min_{D}\kappa_2(AD).$$

For the Gram matrix ($\kappa$ squared) the loss is at most a factor $p$. So standardisation removes essentially *all* the conditioning problem that is attributable to units. What remains is genuine collinearity between features, which is Ridge's job (Section 2).

---

## 2. Ridge regression and Tikhonov regularisation

### 2.1 Closed form

$$\hat\beta(\lambda)=\arg\min_\beta\ \|y_c-Z\beta\|^2+\lambda\|\beta\|^2=(Z^\top Z+\lambda I)^{-1}Z^\top y_c,$$

where $y_c=y-\bar y\mathbf 1$ (explained in 2.4). Derivation: the objective is strictly convex for $\lambda>0$; the gradient $-2Z^\top(y_c-Z\beta)+2\lambda\beta=0$ gives the formula.

**Invertibility.** For $\lambda>0$ and any $Z$ (even $p>n$): $v^\top(Z^\top Z+\lambda I)v=\|Zv\|^2+\lambda\|v\|^2\ge\lambda\|v\|^2>0$. So the matrix is symmetric positive definite, hence invertible. OLS needs $Z$ to have full column rank; Ridge does not.

### 2.2 The spectral picture (the key formula)

Substitute $Z=U\Sigma V^\top$:

$$\boxed{\hat\beta(\lambda)=\sum_{i=1}^{p}\frac{d_i}{d_i^2+\lambda}\,(u_i^\top y_c)\,v_i}\qquad\text{vs. OLS: }\ \hat\beta_{\rm OLS}=\sum_i\frac{1}{d_i}(u_i^\top y_c)\,v_i .$$

So Ridge multiplies the $i$-th OLS component by the **shrinkage factor**

$$f_i(\lambda)=\frac{d_i^2}{d_i^2+\lambda}\in(0,1).$$

- Directions with $d_i^2\gg\lambda$ (well-determined by the data): $f_i\approx1$, almost untouched.
- Directions with $d_i^2\ll\lambda$ (poorly determined): $f_i\approx d_i^2/\lambda\to0$, almost deleted.

Ridge is a **soft, data-adaptive truncation of the SVD**. This is the precise sense in which "adding $\lambda I$ lifts the small eigenvalues": every eigenvalue $d_i^2$ of $Z^\top Z$ becomes $d_i^2+\lambda$, and the condition number of the matrix that is inverted becomes

$$\kappa_\lambda=\frac{d_1^2+\lambda}{d_p^2+\lambda}\ \xrightarrow{\ \lambda\to\infty\ }\ 1 .$$

(Strictly decreasing in $\lambda$: see Problem 2.)

### 2.3 Bias, variance, and effective degrees of freedom

With $y=Z\beta^\star+\varepsilon$, $\operatorname{Cov}(\varepsilon)=s^2I$:

$$\mathbb E\hat\beta(\lambda)=V\,\operatorname{diag}\!\Big(\tfrac{d_i^2}{d_i^2+\lambda}\Big)V^\top\beta^\star,\qquad
\text{Bias}=-\lambda\,V\operatorname{diag}\!\Big(\tfrac{1}{d_i^2+\lambda}\Big)V^\top\beta^\star,$$

$$\operatorname{Cov}\hat\beta(\lambda)=s^2\,V\operatorname{diag}\!\Big(\tfrac{d_i^2}{(d_i^2+\lambda)^2}\Big)V^\top .$$

**Variance is tamed uniformly.** By AM–GM, $(d^2+\lambda)^2\ge4\lambda d^2$, hence

$$\frac{d^2}{(d^2+\lambda)^2}\le\frac1{4\lambda}\quad\text{for every }d,$$

whereas OLS has $1/d^2$, which is unbounded as $d\to0$. This inequality is the quantitative statement that Ridge "damps variance".

**Effective degrees of freedom.** The fitted values are $\hat y=H_\lambda y_c$ with $H_\lambda=Z(Z^\top Z+\lambda I)^{-1}Z^\top$ and

$$\mathrm{df}(\lambda)=\operatorname{tr}H_\lambda=\sum_{i=1}^p\frac{d_i^2}{d_i^2+\lambda},$$

which decreases continuously from $p$ (at $\lambda=0$) to $0$ (as $\lambda\to\infty$). This is the correct "model complexity" axis for Ridge; the number of parameters is always $p$.

### 2.4 Why the intercept is exempt (a real proof, not a slogan)

Minimise over $(\theta_0,\beta)$ the objective $\sum_i(y_i-\theta_0-z_i^\top\beta)^2+\lambda\|\beta\|^2$ (intercept *not* penalised).

Setting $\partial/\partial\theta_0=0$: $\ \theta_0=\bar y-\bar z^\top\beta$. For standardised features $\bar z=0$, so

$$\theta_0=\bar y\quad\text{exactly,}$$

and the problem decouples: $\beta$ is the Ridge solution for the centred target, as in 2.1.

**What goes wrong if you penalise $\theta_0$.** With centred features, the column $\mathbf 1$ is orthogonal to every column of $Z$, so the problem separates and

$$\theta_0^{\rm pen}=\frac{n}{n+\lambda}\,\bar y .$$

The baseline rent is dragged toward $0$ by the factor $n/(n+\lambda)$.

The deeper reason is **translation equivariance**: if every target value is shifted, $y\mapsto y+c\mathbf 1$, a sensible model must shift all predictions by $c$ and leave the slopes alone. With an unpenalised intercept, $\theta_0\mapsto\theta_0+c$ and $\beta$ is unchanged. With a penalised one, this breaks. In matrix form the penalty is $\lambda\,\theta^\top D\theta$ with $D=\operatorname{diag}(0,1,\dots,1)$, which is the "$0$ at position $(0,0)$" in the notes.

### 2.5 Three equivalent views of Ridge

**(a) Penalised ⇔ constrained.** For each $\lambda>0$, $\hat\beta(\lambda)$ solves
$$\min_\beta\|y_c-Z\beta\|^2\quad\text{s.t.}\quad\|\beta\|^2\le C(\lambda),\qquad C(\lambda)=\|\hat\beta(\lambda)\|^2 .$$
(Lagrangian duality; $\lambda$ is the multiplier.) $C(\lambda)$ is a strictly decreasing continuous bijection $(0,\infty)\to(0,\|\hat\beta_{\rm OLS}\|^2)$ (Problem 3), so every budget corresponds to exactly one $\lambda$.

**(b) Tangency condition.** The KKT condition $Z^\top(Z\hat\beta-y_c)=-\lambda\hat\beta$ says: the gradient of the loss at the solution is parallel to $\hat\beta$, i.e. normal to the sphere. The solution is where the **error ellipsoid first touches the ball**.

**(c) Bayesian (MAP).** If $y\mid\beta\sim\mathcal N(Z\beta,s^2I)$ and $\beta\sim\mathcal N(0,\tau^2I)$, the posterior mode is Ridge with
$$\lambda=\frac{s^2}{\tau^2}.$$
Large $\lambda$ = confident prior that coefficients are small. (Lasso corresponds to a Laplace prior.)

**(d) Data augmentation.** Ridge is *ordinary* least squares on augmented data
$$\tilde Z=\begin{bmatrix}Z\\ \sqrt\lambda\,I_p\end{bmatrix},\qquad\tilde y=\begin{bmatrix}y_c\\ 0\end{bmatrix}:$$
$\|\tilde y-\tilde Z\beta\|^2=\|y_c-Z\beta\|^2+\lambda\|\beta\|^2$. Hence it is "$p$ fake observations that say each coefficient is $0$".

### 2.6 Geometry: precise version of "pulled radially inward"

The notes say Ridge pulls coefficients "radially inward toward the origin". Not exactly. In the principal-axis coordinates of the ellipse, component $i$ is scaled by $f_i(\lambda)=d_i^2/(d_i^2+\lambda)$, **different for each axis**. Components along *long* axes of the ellipse (small $d_i$) shrink faster than those along short axes. So the Ridge path $\lambda\mapsto\hat\beta(\lambda)$ is a **curve that bends**, starting at $\hat\beta_{\rm OLS}$ and ending at the origin, tangent to the first principal axis direction with the largest $d$ at the end. The tangency picture (the ellipse touching the ball) is right; "radially" is only exact when $Z^\top Z\propto I$.

---

## 3. The bias–variance landscape and cross-validation

### 3.1 Decomposition

For a new point $x_0$ with target $y_0=f(x_0)+\varepsilon_0$, and an estimator $\hat f$ trained on random data:

$$\mathbb E\big[(y_0-\hat f(x_0))^2\big]=\underbrace{s^2}_{\text{irreducible}}+\underbrace{\big(\mathbb E\hat f(x_0)-f(x_0)\big)^2}_{\text{bias}^2}+\underbrace{\operatorname{Var}\hat f(x_0)}_{\text{variance}} .$$

As $\lambda$ increases: bias$^2$ increases from $\approx0$, variance decreases from its (possibly huge) OLS value. Their sum is U-shaped, with a minimum at some $\lambda^\ast>0$.

**Why a positive $\lambda^\ast$ always beats OLS (Hoerl–Kennard).** In the orthonormal case $Z^\top Z=I$ (write coordinates $z_j=\hat\beta^{\rm OLS}_j\sim\mathcal N(\beta^\star_j,s^2)$): $\hat\beta_j(\lambda)=a\,z_j$ with $a=1/(1+\lambda)$, and

$$\operatorname{MSE}(a)=\sum_j\big[(1-a)^2\beta_j^{\star2}+a^2s^2\big]=(1-a)^2\|\beta^\star\|^2+a^2ps^2 .$$

At $a=1$ ($\lambda=0$) the derivative with respect to $a$ is $2ps^2>0$: decreasing $a$ slightly below $1$ *strictly reduces* the MSE. In general, there is **always** some $\lambda>0$ with lower MSE than OLS. Minimising over $a$ gives $a^\ast=\dfrac{\|\beta^\star\|^2}{\|\beta^\star\|^2+ps^2}$, i.e.

$$\lambda^\ast=\frac{p\,s^2}{\|\beta^\star\|^2}.$$

(Oracle quantity: it requires the unknown $\beta^\star$. That is why we cross-validate.)

### 3.2 Training error is monotone, hence useless for choosing $\lambda$

Let $R(\beta)=\|y_c-Z\beta\|^2$, $P(\beta)=\|\beta\|^2$, and $\beta_1,\beta_2$ the minimisers for $\lambda_1<\lambda_2$. Optimality of each for its own objective:

$$R(\beta_1)+\lambda_1P(\beta_1)\le R(\beta_2)+\lambda_1P(\beta_2),\qquad R(\beta_2)+\lambda_2P(\beta_2)\le R(\beta_1)+\lambda_2P(\beta_1).$$

Adding: $(\lambda_2-\lambda_1)\big(P(\beta_2)-P(\beta_1)\big)\le0$, so **$P$ is non-increasing in $\lambda$**. Then from the first inequality, $R(\beta_2)-R(\beta_1)\ge\lambda_1\big(P(\beta_1)-P(\beta_2)\big)\ge0$, so **training error is non-decreasing in $\lambda$**. Training RSS is therefore minimised at $\lambda=0$, which is exactly the overfit solution.

The optimism of training error for a linear smoother is quantified by (fixed-design, in-sample)

$$\mathbb E[\mathrm{Err}_{\rm in}]=\mathbb E[\mathrm{err}_{\rm train}]+\frac{2s^2}{n}\,\mathrm{df}(\lambda),$$

so a model with large $\mathrm{df}$ (small $\lambda$) systematically *under*-reports its true error.

### 3.3 $k$-fold cross-validation

Partition the data into $k$ folds. For each $\lambda$ on a **log-spaced grid** (e.g. $10^{-3}\ldots10^{3}$):

$$\mathrm{CV}(\lambda)=\frac1n\sum_{f=1}^{k}\sum_{i\in F_f}\big(y_i-\hat y_i^{(-f)}(\lambda)\big)^2,$$

where $\hat y^{(-f)}$ comes from a model trained on the other $k-1$ folds only. Pick $\lambda^\ast=\arg\min\mathrm{CV}$. Report $\sqrt{\mathrm{CV}}$ as RMSE.

- **Left side of the curve** (small $\lambda$): high variance; fold-to-fold predictions are unstable.
- **Right side** (large $\lambda$): high bias; the model is too stiff to fit real structure.
- CV is *nearly* unbiased for the error of a model trained on $\frac{k-1}{k}n$ points: slightly pessimistic (smaller training sets), and its own variance is large at $n=40$.

### 3.4 The mistakes that silently ruin the experiment

1. **Leakage through standardisation.** $\mu_j,\sigma_j$ must be computed from the *training folds only*, then applied to the held-out fold. Standardising the full data before splitting lets validation information flow into training and makes CV optimistic.
2. **Tuning and reporting on the same CV.** The minimum of a noisy curve is itself optimistically biased. Use a separate test set or *nested* CV for the final number.
3. **Linear grid.** $\lambda$ acts multiplicatively on eigenvalues, so use a geometric grid.
4. **Over-trusting the exact minimiser.** With $n=40$ the CV curve is flat near its minimum. The **one-standard-error rule** (pick the largest $\lambda$ whose CV error is within one SE of the minimum) yields a simpler, more stable model.
5. **Re-randomising folds.** Fix the fold assignment across all $\lambda$ so comparisons are paired.

### 3.5 Leave-one-out in closed form

Because Ridge is a linear smoother $\hat y=Hy$ (with $H=A(A^\top A+\lambda D)^{-1}A^\top$ for the full augmented design with unpenalised intercept), deleting observation $i$ gives, by the Sherman–Morrison identity,

$$y_i-\hat y_i^{(-i)}=\frac{y_i-\hat y_i}{1-H_{ii}},\qquad\mathrm{LOOCV}(\lambda)=\frac1n\sum_i\Big(\frac{y_i-\hat y_i}{1-H_{ii}}\Big)^2 .$$

No refitting needed. (Again: this version holds the feature scaling fixed; if you re-standardise inside each deletion it is no longer exact.)

---

## 4. Homework H1: unit recovery and dimensional invariance

### 4.1 The identity

The model fitted in scaled space is
$$\hat y=\theta_0+\sum_j\theta_j\,\frac{x_j-\mu_j}{\sigma_j}.$$
Expand and collect terms in the *raw* $x_j$:

$$\hat y=\Big(\theta_0-\sum_j\frac{\theta_j\mu_j}{\sigma_j}\Big)+\sum_j\frac{\theta_j}{\sigma_j}\,x_j
\ \Longrightarrow\ \boxed{w_j=\frac{\theta_j}{\sigma_j},\qquad b=\theta_0-\sum_j\frac{\theta_j\mu_j}{\sigma_j}}$$

**Dimensional check.** If $x_j$ has units $[x_j]$ and rent has units $[y]$, then $w_j$ must have units $[y]/[x_j]$. Indeed $\theta_j$ is in $[y]$ (target change per one SD) and $\sigma_j$ in $[x_j]$, so $\theta_j/\sigma_j$ has units $[y]/[x_j]$. ✓.

**Invariance logic.** If $x_j\to c\,x_j$ then $\sigma_j\to c\sigma_j$, $\mu_j\to c\mu_j$, but $z_j$ is *unchanged*. So $\theta$ is unit-free, and $w_j\to w_j/c$ automatically as it must.

### 4.2 A subtlety the notes slip past

The notes claim that regularised models in scaled space are "mathematically identical to models operating directly on raw physical units". Be careful:

- The recovered $(w,b)$ defines **exactly the same prediction function** as $(\theta_0,\beta)$ (an identity of functions).
- But $(w,b)$ is **not** the minimiser of the raw-unit Ridge objective $\|y-b-Xw\|^2+\lambda\|w\|^2$. In raw units that penalty has the unit-dependence of Section 1.3. The recovered $w$ minimises instead the **weighted** penalty $\lambda\sum_j\sigma_j^2w_j^2$.

So: unit recovery makes the model *interpretable* in physical units; it does not turn standardised Ridge into raw Ridge.

---

## 5. Homework H2 and H3: coefficient paths and noise rejection

Plot $\hat\beta_j(\lambda)$ versus $\log\lambda$ for all $p$ coefficients.

### 5.1 What the picture should show

- At $\lambda\to0$: the OLS values. At $\lambda\to\infty$: every curve $\to0$ (each is $\propto1/\lambda$ asymptotically, since $\hat\beta\approx Z^\top y_c/\lambda$).
- **Real features** carry large $|\hat\beta_j|$ for a wide range of $\lambda$ because the signal $z_j^\top y_c$ they capture is large; **noise features** start near $0$ and stay near $0$.

### 5.2 The exact statement in the orthonormal case

If $\frac1nZ^\top Z=I$, then $\hat\beta_j(\lambda)=\dfrac{n}{n+\lambda}\,\hat\beta_j^{\rm OLS}$.

All coefficients are shrunk **by the same factor**. Noise coefficients are small *not* because Ridge treats them differently, but because their OLS values already are, $\hat\beta^{\rm OLS}_j\sim\mathcal N(0,s^2/n)$ for a pure-noise column, while real coefficients are $\beta^\star_j+\mathcal N(0,s^2/n)$. So Ridge separates signal from noise only through **magnitude**, never by setting anything to zero.

### 5.3 Correlated features: paths can misbehave

When columns of $Z$ are correlated, Ridge paths are *not* necessarily monotone in absolute value. Example: two nearly identical features with true coefficients $+1$ and $-1$ (they nearly cancel). OLS returns huge values of opposite sign; as $\lambda$ increases these collapse quickly, but intermediate $\lambda$ can make one coefficient *cross zero* or even *grow* while the other shrinks, because the combined fit must be preserved. Coefficient paths are thus a *diagnostic of collinearity*, not simply of "importance".

**Reading guide for the figure.**
1. Where do blue (real) curves separate from grey (noise)? That is where Ridge starts to be informative.
2. Do any curves cross or reverse? Suspect collinearity.
3. The $\lambda^\ast$ from CV should sit where noise curves are already near $0$ but real curves are still sizeable.

---

## 6. Homework H4 and H5: Lasso vs Ridge

### 6.1 The two problems

$$\text{Ridge: }\ \min_\beta\ \tfrac12\|y_c-Z\beta\|^2+\tfrac\lambda2\|\beta\|_2^2,\qquad
\text{Lasso: }\ \min_\beta\ \tfrac12\|y_c-Z\beta\|^2+\lambda\|\beta\|_1 .$$

### 6.2 Exact solution in the orthonormal case: shrinkage vs thresholding

Let $Z^\top Z=I$ and $z_j=Z_{\cdot j}^\top y_c$ (the OLS coefficient). The problem separates into scalar problems $\min_b\ \tfrac12(z-b)^2+\lambda|b|$. Subdifferential optimality: $0\in b-z+\lambda\,\partial|b|$.

- If $b>0$: $b=z-\lambda$, valid iff $z>\lambda$.
- If $b<0$: $b=z+\lambda$, valid iff $z<-\lambda$.
- If $b=0$: need $0\in-z+\lambda[-1,1]$, i.e. $|z|\le\lambda$.

$$\boxed{\hat\beta^{\rm lasso}_j=S_\lambda(z_j)=\operatorname{sign}(z_j)\,(|z_j|-\lambda)_+},\qquad
\hat\beta^{\rm ridge}_j=\frac{z_j}{1+\lambda}.$$

Ridge multiplies by a constant (never reaching $0$); Lasso **subtracts a constant and clips at $0$**. Any coefficient with $|z_j|\le\lambda$ becomes **exactly** zero: a *dead zone* of width $2\lambda$.

### 6.3 General KKT conditions (any design)

$\hat\beta$ is a Lasso solution iff, with residual $r=y_c-Z\hat\beta$:

$$Z_{\cdot j}^\top r=\lambda\operatorname{sign}(\hat\beta_j)\ \ (\hat\beta_j\ne0),\qquad |Z_{\cdot j}^\top r|\le\lambda\ \ (\hat\beta_j=0).$$

Consequences:
- A feature is *excluded* iff its correlation with the current residual is at most $\lambda$.
- At $\hat\beta=0$, $r=y_c$, so everything is zero iff $\lambda\ge\lambda_{\max}=\max_j|Z_{\cdot j}^\top y_c|$. This gives the natural top of the $\lambda$ grid.
- Active coefficients have *equal absolute correlation* $\lambda$ with the residual (basis of the LARS algorithm).

### 6.4 Why corners produce zeros: a precise geometric argument

Constrained form: minimise the loss over $\{\|\beta\|_1\le C\}$. At an optimum on the boundary, $-\nabla L(\hat\beta)$ must lie in the **normal cone** of the constraint set at $\hat\beta$.

- **$\ell_2$ ball.** At every boundary point the normal cone is a *single ray* (the radial direction). The gradient must point exactly radially. For a generic ellipse centred at a generic point, the tangency point has all coordinates non-zero with probability $1$; landing exactly on an axis is a measure-zero coincidence.
- **$\ell_1$ ball (cross-polytope).** At the vertex $Ce_j$ the normal cone is
$$N=\{t\,(s_1,\dots,s_p):\ t\ge0,\ s_j=1,\ |s_k|\le1\ (k\ne j)\}$$
which has **non-empty interior** (it is full-dimensional). So a *positive-probability set* of ellipse positions have their optimum at that vertex, where $p-1$ coordinates are zero. The same holds (with lower-dimensional cones) on the edges and faces, which produce partially sparse solutions.

> Sparsity is not a numerical accident; it is the consequence of the constraint set having **non-differentiable, full-dimensional-cone points on the coordinate subspaces**.

### 6.5 Honest caveats about Lasso

1. **Bias on large coefficients.** Soft-thresholding shrinks *every* surviving coefficient by $\lambda$, even clearly real ones (relaxed Lasso / adaptive Lasso fix this).
2. **At most $\min(n,p)$ nonzeros.** With $n=40$ that is not binding for $p=13$, but it matters when $p>n$.
3. **Correlated groups.** If several features are near-duplicates, Lasso tends to pick one arbitrarily and zero the rest; selection can flip between resamples. Ridge instead spreads weight across the group. (Elastic net, $\lambda_1\|\beta\|_1+\lambda_2\|\beta\|^2$, interpolates.)
4. **Exact support recovery** needs conditions (the *irrepresentable condition*): signal features must not be too correlated with noise features. With $n=40$ do not expect perfect noise rejection.
5. **Uniqueness.** The fitted values $Z\hat\beta$ are unique, but $\hat\beta$ need not be if columns are linearly dependent.
6. **Standardisation matters even more** for Lasso: the $\ell_1$ penalty compares raw magnitudes of coefficients, so unscaled data distorts *which* features survive.

### 6.6 Summary table

| Property | Ridge ($\ell_2$) | Lasso ($\ell_1$) |
|---|---|---|
| Orthonormal solution | $z/(1+\lambda)$ (proportional shrink) | $\operatorname{sign}(z)(|z|-\lambda)_+$ (soft threshold) |
| Exact zeros | Never (for $\lambda<\infty$) | Yes, for $|z|\le\lambda$ |
| Closed form (general) | Yes | No (convex program; coordinate descent / LARS) |
| Correlated features | Shares weight | Picks one, unstable |
| Bayesian prior | Gaussian | Laplace |
| Constraint geometry | Smooth ball, ray normal cone | Polytope, full-dim normal cones at vertices |
| Handles $p>n$ | Yes | Yes (≤ $n$ nonzeros) |

---

## 7. Reference implementation (NumPy)

```python
import numpy as np

def ridge_fit(X, y, lam):
    """Standardise on THIS data, fit ridge with unpenalised intercept,
    return raw-unit coefficients (w, b)."""
    mu, sd = X.mean(0), X.std(0, ddof=0)
    Z = (X - mu) / sd
    ybar = y.mean()
    p = Z.shape[1]
    beta = np.linalg.solve(Z.T @ Z + lam * np.eye(p), Z.T @ (y - ybar))
    theta0 = ybar                       # exact, since Z is centred
    w = beta / sd                       # H1: w_j = theta_j / sigma_j
    b = theta0 - np.sum(beta * mu / sd) # H1: b = theta_0 - sum theta_j mu_j / sigma_j
    return w, b

def cv_rmse(X, y, lam, k=5, seed=0):
    n = len(y)
    idx = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(idx, k)
    sse = 0.0
    for f in folds:
        tr = np.setdiff1d(idx, f)
        w, b = ridge_fit(X[tr], y[tr], lam)   # scaling learned on training folds only
        sse += np.sum((y[f] - (X[f] @ w + b)) ** 2)
    return np.sqrt(sse / n)

# lams = np.logspace(-3, 3, 50)
# curve = [cv_rmse(X, y, l) for l in lams]
```

---

## 8. Proof problems with solutions

**Problem 1 (scale non-invariance).** Show OLS fitted values are invariant under $X\mapsto XD$ for invertible diagonal $D$, but Ridge fitted values are not.
*Solution.* OLS: the column space of $XD$ equals that of $X$, and $\hat y$ is the orthogonal projection of $y$ onto it. Ridge: $\hat y_D=XD(DX^\top XD+\lambda I)^{-1}DX^\top y$. If $\hat y_D=\hat y_I$ for all $y$ we would need $D(D X^\top XD+\lambda I)^{-1}D=(X^\top X+\lambda I)^{-1}$, i.e. $X^\top X+\lambda D^{-2}=X^\top X+\lambda I$, so $D^2=I$. Hence invariance fails unless $D=\pm I$. $\square$

**Problem 2 (conditioning monotone in $\lambda$).** With eigenvalues $a>b>0$, show $\frac{a+\lambda}{b+\lambda}$ is strictly decreasing in $\lambda\ge0$ and tends to $1$.
*Solution.* $\frac{d}{d\lambda}\frac{a+\lambda}{b+\lambda}=\frac{(b+\lambda)-(a+\lambda)}{(b+\lambda)^2}=\frac{b-a}{(b+\lambda)^2}<0$. Also $\frac{a+\lambda}{b+\lambda}=1+\frac{a-b}{b+\lambda}\to1$. $\square$

**Problem 3 (norm monotone).** Show $\|\hat\beta(\lambda)\|^2$ is strictly decreasing in $\lambda$ (when $Z^\top y_c\ne0$).
*Solution.* $\|\hat\beta(\lambda)\|^2=\sum_i\frac{d_i^2c_i^2}{(d_i^2+\lambda)^2}$ with $c_i=u_i^\top y_c$. Each term with $c_i\neq0$ is strictly decreasing in $\lambda$, and at least one $c_i\ne0$. $\square$

**Problem 4 (augmentation).** Prove Ridge equals OLS on $\tilde Z=\begin{bmatrix}Z\\\sqrt\lambda I\end{bmatrix}$, $\tilde y=\begin{bmatrix}y_c\\0\end{bmatrix}$.
*Solution.* OLS normal equations: $\tilde Z^\top\tilde Z\beta=\tilde Z^\top\tilde y$. Compute $\tilde Z^\top\tilde Z=Z^\top Z+\lambda I$ and $\tilde Z^\top\tilde y=Z^\top y_c$. $\square$

**Problem 5 (minimum-norm limit).** If $p>n$ (so $Z^\top Z$ is singular), show $\lim_{\lambda\to0^+}\hat\beta(\lambda)=Z^+y_c$, the minimum-norm interpolating solution.
*Solution.* In SVD form with $r=\operatorname{rank}Z$ nonzero singular values, $\hat\beta(\lambda)=\sum_{i\le r}\frac{d_i}{d_i^2+\lambda}c_iv_i\to\sum_{i\le r}\frac{c_i}{d_i}v_i=Z^+y_c$. Components along the null space of $Z$ are never excited, so the limit has the smallest norm among all least-squares solutions. $\square$

**Problem 6 (optimal $\lambda$ in the orthonormal model).** For $Z^\top Z=I$, $\hat\beta=\frac{1}{1+\lambda}z$, $z\sim\mathcal N(\beta^\star,s^2I_p)$, find the $\lambda$ minimising $\mathbb E\|\hat\beta-\beta^\star\|^2$.
*Solution.* See §3.1: with $a=1/(1+\lambda)$, $\mathrm{MSE}=(1-a)^2\|\beta^\star\|^2+a^2ps^2$. Setting the derivative to zero, $a^\ast=\frac{\|\beta^\star\|^2}{\|\beta^\star\|^2+ps^2}$, so $\lambda^\ast=\frac{ps^2}{\|\beta^\star\|^2}$. At this value MSE $=\frac{ps^2\|\beta^\star\|^2}{\|\beta^\star\|^2+ps^2}<ps^2$, the OLS risk. $\square$

**Problem 7 (intercept shrinkage).** For centred features, show that penalising the intercept yields $\theta_0=\frac{n}{n+\lambda}\bar y$, and that this violates $y\mapsto y+c$ equivariance.
*Solution.* Since $\mathbf 1\perp Z_{\cdot j}$, the normal equations decouple: $(n+\lambda)\theta_0=\mathbf 1^\top y=n\bar y$. Under $y\to y+c\mathbf 1$: $\theta_0\to\frac{n}{n+\lambda}(\bar y+c)$, a shift of $\frac{n}{n+\lambda}c\ne c$. $\square$

**Problem 8 (leave-one-out shortcut).** Prove $y_i-\hat y_i^{(-i)}=\frac{y_i-\hat y_i}{1-H_{ii}}$ for a penalised least-squares fit with hat matrix $H$.
*Solution.* Let $\hat y^{(-i)}$ be the fit without observation $i$ and consider the modified data $y^\ast$ where $y_i$ is replaced by $\hat y_i^{(-i)}$. The fit on $y^\ast$ (all $n$ points) equals $\hat y^{(-i)}$: the extra point lies exactly on the fitted function, so adding it changes neither the loss-minimiser (a point with zero residual contributes zero gradient). Hence $\hat y^{(-i)}_i=\sum_jH_{ij}y_j^\ast=\hat y_i+H_{ii}(\hat y_i^{(-i)}-y_i)$. Rearranging, $\hat y^{(-i)}_i(1-H_{ii})=\hat y_i-H_{ii}y_i$, i.e. $y_i-\hat y_i^{(-i)}=\frac{y_i-\hat y_i}{1-H_{ii}}$. $\square$

**Problem 9 (Lasso zero set).** Show that for the Lasso with centred data, $\hat\beta=0$ iff $\lambda\ge\|Z^\top y_c\|_\infty$.
*Solution.* From §6.3 at $\beta=0$ the residual is $y_c$, and all coordinates being zero requires $|Z_{\cdot j}^\top y_c|\le\lambda$ for all $j$. $\square$

**Problem 10 (why $\ell_1$ ball vertices attract).** In $\mathbb R^2$ with constraint $|\beta_1|+|\beta_2|\le1$, find all gradient directions $g=-\nabla L$ for which the optimum is the vertex $(1,0)$.
*Solution.* Normal cone at $(1,0)$ is spanned by outward normals of the two adjacent edges, $(1,1)$ and $(1,-1)$. So the optimum is at $(1,0)$ iff $g=\alpha(1,1)+\beta(1,-1)$ with $\alpha,\beta\ge0$, i.e. $g_1\ge|g_2|$: an angular sector of $90^\circ$ out of $360^\circ$, a positive fraction. For the circle every boundary point has a single normal direction, so landing on an axis requires $g_2=0$ exactly. $\square$

---

## 9. Corrections and sharpenings of the original notes

| Original statement | Sharper version |
|---|---|
| Standardisation is "affine whitening" | It is **diagonal** rescaling (equilibration), which fixes unit effects but not correlations; true whitening needs $\Sigma^{-1/2}$ and destroys feature identity. |
| Ridge pulls coefficients "radially" toward the origin | Component $i$ shrinks by $d_i^2/(d_i^2+\lambda)$, **axis-dependent**, so the path is curved. Tangency-to-the-ball is the correct statement. |
| Small samples (40 rows vs 13 features) make $A^\top A$ nearly singular | Near-singularity comes from small $d_{\min}$, i.e. **collinearity / low variance directions**; low $n/p$ makes it likelier. $n>p$ still gives a full-rank matrix. |
| Scaled-space models are "identical" to raw-unit models | Identical as **prediction functions** after recovery; not the minimiser of the raw-unit Ridge objective (it carries the weighted penalty $\lambda\sum\sigma_j^2w_j^2$). |
| Noise features "collapse rapidly" under Ridge | In the orthonormal case all coefficients shrink at the **same rate**; noise coefficients are simply small to begin with. With correlated features, paths can be non-monotone. |
| Training error is monotone in complexity | True; precisely, it is non-decreasing in $\lambda$ (proof in §3.2), and its optimism is $2s^2\mathrm{df}(\lambda)/n$. |
| CV is an "unbiased proxy" | Nearly unbiased for training size $\frac{k-1}{k}n$ (slightly pessimistic); high variance for $n=40$; must standardise **inside** folds. |
| Lasso gives true zeros because of corners | Correct; formalised via normal cones (§6.4) and soft thresholding / KKT (§6.2–6.3). Add: selection is unstable under correlation, and survivors are biased. |

---

## 10. One-page cheat sheet

- **Condition number** $\kappa(A^\top A)=\lambda_{\max}/\lambda_{\min}$; GD rate $\frac{\kappa-1}{\kappa+1}$; units change $\kappa$ via $A\to AD$.
- **Standardise** with training-fold statistics only; correlation matrix has unit diagonal.
- **Ridge:** $\hat\beta=(Z^\top Z+\lambda I)^{-1}Z^\top y_c=\sum\frac{d_i}{d_i^2+\lambda}(u_i^\top y_c)v_i$; $\mathrm{df}=\sum\frac{d_i^2}{d_i^2+\lambda}$; variance factor $\le\frac1{4\lambda}$.
- **Intercept:** unpenalised, $\theta_0=\bar y$ for standardised $Z$.
- **Training error** non-decreasing in $\lambda$; choose $\lambda$ by CV on a log grid; consider the 1-SE rule.
- **Unit recovery:** $w_j=\theta_j/\sigma_j$, $b=\theta_0-\sum\theta_j\mu_j/\sigma_j$.
- **Orthonormal case:** Ridge $\frac{z}{1+\lambda}$; Lasso $\operatorname{sign}(z)(|z|-\lambda)_+$.
- **Lasso KKT:** $|Z_j^\top r|\le\lambda$ for zeros, $=\lambda\operatorname{sign}$ for actives; $\lambda_{\max}=\|Z^\top y_c\|_\infty$.
- **Sparsity geometry:** $\ell_1$ vertices have full-dimensional normal cones; the $\ell_2$ ball has single-ray normals.