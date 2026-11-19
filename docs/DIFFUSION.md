# Gaussian diffusion in RNADGG

The public implementation uses a denoising diffusion probabilistic model (DDPM)
on continuous representations of one-hot encoded sequences. A function predictor
supplies gradients during generation. The relevant files are
[`diffusion.py`](../rnadgg/diffusion.py),
[`training.py`](../rnadgg/training.py) and
[`run_diffusion_rbs.py`](../scripts/run_diffusion_rbs.py).

## Forward process

For an encoded sequence `x0` with shape `(4, length)`, the code samples independent
standard Gaussian noise with `torch.randn_like(x0)`. The noise has population
mean 0 and standard deviation 1; empirical moments fluctuate in finite samples.
The noisy state at step `t` is

$$
x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
$$

Here, `alpha_t = 1 - beta_t` and `alpha_bar_t` is the product of alphas through
step `t`. The default schedule has 500 steps, with beta increasing linearly from
`1e-4` to `0.02`. `GaussianDiffusion.q_sample` samples the noisy state directly
at a requested step; training does not need to simulate all earlier steps.

The original 0/1 one-hot tensor has mean 0.25 when averaged over all four channels
and all sequence positions. Each individual channel's mean depends on nucleotide
composition. The input mean is distinct from the noise mean. Conditional on a
fixed sequence, the noisy state's mean is `sqrt(alpha_bar_t) * x0` and its
covariance is `(1 - alpha_bar_t) * I`. The noisy representation need not remain
within `[0, 1]` or have channels that sum to one.

## Training objective

`train_denoiser` draws a random timestep and Gaussian noise for each training
batch. The U-Net receives the noisy tensor and the timestep, and predicts the
added noise. Training minimizes

$$
\mathcal L = \mathbb E_{x_0,t,\epsilon}
\left\|\epsilon - \epsilon_\theta(x_t,t)\right\|^2.
$$

The network predicts noise. The reverse mean is derived from that prediction;
the reverse variance follows the fixed schedule and is not learned.

## Reverse process and guidance

Sampling starts from `torch.randn(shape)`. Without guidance, the reverse mean is

$$
\mu_\theta(x_t,t) = \frac{1}{\sqrt{\alpha_t}}
\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}
\epsilon_\theta(x_t,t)\right).
$$

Except at the final reverse step, `_reverse_step` adds fresh Gaussian noise with
standard deviation `sqrt(beta_t)`. At the final step it returns the mean directly.
Code timesteps are zero-indexed, so this final step is `t == 0`.

Guidance replaces the predicted noise by

$$
\hat\epsilon_t = \epsilon_\theta(x_t,t)
- \sqrt{1-\bar\alpha_t}\,\gamma\,g_t,
$$

where `g_t` is the predictor gradient clipped elementwise to `[-1, 1]` by default.
The predictor class is named `OracleCNN`. The per-sequence objectives are summed
before differentiation to keep the gradient scale independent of batch size.

These are DDPM noise-prediction and reverse-sampling updates. Flow matching
instead trains a vector field for a chosen probability path. The formulations
are mathematically related, and flow matching can also use Gaussian noise and
diffusion probability paths. Gaussian initialization alone does not distinguish
the two training methods.

## Nucleotide decoding and evaluation

`GaussianDiffusion.sample` returns a continuous tensor. `decode_one_hot` selects
the largest channel at each position. The RBS example re-encodes those nucleotide
strings before computing their reported predictor scores. The default alphabet
stores U in the T channel and exports A/C/G/T strings.

Continuous guidance and discrete decoding can favor different candidates;
argmax also discards uncertainty among channels. With the default finite
schedule, `alpha_bar_T` is approximately 0.00635, so the terminal state still
retains a signal coefficient of approximately 0.0797. Initialization from a
standard Gaussian is an approximation to this terminal distribution.

The generic `GaussianDiffusion` and compact RBS example do not apply scaffold
projection. The dedicated `benchmarks/toehold.py` driver projects the state at
initialization and every reverse step, differentiates through the tied stems,
and uses `gamma_t = gamma * t / T`. These settings apply to both guided and
unguided scaffold-constrained libraries. See [EXPERIMENTS.md](EXPERIMENTS.md).

## References

- Ho et al. (2020), [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239).
- Lipman et al. (2023), [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747).
