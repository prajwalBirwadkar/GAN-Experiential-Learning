**GAN Experiential Learning Tasks**

This project involves training and evaluating three variants of Generative Adversarial Networks (GANs) on the CIFAR-10 dataset: BCE GAN, LS-GAN, and WGAN. The goal is to compare their performance using quantitative metrics (Inception Score and Fréchet Inception Distance) and analyze training stability.

📊 Results and Analysis
Training Dynamics
BCE GAN (Binary Cross-Entropy Loss)

Discriminator Loss: Starts at 1.79, fluctuates significantly (min: 0.09, max: 2.21).

Generator Loss: Ranges from 0.38 to 7.49, indicating instability and potential mode collapse.

Behavior: High volatility suggests difficulty in balancing generator and discriminator training.

LS-GAN (Least Squares Loss)

Discriminator Loss: Starts at 0.34 and stabilizes around 0.1–0.5.

Generator Loss: Ranges from 0.04 to 2.74, showing smoother convergence.

Behavior: More stable training compared to BCE GAN, with fewer extreme fluctuations.

WGAN (Wasserstein Loss with Weight Clipping)

Critic Loss: Highly unstable, oscillating between -3.35 and 2.49.

Generator Loss: Spikes to -14.5 and 16.9, indicating critic/generator imbalance.

Behavior: Severe instability likely due to aggressive weight clipping (clip_value=0.01) or insufficient critic updates (n_critic=5).

Evaluation Metrics
Model	Inception Score (↑)	FID Score (↓)
BCE GAN	4.75	720.10
LS-GAN	4.48	706.17
WGAN	2.25	1085.42
Key Observations:

LS-GAN achieves the best FID score, suggesting better image quality and diversity.

BCE GAN has a marginally higher IS but worse FID, indicating a trade-off between diversity and realism.

WGAN performs poorly, likely due to training instability. Adjusting hyperparameters (e.g., higher clip_value, more critic updates) may improve results.

🛠 Setup and Usage
Dependencies
Python 3.11+

PyTorch 2.5.1+ (CUDA 12.4)

torchmetrics, torchvision, numpy, scipy
