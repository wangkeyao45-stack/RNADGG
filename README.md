# RNADGG: A Gradient-Guided Diffusion Framework for Functional RNA Sequence Design

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 📖 Overview
RNADGG (RNA Diffusion with Gradient Guidance) is a unified framework for end-to-end controllable sequence design[cite: 5]. [cite_start]By integrating a pre-trained diffusion backbone with differentiable gradient guidance, RNADGG effectively navigates sparse fitness landscapes and captures latent structural dependencies without requiring explicit secondary structure priors. 

Unlike traditional optimization strategies (e.g., genetic algorithms) that frequently suffer from mode collapse, RNADGG successfully balances functional maximization with biological viability, maintaining high sequence diversity even in rigorous multi-objective design scenarios.

## 🎯 Supported Benchmarks
The framework has been rigorously evaluated across three distinct biological regulatory mechanisms:
* Prokaryotic Ribosome Binding Sites (RBS): Context-sensitive translation initiation optimization.
* Eukaryotic 5' Untranslated Regions (5' UTRs): Sequence-driven ribosome loading enhancement.
* Toehold Switches: Complex, multi-state structural switching requiring multi-objective optimization (High ON / Low OFF).

## ⚙️ Architecture
RNADGG utilizes a three-stage interconnected methodology:
1.  Differentiable Fitness Proxy (Oracle): A 1D-CNN trained to map discrete sequences to continuous functional properties.
2.  Generative Backbone: A 1D U-Net diffusion model that captures underlying biological syntax via an iterative denoising process.
3.  Gradient-Guided Optimization: Dynamic injection of gradient signals during the reverse diffusion process to steer generation toward desired functional outcomes.

## 🚀 Quick Start
(Detailed instructions on how to setup the environment and run the code)

```bash
# Clone the repository
git clone [https://github.com/wangkeyao45-stack/RNADGG.git](https://github.com/wangkeyao45-stack/RNADGG.git)
cd RNADGG

# Install dependencies
pip install -r requirements.txt

#1. Train the Generative Backbone

# Example command for pre-training the diffusion model
python train_diffusion.py --dataset data/toehold.csv --channels 128

#2. Train the Functional Oracle

# Example command for training the surrogate predictor
python train_oracle.py --dataset data/toehold.csv

#3. Gradient-Guided Generation

# Example command for generating sequences with composite guidance
python generate.py --model_path checkpoints/unet.pt --oracle_path checkpoints/oracle.pt --guidance_scale 1.0
