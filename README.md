# 🕊️ Bird vs Drone Classification

This repository contains code for machine learning models that attempt to classify images as either a **Bird** or **Drone**. The goal is to aid in surveillance systems, wildlife monitoring, and airspace security by accurately distinguishing between natural flying objects and artificial ones.

## 📌 Project Overview

With the increasing number of drones in the airspace, it is crucial to distinguish them from birds in automated monitoring systems. This project implements several neural network models as well as a fusion model to try to differentiate between:

- 🐦 Birds
- 🚁 Drones

## 🧠 Models

We used 5 models for this task:

- **Random Baseline:** Baseline model to simulate "random guessing".
- **Custom CNN:** A custom CNN architecture.
- **ResNet50**: ResNet50 model architecture.
- **InceptionV3:** InceptionV3 model architecture.
- **Fusion (XGBoost):** XGBoost model fit on features fused from multiple pretrained nueral network models.

