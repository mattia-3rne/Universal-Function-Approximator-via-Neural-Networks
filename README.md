# Universal Function Approximation via Neural Network

## Project Overview

The goal of this project is to demonstrate the **Universal Approximation Theorem** by constructing a Neural Network from scratch. This implementation proves that a feedforward neural network with a single hidden layer can approximate any continuous function (such as $\sin(x)$ or $x^2$) given sufficient neurons and appropriate training.

The project demonstrates the mathematics behind neural networks without the abstraction of high-level libraries like PyTorch or TensorFlow.

---

## Theoretical Background

### The Universal Approximation Theorem

The Universal Approximation Theorem states that a feedforward network with a single hidden layer containing a finite number of neurons can approximate continuous functions on compact subsets of $\mathbb{R}^n$, under certain assumptions on the activation function.

Formally, let $\sigma(z)$ be a non-constant, bounded, and continuous activation function. Let $I_m$ denote the $m$-dimensional unit hypercube $[0, 1]^m$. The space of continuous functions on $I_m$ is denoted by $C(I_m)$.

Given any function $f \in C(I_m)$ and error tolerance $\varepsilon > 0$, there exists an integer $N$ (number of neurons) and parameters $v_i, b_i, w_i$ such that the function $g(x)$:

$$
g(x) = \sum_{i=1}^{N} v_i \sigma(w_i^T x + b_i)
$$

satisfies:

$$
\vert g(x) - f(x) \vert < \varepsilon \quad \forall x \in I_m
$$

### Activation Functions

Non-linearity is crucial for the network to learn complex patterns. Three primary activation functions are utilized. The choice of activation affects the gradient flow during backpropagation.

| Function | Formula $\sigma(z)$ | Derivative $\sigma'(z)$ | Characteristics                              |
| :--- | :--- | :--- |:---------------------------------------------|
| **Sigmoid** | $\frac{1}{1 + e^{-z}}$ | $\sigma(z)(1 - \sigma(z))$ | Smooth, bounded $(0, 1)$.                    |
| **Tanh** | $\tanh(z)$ | $1 - \tanh^2(z)$ | Smooth, bounded $(-1, 1)$.                   |
| **ReLU** | $\max(0, z)$ | $1 \text{ if } z > 0, \text{ else } 0$ | Efficient. Unbounded output.                 |

### Forward Propagation

The network follows a standard 2-layer architecture:

1.  **Hidden Layer:**

$$
Z^{[1]} = X \cdot W^{[1]} + b^{[1]}
$$

$$
A^{[1]} = \sigma(Z^{[1]})
$$

2.  **Output Layer:**

$$
Z^{[2]} = A^{[1]} \cdot W^{[2]} + b^{[2]}
$$

$$
\hat{Y} = Z^{[2]}
$$

### Backpropagation

To train the network, we minimize the **Mean Squared Error** loss function $J$. We compute the gradient of $J$ with respect to the weights using the Chain Rule.

**Step A: Error at Output**<br>
First, we calculate the derivative of the loss with respect to the output layer's input. Since the output activation is linear for regression:

$$
\delta^{[2]} = \frac{\partial J}{\partial Z^{[2]}} = (\hat{Y} - Y)
$$

**Step B: Propagating to Hidden Layer**<br>
We propagate the error backwards to the hidden layer. This requires the **Hadamard product** ($\odot$), which is element-wise multiplication, to apply the derivative of the activation function:

$$
\delta^{[1]} = (\delta^{[2]} \cdot W^{[2]T}) \odot \sigma'(Z^{[1]})
$$

**Step C: Gradients for Updates**<br>
Finally, we calculate the gradients for the weights and biases:

$$
\frac{\partial J}{\partial W^{[2]}} = A^{[1]T} \cdot \delta^{[2]}
$$

$$
\frac{\partial J}{\partial W^{[1]}} = X^T \cdot \delta^{[1]}
$$

---

## Optimization Methods

This repository contains three separate Jupyter Notebooks, each implementing the approximation using a different optimization strategy. Below is a detailed breakdown of each method used in this project.

### Batch Gradient Descent

#### Theoretical Approach
Batch Gradient Descent (BGD) distinguishes itself by computing the gradient using the **entire training dataset** $\mathcal{D}$ for every single iteration of the optimization process. This contrasts with stochastic methods that approximate the gradient using subsets of data. Given a dataset of size $N$, the algorithm calculates the exact gradient of the global cost function, ensuring the update vector points in the direction of the steepest descent for the total error surface.

#### Mathematical Formulation

##### 1. Global Cost Function
The cost function $J(\theta)$ is defined as the average loss over the total number of samples $N$ in the dataset:

$$
J(\theta) = \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}(\hat{y}^{(i)}, y^{(i)}) = \frac{1}{2N} \sum_{i=1}^{N} (\hat{y}^{(i)} - y^{(i)})^2
$$

##### 2. The Batch Gradient
The gradient used for the update step is the average of the partial derivatives calculated for every sample in the dataset. The summation runs from $1$ to $N$:

$$
\nabla_\theta J(\theta) = \frac{1}{N} \sum_{i=1}^{N} \nabla_\theta \mathcal{L}(\hat{y}^{(i)}, y^{(i)})
$$

### Stochastic Gradient Descent

#### Theoretical Approach
Stochastic Gradient Descent (SGD) is an optimization approach where the gradient is approximated using a **single training example** $(x^{(i)}, y^{(i)})$ chosen at random for each iteration. Unlike Batch Gradient Descent, which calculates the exact gradient by averaging over the entire dataset, SGD uses the gradient of the loss from one sample as an unbiased estimator of the true gradient. This introduces noise into the optimization path, causing the loss to fluctuate, but allows for significantly more frequent parameter updates.

#### Mathematical Formulation

##### 1. Instantaneous Cost Function
Instead of minimizing the average loss over the entire dataset, SGD considers the loss function for a specific, randomly selected sample $i$ at each step:

$$
J(\theta; x^{(i)}, y^{(i)}) = \mathcal{L}(\hat{y}^{(i)}, y^{(i)}) = \frac{1}{2} (\hat{y}^{(i)} - y^{(i)})^2
$$

##### 2. The Stochastic Gradient
The gradient is computed with respect to the parameters $\theta$ using only the $i$-th sample. Consequently, the parameters are updated immediately after processing this single sample, rather than waiting for the entire dataset to be evaluated. This removes the summation over $N$ found in batch methods:

$$
\nabla_\theta J(\theta) \approx \nabla_\theta \mathcal{L}(\hat{y}^{(i)}, y^{(i)})
$$

### Mini-Batch Gradient Descent

#### Theoretical Approach
Mini-Batch Gradient Descent (MBGD) acts as a compromise between Batch Gradient Descent and Stochastic Gradient Descent. Instead of using the full dataset or a single sample, it partitions the training data into **mini-batches of size** $B$. The model computes the gradient and updates its parameters after processing each batch. This approach leverages the computational efficiency of matrix operations (like BGD) while providing the frequent updates and convergence speed of stochastic methods.

#### Mathematical Formulation

##### 1. Mini-Batch Cost Function
For a specific mini-batch $\mathcal{B}$ containing $B$ samples, the cost function is the average loss over that specific subset:

$$
J_{\mathcal{B}}(\theta) = \frac{1}{B} \sum_{k=1}^{B} \mathcal{L}(\hat{y}^{(k)}, y^{(k)}) = \frac{1}{2B} \sum_{k=1}^{B} (\hat{y}^{(k)} - y^{(k)})^2
$$

##### 2. The Mini-Batch Gradient
The gradient is estimated by averaging the partial derivatives over the current mini-batch samples. This provides an approximation of the true gradient that is more stable than SGD but less computationally expensive than BGD:

$$
\nabla_\theta J(\theta) \approx \frac{1}{B} \sum_{k=1}^{B} \nabla_\theta \mathcal{L}(\hat{y}^{(k)}, y^{(k)})
$$

---

## Getting Started

### Prerequisites
* Python 3.8+
* Jupyter Notebook

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/mattia-3rne/universal-function-approximation-via-neural-network.git
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Launch Jupyter Notebook**:
    ```bash
    jupyter notebook
    ```

---

## Project Structure

* `SGD.ipynb`: Stochastic Gradient Descent.
* `BGD.ipynb`: Batch Gradient Descent.
* `MBGD.ipynb`: Mini-Batch Gradient Descent.
* `requirements.txt`: Python package dependencies.
* `README.md`: Project documentation.
