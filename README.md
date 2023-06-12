![](https://github.com/sun-umn/Deeplifting/blob/main/deeplifting.png)

# Deeplifting 🧠
This repository delves into a research field we refer to as "deeplifting". Deeplifting merges the domains of deep learning and optimization, proposing a novel approach to solve optimization problems. By using neural networks to reparameterize input spaces, we can significantly improve solutions, making them both more potent and robust.

## Setup ⚙️
Note 📝: Repostory was configured using Python 🐍 version 3.9.11

IPOPT ("Interior Point OPTimizer, pronounced I-P-Opt") is a software package for large-scale nonlinear optimization.

Before you proceed with the installation of dependencies from `requirements.txt`, you need to set up IPOPT on your system.

Here's a step-by-step guide on how to set it up on Mac and Linux.

### Mac:

1. First, install Homebrew if you haven't installed it yet. Open Terminal and paste the following command:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

2. Install IPOPT via Homebrew:

```bash
brew install ipopt
```

### Linux:

1. First, update your package list:

```bash
sudo apt-get update
```

2. Install the necessary tools and libraries:

```bash
sudo apt-get install gcc g++ gfortran git patch wget pkg-config liblapack-dev libmetis-dev libmumps-seq-dev libblas-dev coinor-libipopt-dev
```

For both Mac and Linux, once you have IPOPT installed, you can proceed with the installation of the Python dependencies from your `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Note: These instructions assume that you have Python and pip already installed on your system. If that's not the case, you'll need to install those first. Also, if you're using a Python virtual environment, ensure that you've activated the environment before running these commands.

Always refer to the official IPOPT documentation or source for the most up-to-date and detailed setup instructions.

### Pre-Commit Hooks 🤖

To conclude, we've integrated pre-commit hooks to maintain a consistent standard and style across all contributed code.

Here's how you can install and set up the pre-commit hooks:

1. First, you need to install `pre-commit`. You can do this with pip:

```bash
pip install pre-commit
```

2. Then, navigate to your project directory, where the `.pre-commit-config.yaml` file is located:

```bash
cd your_project_directory
```

3. In the project directory, run the following command to install the pre-commit hooks:

```bash
pre-commit install
```

With these steps completed, the pre-commit hooks are now set up. They will automatically run checks on your code each time you try to commit changes in this repository. If the checks pass, the commit will be allowed; if they fail, the commit will be blocked, and you'll need to fix the issues before trying again.


