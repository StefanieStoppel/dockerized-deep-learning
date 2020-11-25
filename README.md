# dockerized-deep-learning 
## Debugging Deep Learning Docker Containers
This repo goes hand in hand with [this Medium post I wrote]().

### Getting started
Clone this project: `git clone git@github.com:StefanieStoppel/dockerized-deep-learning.git`.

#### Prerequisites
You need to have the following things installed on your system:

- [Python](https://www.python.org/downloads/) (≥3.6, I used version 3.8.6)
- [Conda or miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html)
- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [VSCode](https://code.visualstudio.com/Download)
- (optional) If your machine comes with a GPU and you want to use it for training the network, 
you need to install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) as well.

#### Create and activate the conda environment
Open a terminal and cd into the directory you cloned the project into.
Execute the following:
```bash
conda env create --file=environment.yml
conda activate docker-deep-learning
```

#### Build the Docker image
In the root directory execute the following:
```bash
docker-compose build ddl
```

#### Debug the application
- Open the project in VSCode.
- Set a breakpoint in a Python file.
- Open the terminal in the project root and run `docker-compose up ddl` to run the Docker container.
- Click on the ▶️play icon in the left hand sidebar of VSCode to launch the debugger.
- Wait until it stops at you breakpoint.

#### 🐛️ Happy debugging! 🐛️
