# udacity-unityMLagents-continuousControl
udacity continuous control programming assignment learning to control the movement of a double jointed arm to track spheres in unity MLagents

# Project Details 
In this project, we trained a DRL agent with the Unity MLAgents environment to teach a robot how to control the movement of a double jointed arm in order to track the movement of spheres in three dimensional space.  It does this by affecting four torque values that control the joints. 

Each action is a vector of these four floating point torque values, corresponding to the torque to be applied to the two joints. Every entry in the action vector should be a number between -1.0 and 1.0. The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm.

A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.  The environment is considered ‘solved’ when the average score is above 30.0 for the trailing 100 consecutive episodes.

The task is episodic. There were two possible projects possible, whereby a single agent is trained by itself, with the objective to achieve the minimum 30.0 score. Alternatively, a set of 20 agents could be trained at the same time, by paralleling the matrix computation required for calculations at every step of training. In that example, solving the environment requires achieving an average of 30.0 across all agents. 

# Getting Started
The project is comprised of three files:
1. **continuous_control.ipynb**: imports packages, connects to Unity environment, and then runs the learning function that will interact with that environment, leveraging various neural networks defined in model.py and the Agent class defined in ddpg_agent.py
2. **model.py**: contains the pytorch neural networks (two actor networks, and two critic networks), that actively learn and store the learnings from the DDPG training process  
3. **ddpg_agent.py**: contains the code to initialize the (four) networks, a replay buffer class, an OUNoise class, and wrappers for selecting actions, submitting them to the environment (through 'step') and then learn from experience stored in the replay buffer

Logically, there are three steps required in order to execute the project: Download the files referred to above, install the necessary Unity MLAgents environment, and execute the code in the environment of your choice, such as a Jupyter Notebook environment (which is recommended for simplicity).  Here are the more detailed instructions for doing so:
1. Download the three files above (from the public github repo)   
Execute this command (at a shell prompt, in the appropriate source repo location where you wish to copy the files): \
$ *git clone https://github.com/kamccall/udacity-unityMLagents-continuousControl* \
This will clone the entire repository - including the three source files as well as associated README and REPORT files - into that source directory. 
1. Install the Unity MLAgents environment (within which the agent will train and subsequently execute)   
Follow the directions below in order to install the needed 'Reacher' environment for Unity MLAgents: \
https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control#getting-started
1. Execute the code (either within a Jupyter Notebook environment, or within an alternative IDE) \
There are dependencies upon various Python packages in order to successfully execute this project, such as numpy, pyplot, torch, and others. Depending upon your Python environment, it is likely that those packages are already installed, in which case the 'import' commands in the code will successfully execute, thereby allowing the use of the code in those packages.\
If your environment does not already have all of these packages installed, it is possible that you will need to install one or more packages before you can successfully execute the 'import' command to utilize them.  If that is the case:
    1. ADD a new cell at the top of your Jupyter notebook
    1. Follow the appropriate directions in this link to install the required packages (which will be different depending upon whether you are using 'pip' or 'conda' as your package manager): https://jakevdp.github.io/blog/2017/12/05/installing-python-packages-from-jupyter/

# Build and Test Instructions
There are three ways to run the code:
1. Install all three files in the same directory, and execute the navigation .ipynb notebook file in a Jupyter notebook server , or
2. Insert the code from 'model.py' and 'dqn_agent.py' into new (inserted) cells in the .ipynb notebook file, and execute everything from the single notebook file within a Jupyter notebook environment, or
3. (More difficult): Copy and paste the Python code from within the .ipynb notebook file into a Python source code file, and then execute the program from the command line (with all appropriate dependencies based on your IDE and OS)

