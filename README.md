# Zircon Project

An extension of [MoCapAct](https://microsoft.github.io/MoCapAct/) and [dm_control](https://www.deepmind.com/publications/dm-control-software-and-tasks-for-continuous-control) for obstacle avoidance tasks.

## Setup

Run the following script to clone the repository, update the submodules, and install the dependencies in editable mode.

```bash
#!/bin/bash

destinationFolder="MajorProjectMujoco";
envName="ZirconProject"

# Clone the repository if not already present
if [[ -d $destinationFolder ]];
then
    echo "Repository already present! Skipping clone.";
else
    git clone git@github.com:Team-Zircon/ZirconProject.git $destinationFolder;
fi;

# Update the submodules
cd $destinationFolder;
git pull;
git submodule init;
git submodule update;

# Create conda environment if not already present
eval "$(conda shell.bash hook)"
conda env list | grep $envName
if [[ $? != 0 ]]; # if $envName not found in conda environment list
then
    conda create -n $envName python=3.8 -y;
    conda activate $envName;
    pythonPath=$(which python)
    if [[ $pythonPath != *"$envName"* ]];
    then
        echo "Conda is not activated";
        exit 1;
    fi;
else
    echo "Environment already created! Using existing environment."
fi;

# Install dependencies in editable mode
cd MoCapAct_source;
pip install -e .;
cd ../dm_control_source;
pip install -e .;
cd ..;
```
