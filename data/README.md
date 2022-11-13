#### clone the ISMIR tutorial git repo (to somewhere OUTSIDE the cs230_project folder)
git clone git@github.com:source-separation/tutorial.git

#### create a condo environment with python 3.9 (conda will default to downloading python 3.11, which DOESN’T work with the ISMIR repo)
conda create --name ismir-tutorial-environment python=3.9

cd into the ismir tutorial folder

#### activate the ismir-tutorial-environment
conda activate ismir-tutorial-environment

#### update the conda environment with the environment.yml
conda env update --file environment.yml --prune

#### now you should be able to run get_musdb18_data.py!
