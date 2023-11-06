# 701_proj

# getting set up
Thinking we could try to keep a conda environment defined/version controlled here so that we are all using the same versions of things? Not strictly necessary, happy to do whatever we want here.  

If we want to stick to a version controlled conda enviroment:

First setup:
(In the root project dir) conda env create ./env --file environment.yml

If anything changes in the environment file and you want to update your environment:
(in the root project dir) conda env update --prefix ./env --file environment.yml --prune
