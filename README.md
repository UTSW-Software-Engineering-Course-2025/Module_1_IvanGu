[![Docs Butler](https://github.com/UTSW-Software-Engineering-Course-2025/Module_1_IvanGu/actions/workflows/static.yml/badge.svg?branch=docs)](https://github.com/UTSW-Software-Engineering-Course-2025/Module_1_IvanGu/actions/workflows/static.yml)

# Module1:t-sne and GraphDR (Ivan's Version)
This repo implements two dimensionality reduction algorithms: t-sne and GraphDR. t-sne (t-distributed Stochastic Neighbor Embedding) is a classic non-linear dimensionality reduction algorithme; whereas GraphDR is a quasilinear visualization and general representation tool. 

For a more detailed description of t-SNE, please refer to https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding

# Getting started (t-sne)

- `conda create -n utsw_swe_25 python=3.9 matplotlib` this creates a conda env called `utsw_swe_25`, with required packages `matplotlib` (which includes `numpy`)
- `conda activate utsw_swe_25` this activates the environment
- navigate to `src`. this contains the source code
- `python tsne.py path_to_your_input_X`. simplest usage of the implementation
- `python tsne.py path_to_your_input_X --perplexity=35 --min_gain=0.05 --print_all`. This runs the t-sne algorithm with customized CLI arguments. In this case, perplexity is set to 35, minimum gain is set to 0.05, and helper information will be printed during code execution.

A full list of parameter options:

   Parameter | Description | Default (if any)
   ---------------------------------------- | ------------- | -------------
   (required)
   --path_to_X  |   file path of the input file
   (optional)
   --path_to_labels | path to the labels file | `../data/mnist2500_labels.txt`
   --no_dims | number of dimensions | 2
   --perplexity | perplexity of tsne, used for precision adjustment | 30
   --T | number of time steps | 1000
   --initial_momentum | initial momentum during early stage | 0.5
   --final_momentum | momentum after early stage | 0.8
   --eta | assumed radius for cells to use in cell segmentation | 500
   --min_gain |  minimum gains used for clipping during optimization | 0.01
   --print_all |  print t-sne progress during code execution | 


# Getting started (GraphDR)
- `conda env create -f environment.yml` this creates an env called `utsw_swe_25`
- `python graphdr.py` runs GraphDR. GraphDR (Ivan's Version) do not currently have CLI capability. Navigate to the source code and search for `TODO` to change the file paths and optional arguments (e.g. `lambda_`, `no_rotations`)