# Module1 t-sne from scratch IvanGu
This repo implements a classic non-linear dimensionality reduction algorithmed called t-SNE: t-distributed Stochastic Neighbor Embedding.

For a more detailed description of t-SNE, please refer to https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding

# Getting started

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