# Freegap

The evaluation code for paper [PVLDB'20] Free Gap Information from the Differentially Private Sparse Vector and Noisy Max Mechanisms. 

## Datasets

Run `bash scripts/download_datasets.sh` in root directory and it will download the required datasets in `./datasets` directory.

## Running the Experiments

Install dependencies via `pip install -e .` and then issue `python -m freegap -h` to see the following help message:

```
usage: __main__.py [-h] [--datasets DATASETS] [--output OUTPUT] [--clear] [--counting] algorithm

positional arguments:
  algorithm            The algorithm to evaluate, options are `All, AdaptiveSparseVector, AdaptiveEstimates,
                       GapSparseVector, GapTopK`.

optional arguments:
  -h, --help           show this help message and exit
  --datasets DATASETS  The datasets folder
  --output OUTPUT      The output folder
  --clear              Clear the output folder
  --counting           Set the counting queries
```

Run `python -m freegap <algorithm>` to run evaluation conducted in our paper for `<algorithm>`, the results will be 
generated in `output/<algorithm>` folder with name `<dataset>-<metric>-<epsilon_value>.pdf`.

For more details on the code, checkout the comments and function doc-strings.
 
:warning: : We will cache the results in `output/<algorithm>/<algorithm>-<dataset>.json`, if this file is detected, no 
actual experiments will be conducted, instead, this cached results will be used directly to generate plots. Remember to
delete or move the `output` folder or run with the `--clear` flag to run fresh experiments.

:warning: : we use `numba` to JIT the functions for best performance, therefore the first run of the program might be 
slow due to numba's compilations.
