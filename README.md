
# psog_nn

This repository contains code for ongoing project "Power-efficient and shift-robust PSOG".

Related publications (\<a branch name for the corresponding version of the codebase\>):
* ETRA 2019: "Power-efficient and shift-robust eye-tracking sensor for portable VR headsets" (etra2019) 
* ICCV 2019: "Assessment of Shift-Invariant CNN Gaze Mappings for PS-OG Eye Movement Sensors" (iccv2019)

## Dataset
The full dataset (9.15 Gb) can be downloaded from [here](https://digital.library.txstate.edu/bitstream/handle/10877/7955/UnprocessedDataset.zip?sequence=2&isAllowed=y). 

The 'slim' version of the dataset (59 Mb) doesn't provide the set of images for each subject but includes the result of the whole  preprocessing step instead. If you are interested only in machine-learning or plotting modules, you can download it [here](https://digital.library.txstate.edu/bitstream/handle/10877/7955/SlimDataset.zip?sequence=1&isAllowed=y). 

It's recommended to extract the dataset into '.\dataset' directory.

## Install
It's highly recommended to install [conda](https://conda.io/en/latest/miniconda.html) to run this project.
Note that you should have Nvidia GPU to run it efficiently. Assuming that, the next command will create 'psog_nn_etra2019' conda environment with all sufficient packages needed:
```
$ conda env create -f env.yml
```
In case you don't have Nvidia GPU and/or you're not interested in machine learning part, the next command will create 'fallback' conda environment with the same name:
```
$ conda env create -f env_nogpu.yml
```
All following commands should be run inside newly created environment. To activate it, use:
```
$ conda activate psog_nn_etra2019
```

## Project structure
The project consists of three modules:
- preproc: the whole preprocessing pipeline
- ml: machine learning part of the project
- plots: scripts to create plots used in the paper

For more specific details, consult the corresponding section of the paper itself.

## Usage:
```
python main.py [-h] [--root ROOT] {preproc,ml,plot}
```
modes:
&nbsp;&nbsp;&nbsp;&nbsp;{preproc, ml, plot} (can be run in only one mode at once!)

For more details, consult a help page of the corresponding mode:
```
python main.py {preproc,ml,plot} [-h]
```


## Examples
(assumes that dataset is download and unpacked to default '.\dataset' directory, unless specified):

1. To run the whole preprocessing, use:
```$ python main.py preproc --missed --head_mov -- shift_crop --psog```
	1. To run the PSOG simulation only for subject 12, with dataset in the custom directory, use:
```$ python main.py --root ~/dev/research/psog_nn/dataset preproc --psog 12```
2. To run the whole machine learning part, use: 
```$ python main.py ml --grid_search --evaluate```
	1. To run the grid-search only for 'MLP' architecture, low-power setup, use:
```$ python main.py ml --grid_search --arch mlp --setup lp```
3. To create all plots used in the paper, use:
```$ python main.py plot --error_bars --samples_distrib```
	1. To create additional boxplots of spatial accuracy for per subject evaluation, use:
```$ python main.py plot --boxplots```

	P.S. plot mode can be restricted to specific architecture and/or setup as shown in 2.i

## Notes:

1. Grid-search and evaluation are very time-consuming tasks. It may take up to one week to reproduce them all. 

2. Backup of all results obtained for ETRA 2019 paper is provided in 'etra2019_full_results.zip' archive. To use them (for example, to plot or analyse the results), just unpack the archive to the project directory. 

	Note that files included were obtained using older version of the software (with the bug of incorrect time saved in the last part of a grid-search log) and their names are changed to adhere to new naming conventions.
3. Right now if '.\ml\log' and '.\ml\results' contains files with proper naming, the grid-search and evaluation steps respectively will be skipped . This behavior can be altered in the main.py, with 'redo' arguments changed to 'True'.
	
	Note that it is the case when the archive with ETRA results is unpacked.
