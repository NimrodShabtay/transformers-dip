# DIP-transformers

## Preparations
* Create `exps` directory at project's root
* install requirements using `pip install -r requirments.txt`

## Code walkthrough
* The main entrypoint is `main_transformer.py` <br>
Chaning `EXP` variable to run the desired architecture - the options are `transformer` or `org` <br>
All the visualiztions (training curves, intermediate results and the best result during the training will be under `exps/YYYY_MM_DD_HH_mm/`

* Our architecture is in `new_nets.py`
<br>Where all transformer related blocks are in `vit_model.py`

## Run the code
To run the code: <br>
`python main_transformer.py`
