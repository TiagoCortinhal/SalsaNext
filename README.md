# SalsaNext


First create the anaconda env with:
```conda env create -f salsanext.yml``` then activate the environment with ```conda activate salsanext```.

To train/eval you can use the following scripts:


 * [Training script](train.sh) (you might need to chmod +x the file)
   * We have the following options:
     * ```-d [String]``` : Path to the dataset
     * ```-a [String]```: Path to the Architecture configuration file 
     * ```-m [String]```: Which model to use (rangenet,salsanet,salsanext)
     * ```-l [String]```: Path to the main log folder
     * ```-n [String]```: additional name for the experiment
     * ```-c [String]```: GPUs to use
   * For example if you have the dataset at ``/dataset`` the architecture config file in ``/salsanext.yml``
   and you want to save your logs to ```/logs``` to train "salsanext" with 2 GPUs with id 3 and 4:
     * ```./train.sh -d /dataset -a /salsanext.yml -m salsanext -l /logs -c 3,4```
<br>
<br>

 * [Eval script](eval.sh) (you might need to chmod +x the file)
   * We have the following options:
     * ```-d [String]```: Path to the dataset
     * ```-p [String]```: Path to save label predictions
     * ``-m [String]``: Path to the location of saved model
     * ``-s [String]``: Eval on Validation or Train (standard eval on both separately)
     * ```-n [String]```: Which model to use (rangenet,salsanet,salsanext)
   * If you want to infer&evaluate a model that you saved to ````/salsanext/logs/[the desired run]```` and you
   want to infer$eval only the validation and save the label prediction to ```/pred```:
     * ```./eval.sh -d /dataset -p /pred -m /salsanext/logs/[the desired run] -s validation -n salsanext```
     
     
     
 The model is defined [here](train/tasks/semantic/modules/segmentator.py), the training logic is
 [here](train/tasks/semantic/modules/trainer.py).
 
 