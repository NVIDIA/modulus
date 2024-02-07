
## Setup
download torch_scatter from https://data.pyg.org/whl/. 
In a conda env with pytorch / cuda available, run:
```
pip install -r requirements.txt
pip install ./torch_scatter-2.0.6-cp38-cp38-linux_x86_64.whl
```
note: pyvista is required only if need to run data proprocessing with the raw simulation data files 

## Train: 

> python train.py --data_path={data path for the pre-processed data in tfrecord format, i.e. data/large-time-step/step_100} --model_path=models/{path to store training model} --loss={i.e.me loss} --noise_std={i.e.1e-9}

Currently default params: 

- INPUT_SEQUENCE_LENGTH = 5
- PREDICT_LENGTH = 1
- NUM_PARTICLE_TYPES = 3

## Test: 

> python train.py --mode=eval_rollout --batch_size=1 --model_path={path to model trained ckpt} --noise_std=0 --output_path={path to store outputs} --data_path={preprocessed test data tfrecord}
 
i.e.

> python train.py --mode=eval_rollout --batch_size=1 --model_path=models/m2_lr6/model_loss-4.17E-06_step-1113000.pt --noise_std=0 --output_path=rollouts/ --data_path=data/

To generate visualization of the test result: 

> python  render_rollout.py --rollout_path={selected predicted .pkl} --metadata_path={metadata path} --test_build={test file name}

i.e.
> python render_rollout.py --rollout_path=rollouts/rollout_test_2.pkl --metadata_path=data --test_build=2


(100 step model)/home/chenle/data/large-time-step

## Data: 

- Test data 

    - Same voxel resolution as train

- To generate your own tfrecord from Physical simulation output: 
  
    - Run: 
  > python data_process/rawdata2tfrecord.py raw_data_dir, metadata_json_path, mode
    
    i.e.
- > python data_process/rawdata2tfrecord.py /home/VF-simulation-data data test

Defition of step_context & methods tried: 

- appending only the previous step global context / ( sinter temperature)
    > tensor_dict['step_context'] = tensor_dict['step_context'][-predict_length - 1][tf.newaxis]

- appending previous sequence of global context / (sequence of sinter temperature)
    >  tensor_dict['step_context'] = tf.reshape(tensor_dict['step_context'][:-1], [1, -1])

-  appending the entire sequence of sintering profile 
    > tensor_dict['step_context'] = tf.reshape(tensor_dict['step_context'],[1, -1])

## Reference

Learning to Simulate Complex Physics with Graph Networks -- https://arxiv.org/abs/2002.09405

```
@inproceedings{sanchezgonzalez2020learning,
  title={Learning to Simulate Complex Physics with Graph Networks},
  author={Alvaro Sanchez-Gonzalez and
          Jonathan Godwin and
          Tobias Pfaff and
          Rex Ying and
          Jure Leskovec and
          Peter W. Battaglia},
  booktitle={International Conference on Machine Learning},
  year={2020}
}
```