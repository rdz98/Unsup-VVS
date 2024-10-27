We sincerely thank the anonymous reviewers for taking the time to review our paper. 
This is an official implementation of our paper titled "Improving Unsupervised Task-driven Models of Ventral Visual Stream via Relative Position Predictivity".
The part of contrastive learning of this repo is based on [ae-foster/pytorch-simclr](https://github.com/ae-foster/pytorch-simclr).

## Install requirements
To satisfy the environmental requirements, you can run:
```
$ conda env create -f environment.yml
$ pip install -r requirements.pip
```

## Running the code
To train the base model by our proposed method with the balancing weight alpha = 0.01, you can run:
```
$ python train_base_model.py --cosine-anneal --alpha=0.01 --filename=alpha_0.01
```
Due to the space limitation, we did not upload the checkpoint files of the trained base models, but you can easily obtain them by running the above command.

After the training:
+ To evaluate the task accuracy of the trained base model, including image classification (IC) accuracy and relative position prediction (RPP) accuracy, you can run:
  ```
  $ python evaluate_task_accuracy.py --load-from=alpha_0.01
  ```
+ To evaluate the brain similarity of the trained base model to four cortical regions (V1, V2, V4, IT) of ventral visual stream, you can run:
  ```
  $ python evaluate_brain_similarity.py --load-from=alpha_0.01
  ```
+ To show the results, you can run:
  ```
  $ python view_barin_similarity.py --load-from=alpha_0.01
  ```
  The complete results are recorded in `brain_similarity.log`.


## Statements
The codes are for learning and research purposes only.