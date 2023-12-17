# AI_and_Cybersecurity_Project_FaceCure
# FaceCure - Reproduced Expirements 
This repository was created to reproduce experiments from the original [FaceCure](https://github.com/ftramer/FaceCure/tree/main) repository. 
It contains the code files and the respective commands to run those files for baseline, adaptive and oblivious "defenses" against poisoning attacks on facial recognition systems. 

The dataset used for these experiments is the FaceScrub dataset which can be downloaded from [FaceScrub](https://vintage.winklerbros.net/facescrub.html). It contains 106,863 images of 530 hollywood celebrities, 265 males and 265 females. 

There are two files with the names of the actors and actresses, which contains the urls to download the images. 
The python script  download_facescrub.py is used to download the dataset. 
For actors: 
```sh
pip install requests 
pip install Pillow
python download_facescrub.py facescrub_actors.txt FaceScrub/download/
```
For actresses: 
```sh
python download_facescrub.py facescrub_actresses.txt FaceScrub/download/
```


# Attack Setup
This repository contains code to evalute three poisoning attacks against large-scale facial recognition systems. The three attacks being: 
1. Fawkes version 1.0 [Fawkes v1.0](https://github.com/Shawn-Shan/fawkes)
2. Fawkes version 0.3 [Fawkes v0.3](https://github.com/Shawn-Shan/fawkes/releases/tag/v0.3)
3. Lowkey [Lowkey](https://openreview.net/forum?id=hJmtwocEqzc)

The following defense strategies against the above attacks are evaluated:

1. Baseline: A model trainer collects perturbed pictures and trains a standard facial recognition model. 
2. Adaptive: A model trainer uses the same attack as users (as a black-box) to build a training dataset augmented with perturbed pictures, and trains a model that is robust to the attack. 
3. Oblivious: A model trainer collects perturbed pictures, waits until a new facial recongition model is released, and uses the new model to nullify the protection of previously collected pictures. 


Now, to perform the attacks, the downloaded images need to be perturbed with the three attacks: 
1. To perturb all images of one user with Fawkes v0.3: 
```sh
python fawkes3_to_perturb/fawkes/protection.py --gpu 0 -d facescrub/download/Adam_Sandler/face --batch-size 1 -m high --no-align
```
For each picture filename.png, this will create a perturbed picture filename_cloaked.png in the same directory.

Original Picture | Picture perturbed with Fawkes
-----------------|-----------------
<img src="Adam_Sandler_original.jpeg" alt="original picture" width="224"/> | <img src="Adam_Sandler_fawkes3.png" alt="perturbed picture" width="224"/> 
2. To perturb all images of one user with Fawkes v1.0: 
```sh
python fawkes1_to_perturb/fawkes/protection.py --gpu 0 -d facescrub/download/Adam_Sandler/face --batch-size 1 -m high --no-align
```
For each picture filename.png, this will create a perturbed picture filename_cloaked.png in the same directory. 

Original Picture | Picture perturbed with Fawkes
-----------------|-----------------
<img src="Adam_Sandler_original.jpeg" alt="original picture" width="224"/> | <img src="Adam_Sandler_fawkes1.png" alt="perturbed picture" width="224"/>
3. To perturb all images of one user with Lowkey: 
```sh
cd lowkey
python lowkey_attack.py  --dir ../facescrub/download/Adam_Sandler/face 
```
The models folder in the lowkey contains the models that are used in the script. 
For each picture filename.png, this will create a resized picture filename_small.png and a perturbed picture filename_attacked.png in the same directory.

Original Picture | Picture perturbed with Fawkes
-----------------|-----------------
<img src="Adam_Sandler_original.jpeg" alt="original picture" width="224"/> | <img src="Adam_Sandler_lowkey.png" alt="perturbed picture" width="224"/>

# Defense Setup
Three common facial recognition approaches are considered:

- NN: 1-Nearest Neighbor on top of a feature extractor.
- Linear: Linear fine-tuning on top of a frozen feature extractor.
- End-to-end: End-to-end fine-tuning of the feature extractor and linear classifier.

The defense setup assumes that there are four folders in the system after perturbation of the user images:

1. A directory with the original FaceScrub pictures: facescrub/download/
2. A directory with users protected by Fawkes v0.3: facescrub_fawkes3_attack/download/
3. A directory with users protected by Fawkes v1.0: facescrub_fawkes_attack/download/
4. A directory with users protected by LowKey: facescrub_lowkey_attack/download/ 

In each of the experiments below, one FaceScrub user is chosen as the attacker. All of the training images of that user are replaced by perturbed images.

A facial recognition model is then trained on the entire training set. 
The evaluation metric is the attack's protection rate (a.k.a. the trained model's test error when evaluated on unperturbed images of the attacking user). 

## Baseline evaluation with NN and linear classifiers
### NN Classifier 
Train a nearest neighbor classifier on top of the Fawkesv0.3 feature extractor, with one attacking user. 
1. To evaluate Fawkes v0.3 attack: 
```sh
python3 FaceCure/eval.py --facescrub-dir facescrub/download/ --attack-dir facescrub_fawkes3_attack/download/ --unprotected-file-match .jpeg --protected-file-match cloaked.png --classifier NN --names-list Adam_Sandler  
``` 

2. To evaluate Fawkes v1.0 attack: 
```sh
python3 FaceCure/eval.py --facescrub-dir facescrub/download/ --attack-dir facescrub_fawkes_attack/download/ --unprotected-file-match .jpeg --protected-file-match cloaked.png --classifier NN --names-list Adam_Sandler 
``` 

3. To evaluate Lowkey attack: 
```sh
python3 FaceCure/eval.py --facescrub-dir facescrub/download/ --attack-dir facescrub_lowkey_attack/download/ --unprotected-file-match small.png --protected-file-match attacked.png --classifier  NN --names-list Adam_Sandler 
``` 


Results:

Fawkes v0.3 (baseline NN) |  Fawkes v1.0 (baseline NN) | Lowkey (baseline NN)
---------------------|---------------------|--------------------- 
```Protection rate: 0.89```|```Protection rate: 0.72``` |```Protection rate: 0.44``` 

The `--classifier` argument is set to linear to train a linear classifier instead of a nearest neighbor one. 



    
## Adaptive evaluation with NN and linear classifiers
### NN Classifier  
Same as for the baseline classifier above, but the option --robust-weights cp-robust-10.ckpt is added to use a robustified feature extractor. This feature extractor was trained using the train_robust_features.py script, which finetunes a feature extractor on known attack pictures.


1. To evaluate Fawkes v0.3 attack: 
```sh
python3 FaceCure/eval.py --facescrub-dir facescrub/download/ --attack-dir facescrub_fawkes3_attack/download/ --unprotected-file-match .jpeg --protected-file-match cloaked.png --classifier NN --names-list Adam_Sandler  --robust-weights FaceCure/cp-robust-10.ckpt 
``` 

2. To evaluate Fawkes v1.0 attack: 
```sh
python3 FaceCure/eval.py --facescrub-dir facescrub/download/ --attack-dir facescrub_fawkes_attack/download/ --unprotected-file-match .jpeg --protected-file-match cloaked.png --classifier NN --names-list Adam_Sandler --robust-weights FaceCure/cp-robust-10.ckpt 
``` 

3. To evaluate Lowkey attack: 
```sh
python3 FaceCure/eval.py --facescrub-dir facescrub/download/ --attack-dir facescrub_lowkey_attack/download/ --unprotected-file-match small.png --protected-file-match attacked.png --classifier  NN --names-list Adam_Sandler --robust-weights FaceCure/cp-robust-10.ckpt 
``` 



Results:

Fawkes v0.3 (adaptive NN) |  Fawkes v1.0 (adaptive NN) | Lowkey (adaptive NN)
---------------------|---------------------|--------------------- 
```Protection rate: 0.06```|```Protection rate: 0.22``` |```Protection rate: 0.22``` 

The `--classifier` argument is set to linear to train a linear classifier instead of a nearest neighbor one. 


 
# Oblivious Defense evaluation with NN and linear classifiers
## Oblivious NN classifier
The experiments are repeated using different feature extractors. The following feature extractors were trained and evaluated. 
1. Fawkes v1.0 Extractor: This is the WebFace feature extractor which is used as a surrograte model in the Fawkes attack so it is used from there.  
2. MagFace: Downloded from [MagFace](https://github.com/IrvingMeng/MagFace/). 
3. CLIP: Downloded from [CLIP](https://github.com/openai/CLIP). 
Or can be installed from the following commands: 
```sh
pip install --yes -c pytorch torchvision 
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
``` 
4. VGG-Face: Can be downloaded from [DeepFace](https://github.com/serengil/deepface)  
5. FaceNet: Can be downloaded from [DeepFace](https://github.com/serengil/deepface)
6. ArcFace: Can be downloaded from [DeepFace](https://github.com/serengil/deepface)
7. SFace: Can be downloaded from [DeepFace](https://github.com/serengil/deepface)

Or can be installed by: 
```sh
pip install deepface 
```


And then all three attacks were tested on all of the above feature extractors. 
Additional code was added in eval_oblivious.py file to run VGG-Face, FaceNet, ArcFace and SFace feature extractors. 
But also an additional attack was tested where 50% of the user's images were from Fawkes v1.0 and the other 50% from Fawkes v0.3. For this a new folder was created facescrub_fawkes_50_50_attack which contains these pictures. 

### 1. Fawkes v1.0 Extractor 
1. To evaluate Fawkes v0.3 attack: 
```sh
python3 FaceCure/eval_oblivious.py --facescrub-dir facescrub/download/ --attack-dir facescrub_fawkes3_attack/download/ --unprotected-file-match .jpeg --protected-file-match cloaked.png --classifier  NN --names-list Adam_Sandler --model fawkesv10
``` 

2. To evaluate Fawkes v1.0 attack: 
```sh
python3 FaceCure/eval_oblivious.py --facescrub-dir facescrub/download/ --attack-dir facescrub_fawkes_attack/download/ --unprotected-file-match .jpeg --protected-file-match cloaked.png --classifier  NN --names-list Adam_Sandler --model fawkesv10
``` 

3. To evaluate Lowkey attack: 
```sh
python3 FaceCure/eval_oblivious.py --facescrub-dir facescrub/download/ --attack-dir facescrub_lowkey_attack/download/ --unprotected-file-match small.png --protected-file-match attacked.png --classifier  NN --names-list Adam_Sandler --model fawkesv10
```

4. To evaluate 50% Fawkes v1.0 and 50% Fawkes v0.3 attack: 
```sh
python3 FaceCure/eval_oblivious.py --facescrub-dir facescrub/download/ --attack-dir facescrub_fawkes_50_50_attack/download/ --unprotected-file-match .jpeg --protected-file-match cloaked.png --classifier  NN --names-list Adam_Sandler --model fawkesv10
```

Results: 
Fawkes v0.3 (oblivious NN) |  Fawkes v1.0 (oblivious NN) | Lowkey (oblivious NN)| 50% Fawkes v1.0 and 50% Fawkes v0.3 (oblivious NN)
---------------------|---------------------|--------------------- |--------------------- 
```Protection rate: 0.89```|```Protection rate: 1.00``` |```Protection rate: 0.44```  |```Protection rate: 1.00```  

### 2. MagFace
1. To evaluate Fawkes v0.3 attack: 
```sh
python3 FaceCure/eval_oblivious.py --facescrub-dir facescrub/download/ --attack-dir facescrub_fawkes3_attack/download/ --unprotected-file-match .jpeg --protected-file-match cloaked.png --classifier  NN --names-list Adam_Sandler --model magface --resume MagFace/models_pth/magface_epoch_00025.pth 
``` 

2. To evaluate Fawkes v1.0 attack: 
```sh
python3 FaceCure/eval_oblivious.py --facescrub-dir facescrub/download/ --attack-dir facescrub_fawkes_attack/download/ --unprotected-file-match .jpeg --protected-file-match cloaked.png --classifier  NN --names-list Adam_Sandler --model magface --resume MagFace/models_pth/magface_epoch_00025.pth 
``` 

3. To evaluate Lowkey attack: 
```sh
python3 FaceCure/eval_oblivious.py --facescrub-dir facescrub/download/ --attack-dir facescrub_lowkey_attack/download/ --unprotected-file-match small.png --protected-file-match attacked.png --classifier  NN --names-list Adam_Sandler --model magface --resume MagFace/models_pth/magface_epoch_00025.pth 
```

4. To evaluate 50% Fawkes v1.0 and 50% Fawkes v0.3 attack: 
```sh
python3 FaceCure/eval_oblivious.py --facescrub-dir facescrub/download/ --attack-dir facescrub_fawkes_50_50_attack/download/ --unprotected-file-match .jpeg --protected-file-match cloaked.png --classifier  NN --names-list Adam_Sandler --model magface --resume MagFace/models_pth/magface_epoch_00025.pth 
```

Results:

Fawkes v0.3 (oblivious NN) |  Fawkes v1.0 (oblivious NN) | Lowkey (oblivious NN)| 50% Fawkes v1.0 and 50% Fawkes v0.3 (oblivious NN)
---------------------|---------------------|--------------------- |--------------------- 
```Protection rate: 0.07```|```Protection rate: 0.06``` |```Protection rate: 1.00```  |```Protection rate: 0.06```   

### 3. CLIP 
1. To evaluate Fawkes v0.3 attack: 
```sh
python3 FaceCure/eval_oblivious.py --facescrub-dir facescrub/download/ --attack-dir facescrub_fawkes3_attack/download/ --unprotected-file-match .jpeg --protected-file-match cloaked.png --classifier  NN --names-list Adam_Sandler --model clip 
``` 

2. To evaluate Fawkes v1.0 attack: 
```sh
python3 FaceCure/eval_oblivious.py --facescrub-dir facescrub/download/ --attack-dir facescrub_fawkes_attack/download/ --unprotected-file-match .jpeg --protected-file-match cloaked.png --classifier  NN --names-list Adam_Sandler --model clip 
``` 

3. To evaluate Lowkey attack: 
```sh
python3 FaceCure/eval_oblivious.py --facescrub-dir facescrub/download/ --attack-dir facescrub_lowkey_attack/download/ --unprotected-file-match small.png --protected-file-match attacked.png --classifier  NN --names-list Adam_Sandler --model clip 
```

4. To evaluate 50% Fawkes v1.0 and 50% Fawkes v0.3 attack: 
```sh
python3 FaceCure/eval_oblivious.py --facescrub-dir facescrub/download/ --attack-dir facescrub_fawkes_50_50_attack/download/ --unprotected-file-match .jpeg --protected-file-match cloaked.png --classifier  NN --names-list Adam_Sandler --model clip 
```

Results:

Fawkes v0.3 (oblivious NN) |  Fawkes v1.0 (oblivious NN) | Lowkey (oblivious NN)| 50% Fawkes v1.0 and 50% Fawkes v0.3 (oblivious NN)
---------------------|---------------------|--------------------- |--------------------- 
```Protection rate: 0.14```|```Protection rate: 0.23``` |```Protection rate: 0.20```  |```Protection rate: 0.16```   

### 4. VGG-Face
1. To evaluate Fawkes v0.3 attack: 
```sh
python3 FaceCure/eval_oblivious.py --facescrub-dir facescrub/download/ --attack-dir facescrub_fawkes3_attack/download/ --unprotected-file-match .jpeg --protected-file-match cloaked.png --classifier  NN --names-list Adam_Sandler --model vggface 
``` 

2. To evaluate Fawkes v1.0 attack: 
```sh
python3 FaceCure/eval_oblivious.py --facescrub-dir facescrub/download/ --attack-dir facescrub_fawkes_attack/download/ --unprotected-file-match .jpeg --protected-file-match cloaked.png --classifier  NN --names-list Adam_Sandler --model vggface 
``` 

3. To evaluate Lowkey attack: 
```sh
python3 FaceCure/eval_oblivious.py --facescrub-dir facescrub/download/ --attack-dir facescrub_lowkey_attack/download/ --unprotected-file-match small.png --protected-file-match attacked.png --classifier  NN --names-list Adam_Sandler --model vggface
``` 

4. To evaluate 50% Fawkes v1.0 and 50% Fawkes v0.3 attack: 
```sh
python3 FaceCure/eval_oblivious.py --facescrub-dir facescrub/download/ --attack-dir facescrub_fawkes_50_50_attack/download/ --unprotected-file-match .jpeg --protected-file-match cloaked.png --classifier  NN --names-list Adam_Sandler --model vggface 
```

Results:

Fawkes v0.3 (oblivious NN) |  Fawkes v1.0 (oblivious NN) | Lowkey (oblivious NN)| 50% Fawkes v1.0 and 50% Fawkes v0.3 (oblivious NN)
---------------------|---------------------|--------------------- |--------------------- 
```Protection rate: 0.17```|```Protection rate: 0.56``` |```Protection rate: 0.67```  |```Protection rate: 0.22```   

### 5. FaceNet 
1. To evaluate Fawkes v0.3 attack: 
```sh
python3 FaceCure/eval_oblivious.py --facescrub-dir facescrub/download/ --attack-dir facescrub_fawkes3_attack/download/ --unprotected-file-match .jpeg --protected-file-match cloaked.png --classifier  NN --names-list Adam_Sandler --model facenet  
``` 

2. To evaluate Fawkes v1.0 attack: 
```sh
python3 FaceCure/eval_oblivious.py --facescrub-dir facescrub/download/ --attack-dir facescrub_fawkes_attack/download/ --unprotected-file-match .jpeg --protected-file-match cloaked.png --classifier  NN --names-list Adam_Sandler --model facenet 
``` 

3. To evaluate Lowkey attack: 
```sh
python3 FaceCure/eval_oblivious.py --facescrub-dir facescrub/download/ --attack-dir facescrub_lowkey_attack/download/ --unprotected-file-match small.png --protected-file-match attacked.png --classifier  NN --names-list Adam_Sandler --model facenet 
```

4. To evaluate 50% Fawkes v1.0 and 50% Fawkes v0.3 attack: 
```sh
python3 FaceCure/eval_oblivious.py --facescrub-dir facescrub/download/ --attack-dir facescrub_fawkes_50_50_attack/download/ --unprotected-file-match .jpeg --protected-file-match cloaked.png --classifier  NN --names-list Adam_Sandler --model facenet  
```

Results:

Fawkes v0.3 (oblivious NN) |  Fawkes v1.0 (oblivious NN) | Lowkey (oblivious NN)| 50% Fawkes v1.0 and 50% Fawkes v0.3 (oblivious NN)
---------------------|---------------------|--------------------- |--------------------- 
```Protection rate: 0.11```|```Protection rate: 0.61``` |```Protection rate: 0.33```  |```Protection rate: 0.17```   

### 6. ArcFace 
1. To evaluate Fawkes v0.3 attack: 
```sh
python3 FaceCure/eval_oblivious.py --facescrub-dir facescrub/download/ --attack-dir facescrub_fawkes3_attack/download/ --unprotected-file-match .jpeg --protected-file-match cloaked.png --classifier  NN --names-list Adam_Sandler --model arcface 
``` 

2. To evaluate Fawkes v1.0 attack: 
```sh
python3 FaceCure/eval_oblivious.py --facescrub-dir facescrub/download/ --attack-dir facescrub_fawkes_attack/download/ --unprotected-file-match .jpeg --protected-file-match cloaked.png --classifier  NN --names-list Adam_Sandler --model arcface 
``` 

3. To evaluate Lowkey attack: 
```sh
python3 FaceCure/eval_oblivious.py --facescrub-dir facescrub/download/ --attack-dir facescrub_lowkey_attack/download/ --unprotected-file-match small.png --protected-file-match attacked.png --classifier  NN --names-list Adam_Sandler --model arcface
```

4. To evaluate 50% Fawkes v1.0 and 50% Fawkes v0.3 attack: 
```sh
python3 FaceCure/eval_oblivious.py --facescrub-dir facescrub/download/ --attack-dir facescrub_fawkes_50_50_attack/download/ --unprotected-file-match .jpeg --protected-file-match cloaked.png --classifier  NN --names-list Adam_Sandler --model arcface 
```

Results:

Fawkes v0.3 (oblivious NN) |  Fawkes v1.0 (oblivious NN) | Lowkey (oblivious NN)| 50% Fawkes v1.0 and 50% Fawkes v0.3 (oblivious NN)
---------------------|---------------------|--------------------- |--------------------- 
```Protection rate: 0.17```|```Protection rate: 0.50``` |```Protection rate: 0.50```  |```Protection rate: 0.17```   

### 7. SFace 
1. To evaluate Fawkes v0.3 attack: 
```sh
python3 FaceCure/eval_oblivious.py --facescrub-dir facescrub/download/ --attack-dir facescrub_fawkes3_attack/download/ --unprotected-file-match .jpeg --protected-file-match cloaked.png --classifier  NN --names-list Adam_Sandler --model sface 
``` 

2. To evaluate Fawkes v1.0 attack: 
```sh
python3 FaceCure/eval_oblivious.py --facescrub-dir facescrub/download/ --attack-dir facescrub_fawkes_attack/download/ --unprotected-file-match .jpeg --protected-file-match cloaked.png --classifier  NN --names-list Adam_Sandler --model sface 
``` 

3. To evaluate Lowkey attack: 
```sh
python3 FaceCure/eval_oblivious.py --facescrub-dir facescrub/download/ --attack-dir facescrub_lowkey_attack/download/ --unprotected-file-match small.png --protected-file-match attacked.png --classifier  NN --names-list Adam_Sandler --model sface 
```

4. To evaluate 50% Fawkes v1.0 and 50% Fawkes v0.3 attack: 
```sh
python3 FaceCure/eval_oblivious.py --facescrub-dir facescrub/download/ --attack-dir facescrub_fawkes_50_50_attack/download/ --unprotected-file-match .jpeg --protected-file-match cloaked.png --classifier  NN --names-list Adam_Sandler --model sface 
```

Results:

Fawkes v0.3 (oblivious NN) |  Fawkes v1.0 (oblivious NN) | Lowkey (oblivious NN)| 50% Fawkes v1.0 and 50% Fawkes v0.3 (oblivious NN)
---------------------|---------------------|--------------------- |--------------------- 
```Protection rate: 0.28```|```Protection rate: 0.94``` |```Protection rate: 0.44```  |```Protection rate: 0.17```   



The `--classifier` argument can be set to linear to train a linear classifier instead of a nearest neighbor one. 

All the above experiments can be repeated by just changing the name of the user and the attack and defense can be done for any user. 


