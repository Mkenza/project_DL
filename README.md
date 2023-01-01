<center> <h1> Re-identification with Market 1501 </h1>
 <h2> Kenza MAKHLOUF</h2></center>


L'objectif est pouvoir identifier la personne sous plusieurs points de vue, avec le dataset MARKET 1501 [1].

Pour cela, nous allons tenté de finetune un modèle pré-entraîné avec une triplet loss proposée dans l'article [2].

**Step 1: Pré-traitement du dataset:**
* Les dossiers contiennent des fichiers et des images qu'on ne veut pas ajouter à l'entraînement, ceux-ci seront supprimés.
* On investigera le nombre d'occurence des identités, pour pouvoir prélever dans chaque batch un nombre équilibré d'instances de même identité.

**Step 2: DataLoader reid_data.py**

Afin d'optimiser l'import et l'encodage des images, nous allons créer une classe Dataset, qui permettra l'import des images dans le bon format (compatible avec le modèle pré-entraîné). On aura par la suite une fonction get_loader, qui retournera un torch DataLoader de ce dataset personnalisé.

**Step 3: Modèle model.py**

Nous allons créer une classe héritant de torch.nn.Module qui aura un modèle pré-entraîné, deux couches denses avec une fonction d'activation ReLu, Adam Optimizer et une fonction loss que nous allons créer.

La classe TripletLoss permet de définir la fonction forward_loss de notre modèle. Celle-ci compare l'exemple positif les plus loins de l'élément anchor avec l'exemple négatif (identité différente) le plus proche. Ceci permet d'éloigner les exemples négatifs les plus durs et rapprocher les exemples positifs.

**Step 4: Entraînement**

Nous allons entraîner le modèle avec les optios définis dans le dictionnaire options, montrer la loss sur l'entraînement, puis calculer le recall@1 sur la validation (le test). Les informations sur la loss, le nombre d'itérations sont disponible dans le logger.
```bash
python train.py
```
Le code à exécuter est disponible sur le notebook, mais peut aussi être exécuter dans un terminal (train.py avec les arguments dans option, exécuter train.py --h pour plus d'informations)

**Step 5: Evaluation**

Une fonction recall permet le calcul du recall@1 souvent utilisé dans ce contexte de recherche. Il correspond aux de bonnes réponses classées en premier par le modèle. Ainsi, une bonne réponse dans ce contexte est une image d'une personne de même identité.

On peut aussi visualiser les images les plus proches d'un élément selon le modèle, avec la fonction similar_images. 
![Alt text](similarities.PNG?raw=true "Title")


Ou encore visualiser l'espace encodé des images pour comprendre la représentation du modèle.

--------------------------------------
[1] Liang Zheng, Liyue Shen, Lu Tian, Shengjin Wang, Jingdong Wang, and Qi Tian.
Scalable person re-identification: A benchmark. In Computer Vision, IEEE Inter-
national Conference on, 2015.
10

[2] Alexander Hermans, Lucas Beyer, and Bastian Leibe. In defense of the triplet loss
for person re-identification, 2017, https://arxiv.org/pdf/1703.07737.pdf
