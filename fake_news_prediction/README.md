Authors: GAHN ALIOUNE BADARA BA

Pour construire Notre modèle nous avons 3 Notebooks contenant du code: 
Ces Notebooks se trouvent dans le dossier Codes. 
-Le fichier "Textminingkaggle.ipynb" contenant les differents traitements orientés Text Mining. 
    Aprés execution de ce fichier, on a en sortie deux Dataframe (dfxtraintext et dfxtesttext) enregistrés sous format csv.
    Ces dataframes contiennent les features Text Mining retenus pour faire les prédictions 
    
- -Le fichier "networkminingkaggle.ipynb" contenant les differents traitements orientés Network Mining. 
    Aprés execution de ce fichier, on a en sortie deux Dataframe (dfxtrainnetwork et dfxtestnetwork) enregistrés sous format csv.
    Ces dataframes contiennent les features Network Mining retenus pour faire les prédictions 

- Enfin le fichier le fichier "predictionkaggle.ipynb" contient le code pour faire les prédictions. En entrée de ce fichier,
nous chargeons les fichiers csv obtenus en sortie des fichiers "Textminingkaggle.ipynb" et "networkminingkaggle.ipynb" . 
Nous regroupons dans ce fichier les dataframes text et train pour faire les prédictions. 

Le fichier "ArchictectureprojetKaggle.pdf" décrit l'architecture du projet.

En text mining, nous avons utilisé un word2vec pré-entrainé par Google. Il faut donc le télécharger sur le lien github suivant:

        https://github.com/eyaler/word2vec-slim/blob/master/GoogleNews-vectors-negative300-SLIM.bin.gz

Nous avons modifié le fichier "labels_training.txt" pour faciliter son importation avec notre code. Nous avons juste enlevé la première ligne ('doc,class'). Nous le joignons dans notre dossier (vous pouvez le modifier dans votre fichier si vous préferez).

Hormis cela, nous utilisons les fichiers téléchargés sur le kaggle de la compétition. 

En Network mining, il faut installer la librairie python-louvain pour faire tourner la méthode louvain pour trouver les communautés.



Librairies à installer:
_ pandas
_ numpy
_ pickle
_ sklearn
_ matplotlib.pyplot
_ networkx
- os
- re
-community
