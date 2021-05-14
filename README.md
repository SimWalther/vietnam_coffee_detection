# Une IA pour surveiller les engagements de zéro déforestation 

## Énoncé
Dans le cadre d'un projet de collaboration avec le Centre de Recherche en Agriculture Tropicale - CIAT et le King's College London (KCL), on développe des outils exploitant des algorithmes de Machine Learning pour traiter des informations fournies par des capteurs de télédétection (e.g. par des satellites), avec l'objectif de détecter la déforestation et surveiller les changements dans l'utilisation des sols. 
Dans le cadre de ce projet, nous avons l'objectif de traiter des images fournies par des satellites pour entraîner des réseaux de neurones (Deep Learning) pour détecter des champs de café ayant remplacé la forêt au Vietnam. 
Beaucoup d’entreprises au Vietnam ont signé des accords de zéro déforestation dans leur "commodity chain". Or, leur problème est qu'actuellement ils ne savent pas comment vérifier si le produit qu'ils achètent vient d'une région déforestée récemment ou non1.
En collaboration avec le centre de recherche sur l'Agriculture Tropicale (CIAT) au VietNam, nous aurons accès à des annotations de champs de café pour entraîner des réseaux de neurones et nous travaillerons sur la mise en place d'un système de vérification de non-déforestation d'une région.
Plus concrètement, il s'agit de:
  - Exploiter des images satellite (Landsat 8) historiques (p.ex. à partir de 2013) pour détecter la deforestation au Vietnam, en utilisant les techniques du Machine Learning 
  - Comparer les résultats avec des cartes existantes et utiliser d'autres sources de données pour affiner la détection de la déforestation   
  - Utiliser et adapter des modèles Machine Learning de détection des champs de café et autres "commodities" pour évaluer si ces champs sont le produit d'une déforestation après 2013 ou non.

## Installer les dépendances
Il est possible d'installer les dépendances requises grâce au fichier requirements.txt.

Il suffira de faire :

```
pip install -r requirements.txt
```
