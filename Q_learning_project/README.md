# Q_learning_project

17.12.2022: Pour mettre à jour le modèle q_network à chaque update_interval, utilisez la fonction updateClassifier. 
Nous n'avons pas besoin de calculer la perte (loss), comme je l'ai fait manuellement dans DataWrapper.java. Les poids du modèle 
q_network sont automatiquement mis à jour en fonction de la perte avec la fonction.

q_network.updateClassifier(data.instance(0));
q_network.toString();
