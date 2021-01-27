## Les réseaux convolutifs

### Limites des réseaux "standards"


Les réseaux de neurones "standard" et **denses** ont d'indéniables qualités,
nous permettant d'effectuer aisément et efficacement de la **classification**
ou de la **régression** sur des données dont les **caractéristiques sont
*hétérogènes*.
Par exemple, si les données en entrée sont celle de la base des vins, où chaque
vin est caractérisé par une acidité, un ph... c'est vers ce type de réseau qu'il
faut se tourner.

En revanche, dans le cas de données de type sonore ou d'images, la situation
change un peu :

- le nombre de caracteristiques est souvent très grand
- toutes les caractéristiques sont de même type
- la position d'un pattern dans les données peut varier.

Ceci pose des problèmes aux réseaux "standard"

#### Explosion du nombre de paramètres

dans un problème, même relativement complexe de données hétérogènes, le nombre
de caractéristiques reste souvent relativement faible. Imaginons que l'on
cherche à classifier des êtres humains, on prendra un certain nombre
d'informations sur eux (taille, poid, salaire, joie de vivre,...).
Même en cherchant à travailler sur des données très précises, on obtiendra
souvent au maximum une centaine de caracteristiques.

Dans un réseau "standard" (dense), chaque neurone de la couche d'entrée aura
donc au maximum une centaine de paramètres à régler
(ses poids vers chaque caractéristique). Disons que l'on ait 10 neurones sur la
couche d'entrée, cela fera un millier de paramètres à régler (pour la seule
couche d'entrée)

Si maintenant on travaille sur des images de visages pour les reconnaitre,
le nombre de caractéristique devient rapidement très grand. Imaginons que l'on
travaille avec des images de taille 256x256 (ce qui reste modeste), le nombre
de caractéristiques en entrée passe à 65536.

Un réseau dense travaillant sur ces images est composé de neurones qui regardent
tous tous les pixels, et définit le poid relatif de chaque pixel pour lui.
Un tel réseau, avec une première couche toujours composée de 10 neurones aura
maintenant 655360 paramètres libres à régler.

Ceci, si la base d'exemples n'est pas assez grande, risque fort de conduire à
du **sur apprentissage** (ou **overfitting** en anglais).

#### Importance des structures locales

Prenons l'exemple d'une base très classique :
la base MNIST dont voici quelques exemples.

![base Mnist](mnist_sample.png)

Celle ci est composée de chiffres manuscrits, de taille similaires et centrés
dans des images de 28x28 pixels. L'objectif est ici de concevoir un classifieur
capable de reconnaitre quel chiffre est présenté (10 possibilités)

Dans de telles images, on peut penser que pour reconnaitre les chiffres, il
pourrait être interessant de détecter des motifs simples
(des portions verticales, des portions arrondies, des portions de telle ou telle
orientation) puis que la composition de ces portions conduisent à la
reconnaissance d'un chiffre.

De la même façon, pour la reconnaissance de visage, on peut imaginer qu'il
serait efficace d'avoir des motifs simples (les mêmes que précédement)
qui, combinés d'une certaine manière, conduisent a reconnaitre par exemple
des cercles, et qu'une combinaison de cercles amène a penser qu'on "voit" un
oeil...

Un réseau standard ne fonctionne pas du tout comme ceci (les réseaux convolutif
le feront)


#### Non invariance en translation

Enfin, un dernier problème est lié au précédent :
Par exemple, dans l'image suivante, le même motif apparait dans 3 images
différentes, mais sa position a changée :

![tranlation](invarianceTranslation.png)

Dans ce genre d'exemple, un réseau de neurone "standard" entrainé a reconnaitre
ce motif quand il est centré ne pourra pas le reconnaitre lorsqu'il est décalé.

### Principe des Réseaux de Neurones Convolutifs

On les appelle aussi *Convolutional Neural Networks* ou **CNN** en anglais.

L'idée de base est de se concentrer sur cette idée de structure locale a
détecter.

#### La convolution

On peut imaginer une portion de couche d'un CNN comme un neurone unique qui va
parcourir les données d'entrée en n'en voyant qu'une partie à la fois et
fournira une réponse pour chaque partie qu'il voit.


Pour mieux comprendre cette idée, imaginons la suite de valeurs suivante en
entrée :

[0,1,1,-3,4,4,8, -10,-10,0]

Imaginons de plus que :
- notre neurone regarde 2 valeurs consécutives de cette
suite.
- Ses poids, fixés ici pour l'exemple seront respectivement
[-1/2, 1/2]
- son biais sera nul, et sa fonction d'activation sera l'identité

Quelle seront les sorties de ce neurone ?

Si il regarde les 2 premieres valeurs de la suite d'entrée, il effectue la somme
suivante : -1/2x0 + 1/2x1 -> 1/2

Si il regarde les 2 valeurs suivantes de la suite d'entrée, il effectue la somme
suivante : -1/2x1 + 1/2x1 -> 0

De fait, ce neurone calcule une approximation de la dérivée :
(valeur en un point - valeur au point précédent) /2

En déroulant le long de la suite, on obtient la suite suivante :
[1/2,0,-2,3.5,0, 2,-9,0 ]

On a effectué la **convolution** du signal d'entrée par le filtre [-1/2,1/2].
(en fait, c'est une *correlation* mais c'est très semblable)

Notre neurone est ici une machine capable de signaler par une sortie forte
des transitions importantes gauche/droite, par des valeurs très négatives des
transitions importantes droite/gauche et des valeurs faibles les zones sans
grande variation

Un petit résumé en image, avec un filtre différent :

![convolution](convolution.png)

On peut donc penser cette opération comme un neurone unique qui parcourt les
données en entrées. On peut également le penser comme un ensemble de neurones
qui regardent chacun une portion des données en entrée, **tous ces neurones
ayant le même ensemble de poids**.

On peut également faire le même type d'opération sur des données 2D, avec un
filtre éventuellement 2D, comme dans l'image suivante :

![convolution 2D](conv2D.png)

ou animé :

![convolution 2D bis](conv2D_animated.png)

#### Remarques sur la convolution

Ceci est moins majeur, mais je le place la quand même pour les curieux :

- la taille du filtre doit etre fixée par le programmeur (on peut créer un
filtre comme [1/4, 1/4, 1/4, 1/4] qui fait une moyenne des poids des 4 entrées
consécutives ou un filtre comme [1/3, 1/3, 1/3] qui fait la meme chose sur
3 entrées consécutives).

- Quand le neurone parcourt l'image, on peut choisir de le déplacer d'une
position à la suivante ou de lui faire sauter une ou plusieurs positions.
c'est le **pas** de la convolution
(ou **stride** en anglais quand on codera ceci).

- Quand on déplace le neurone, certaines positions à droite ne peuvent être
calculées (la dernière dans le cas de notre filtre de longueur 2).
La sortie est donc plus courte que l'entrée, a moins qu'on ajoute des zéros
autour pour que toutes les positions puissent être calculées
(on parle alors de **padding**). Sinon, si le filtre est de taille n, la sortie
est perd n-1 positions par rapport à l'entrée.

### Les Maps de caractéristiques

En fait, dans un CNN, une couche n'est pas composée d'un seul neurone, mais de
plusieurs. Chaque neurone va effectuer son filtrage et fournir une **map**
en sortie.

Une illustration de ceci :

![maps](CNN_maps.png)

C'est le développeur qui choisit la profondeur de sa couche (le nombre de maps)

La couche suivante va pouvoir regarder toutes les maps de la couche précédente comme des entrées. Un de ses neurones regardera toujours une portion de taille donnée mais toute la profondeur de la couche précédente.

Notons par exemple que si notre réseau doit traiter des images RGB, on peut
considérer qu'il travaille en fait sur des entrées réparties en 3 maps (R,G,B)

**Important** : Comme dans tout réseau de neurones, les poids ne sont pas
prédéterminés mais **appris**.

Dans un contexte d'images en entrées, c'est bien lors de l'apprentissage que,
par exemple, la première couche va décider que sa première map va se spécialiser
dans la détection de structures verticales, la seconde map dans la détection de
structures obliques.
Ceci est guidé par les performances en apprentissage.

L'extraction des caractéristiques analysées par le réseau sera donc induit par
les données et l'objectif du réseau. C'est la raison principale du succès des
CNN dans de nombreuses applications (notamment en traitement d'images)

### Vers une extraction de caractéristiques plus sémantiques...

Si l'on observe le fonctionnement d'un CNN a plusieurs couches, on peut
maintenant avoir l'intuition suivante :
- la première couche détecte des caractéristiques bas niveau (forte intensité, faible intensité, gradient dans une direction,...)
- la couche suivante va pouvoir intégrer ces caractéristiques bas niveau.
Elle pourrait ainsi avoir une map détectant des coins, ou des plus grandes lignes verticales...
- les couches suivantes font de même, détectant des structures de plus en plus
complexes dans l'image.

Ceci pourrait conduire a une map spécilaisée dans la détection d'yeux, ou de
canards...

Plus on progresse dans le réseau, plus on extrait une information complexe
des données d'origine.

**A noter** : Imaginez un réseau entrainé a reconnaitre toutes sortes d'animaux
et qui le fasse efficacement.
Il est vraisemblable que ses premières couches soient capables d'extraire des
informations très pertinentes pour toute sorte de problèmes (reconnaissance de
visages par exemple). Il est ainsi possible de prendre un réseau pré-entrainé
pour une tâche, d'en supprimer les couches finales pour les remplacer par
d'autres (non entrainées) et de lui faire apprendre uniquement sur ces couches
finales pour qu'il devienne rapidement efficace dans la reconnaissance de
visages. C'est ce que l'on appelle **Transfert Learning**)

### Les couches de pooling (sous echantillonage)
