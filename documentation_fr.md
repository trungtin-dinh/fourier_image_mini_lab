## Table des matières

1. [La Transformée de Fourier Discrète 2D](#1-la-transformee-de-fourier-discrete-2d)
2. [Filtrage Fréquentiel : Cadre Général](#2-filtrage-frequentiel-cadre-general)
3. [Filtres Fréquentiels Idéaux](#3-filtres-frequentiels-ideaux)
4. [Filtres Gaussiens dans le Domaine Fréquentiel](#4-filtres-gaussiens-dans-le-domaine-frequentiel)
5. [Filtre de Butterworth 2D](#5-filtre-de-butterworth-2d)
6. [Filtre à Rejet de Tache (Notch)](#6-filtre-a-rejet-de-tache-notch)
7. [Filtres Non-Linéaires Spatiaux](#7-filtres-non-lineaires-spatiaux)
8. [Manipulation de la Phase de Fourier](#8-manipulation-de-la-phase-de-fourier)
9. [Modèle de Dégradation et Bruit](#9-modele-de-degradation-et-bruit)
10. [Débruitage : Méthodes Avancées](#10-debruitage-methodes-avancees)
11. [Déconvolution et Restauration d'Image](#11-deconvolution-et-restauration-dimage)

---

## 1. La Transformée de Fourier Discrète 2D

### 1.1 Définition

Soit $f(x, y)$ une image de taille $H \times W$ à valeurs réelles, avec $x \in \{0, \ldots, W-1\}$ et $y \in \{0, \ldots, H-1\}$. La **Transformée de Fourier Discrète 2D** (DFT 2D) est définie par :

$$F(u, v) = \sum_{x=0}^{W-1} \sum_{y=0}^{H-1} f(x, y)\, e^{-j2\pi\left(\frac{ux}{W} + \frac{vy}{H}\right)}$$

où $(u, v) \in \{0, \ldots, W-1\} \times \{0, \ldots, H-1\}$ sont les coordonnées fréquentielles discrètes. La transformée inverse est :

$$f(x, y) = \frac{1}{WH} \sum_{u=0}^{W-1} \sum_{v=0}^{H-1} F(u, v)\, e^{j2\pi\left(\frac{ux}{W} + \frac{vy}{H}\right)}$$

$F(u, v)$ est un nombre complexe : $F(u, v) = |F(u, v)|\, e^{j\angle F(u,v)}$, où $|F(u, v)|$ est le **module** (amplitude spectrale) et $\angle F(u, v)$ est la **phase spectrale**.

### 1.2 Interprétation Physique

Chaque coefficient $F(u, v)$ mesure la **corrélation** de l'image $f$ avec l'onde plane complexe $e^{j2\pi(ux/W + vy/H)}$ — une sinusoïde 2D de fréquence spatiale $(u/W, v/H)$ cycles par pixel dans les directions $x$ et $y$ respectivement. Un grand $|F(u, v)|$ signifie que l'image contient des structures fortement périodiques à cette fréquence.

Les basses fréquences (autour de $(u, v) = (0, 0)$) correspondent aux variations lentes (fond, éclairage global). Les hautes fréquences correspondent aux détails fins, contours et bruit.

### 1.3 Centrage et `fftshift`

Par convention de NumPy, $F(0, 0)$ est la composante DC (valeur moyenne de l'image), et les fréquences élevées se trouvent aux extrémités du tableau. Pour visualiser le spectre avec les basses fréquences au centre (convention utilisée dans l'app), on applique `np.fft.fftshift` : cette opération effectue une translation circulaire de $\lfloor W/2 \rfloor$ colonnes et $\lfloor H/2 \rfloor$ lignes, plaçant la composante DC en $(W/2, H/2)$.

Dans le spectre centré, la **distance radiale à l'origine** d'un point $(u, v)$ est :

$$d(u, v) = \sqrt{\left(u - \frac{W}{2}\right)^2 + \left(v - \frac{H}{2}\right)^2}$$

exprimée en pixels fréquentiels. C'est cette distance $d$ qui est utilisée pour définir tous les masques de filtrage fréquentiel de l'app.

### 1.4 Spectre de Module et de Phase

Le **spectre de module** affiché dans l'app est $\log(1 + |F_s(u,v)|)$, normalisé dans $[0, 1]$, où $F_s$ désigne le spectre centré. Le logarithme est indispensable : la composante DC est typiquement $10^4$ fois plus grande que les composantes hautes fréquences ; sans compression logarithmique, seule la tache centrale serait visible.

Le **spectre de phase** $\angle F_s(u, v) \in [-\pi, +\pi]$ est remappé linéairement dans $[0, 1]$ pour l'affichage. La phase encode l'information spatiale structurelle : comme le montrent les expériences de manipulation de phase (section 8), la reconnaissance des objets dans une image est bien mieux préservée par la phase que par le module.

### 1.5 La DFT 2D comme Produit de DFT 1D

La DFT 2D est **séparable** : elle peut être calculée comme une suite de DFT 1D, d'abord sur les lignes puis sur les colonnes (ou l'inverse) :

$$F(u, v) = \sum_{y=0}^{H-1} \left[\sum_{x=0}^{W-1} f(x,y)\, e^{-j2\pi ux/W}\right] e^{-j2\pi vy/H}$$

C'est cette séparabilité qui permet à l'algorithme FFT (Fast Fourier Transform) de calculer la DFT 2D de façon très efficace : le nombre d'opérations arithmétiques est proportionnel à $WH\log(WH)$ — par exemple environ 17 millions d'opérations pour une image $1024 \times 1024$ — au lieu d'environ $W^2 H^2$, soit plus d'un billion d'opérations pour la même image avec la définition naïve. L'algorithme FFT est donc indispensable en pratique dès que l'image dépasse quelques dizaines de pixels par côté.

---

## 2. Filtrage Fréquentiel : Cadre Général

### 2.1 Le Théorème de Convolution

Le résultat fondamental qui justifie le filtrage fréquentiel est le **théorème de convolution** : la convolution spatiale correspond à la multiplication fréquentielle,

$$g(x, y) = (f * h)(x, y) \quad \Longleftrightarrow \quad G(u, v) = F(u, v) \cdot H(u, v)$$

où $h(x, y)$ est la réponse impulsionnelle du filtre et $H(u, v) = \mathcal{F}\{h\}$ sa **fonction de transfert** (ou réponse en fréquence). Filtrer une image revient donc à multiplier son spectre point par point par $H(u, v)$, puis à calculer la transformée inverse.

### 2.2 Pipeline de Filtrage Fréquentiel

L'app applique systématiquement le pipeline suivant pour chaque canal de couleur $c \in \{R, G, B\}$ :

$$G_c(u, v) = \mathcal{F}\{f_c\}(u, v) \cdot H(u, v)$$

$$g_c(x, y) = \text{Re}\left[\mathcal{F}^{-1}\{G_c\}(x, y)\right]$$

La partie réelle est extraite car $f_c$ est réelle ; les faibles résidus imaginaires provenant d'erreurs d'arrondi flottant sont simplement ignorés.

**Remarque importante** : la multiplication fréquentielle est équivalente à une convolution **circulaire** (periodique) dans le domaine spatial, pas à une convolution linéaire. Pour une image de taille $H \times W$, le résultat est périodisé avec la même période. Cela introduit des **artefacts de bord** pour les filtres à forte transition : les structures du bord droit "passent" dans le bord gauche. Ces artefacts sont inhérents à la DFT et peuvent être atténués par l'extension de l'image avant transformée.

### 2.3 Masque Fréquentiel

Dans l'app, tous les filtres linéaires sont définis par un **masque** $H(u, v) \in [0, 1]^{W \times H}$, une matrice réelle, symétrique par rapport à l'origine, définie dans le domaine fréquentiel centré. La valeur $H(u, v) = 1$ signifie que la composante fréquentielle $(u, v)$ est transmise intégralement ; $H(u, v) = 0$ qu'elle est totalement supprimée.

---

## 3. Filtres Fréquentiels Idéaux

### 3.1 Filtre Passe-Bas Idéal

Le filtre passe-bas idéal (Low-pass) de rayon de coupure $D_0$ transmet toutes les fréquences dont la distance à l'origine est inférieure à $D_0$ et rejette toutes les autres :

$$H_{\text{LP}}(u, v) = \begin{cases} 1 & \text{si } d(u,v) \leq D_0 \\ 0 & \text{sinon} \end{cases}$$

Sa réponse impulsionnelle spatiale $h_{\text{LP}}(x, y) = \mathcal{F}^{-1}\{H_{\text{LP}}\}$ est une **fonction de Bessel** $J_1$ (analogue 2D de la fonction sinc 1D), qui oscille autour de zéro avec des lobes décroissants. C'est précisément cette queue oscillante qui produit le **ringing** (anneau de Gibbs) visible autour des contours dans une image filtrée par passe-bas idéal.

### 3.2 Filtre Passe-Haut Idéal

Le passe-haut idéal est simplement le complément du passe-bas :

$$H_{\text{HP}}(u, v) = 1 - H_{\text{LP}}(u, v) = \begin{cases} 0 & \text{si } d(u,v) \leq D_0 \\ 1 & \text{sinon} \end{cases}$$

Il supprime les basses fréquences (fond lisse) et conserve les hautes fréquences (contours, détails, bruit). Le résultat est une image de type "détection de contours", grise au fond et brillante aux transitions.

### 3.3 Filtres Passe-Bande et Coupe-Bande Idéaux

Le **passe-bande idéal** transmet une couronne annulaire de fréquences $[D_{\text{low}}, D_{\text{high}}]$ :

$$H_{\text{BP}}(u, v) = \begin{cases} 1 & \text{si } D_{\text{low}} \leq d(u,v) \leq D_{\text{high}} \\ 0 & \text{sinon} \end{cases}$$

Le **coupe-bande idéal** (band-stop) est son complément :

$$H_{\text{BS}}(u, v) = 1 - H_{\text{BP}}(u, v)$$

Il est utile pour supprimer des artéfacts périodiques localisés dans une bande de fréquences connue (par exemple, le bruit d'une trame de balayage).

### 3.4 Discontinuités et Artefacts de Gibbs

La transition abrupte ($0 \to 1$ en un seul pixel) des filtres idéaux produit un phénomène analogue au **phénomène de Gibbs** en 1D : des oscillations spatiales (ringing) apparaissent dans le voisinage des contours. Ces oscillations sont une conséquence directe de la troncature du spectre : en fréquence, multiplier par un disque $H_\text{LP}$ revient à convoluer la réponse impulsionnelle idéale (infinie) avec le support disque fini, ce qui crée des lobes secondaires. Les filtres Gaussien et Butterworth (sections 4 et 5) sont conçus précisément pour éviter cette discontinuité.

---

## 4. Filtres Gaussiens dans le Domaine Fréquentiel

### 4.1 Le Filtre Gaussien Passe-Bas

La **fonction de transfert gaussienne passe-bas** de paramètre $\sigma$ est :

$$H_{\text{Gauss,LP}}(u, v) = \exp\!\left(-\frac{d(u,v)^2}{2\sigma^2}\right)$$

où $d(u,v) = \sqrt{(u - W/2)^2 + (v - H/2)^2}$ est la distance à l'origine dans le spectre centré.

La fréquence de coupure à $-3\,\text{dB}$ (gain = $1/\sqrt{2}$) est $d_{3\text{dB}} = \sigma\sqrt{\ln 2} \approx 0.832\,\sigma$.

### 4.2 Propriété Remarquable : Auto-Transformée

La gaussienne est sa propre transformée de Fourier (à un facteur de normalisation près). Plus précisément, si $H_{\text{Gauss,LP}}(u,v)$ est gaussienne de variance $\sigma^2$ dans le domaine fréquentiel, alors $h_{\text{Gauss,LP}}(x, y) = \mathcal{F}^{-1}\{H_{\text{Gauss,LP}}\}$ est gaussienne de variance $(W H)/(4\pi^2\sigma^2)$ dans le domaine spatial.

Cette dualité implique : un filtre passe-bas gaussien étroit dans le domaine fréquentiel ($\sigma$ petit) correspond à un noyau de convolution spatial large (beaucoup de flou), et vice versa. Le **produit des largeurs** fréquentielle et spatiale est constant — c'est l'expression du **principe d'incertitude de Heisenberg** en traitement du signal 2D.

### 4.3 Filtre Gaussien Passe-Haut

$$H_{\text{Gauss,HP}}(u, v) = 1 - H_{\text{Gauss,LP}}(u, v) = 1 - \exp\!\left(-\frac{d(u,v)^2}{2\sigma^2}\right)$$

Ce filtre est à zéro pour $d = 0$ (composante DC supprimée) et tend vers 1 pour les hautes fréquences. Il produit une mise en évidence des contours sans le ringing du passe-haut idéal, car la transition est continue et infiniment dérivable.

### 4.4 Avantage sur les Filtres Idéaux

La gaussienne est la **seule fonction** qui soit à la fois à décroissance gaussienne dans le domaine spatial et dans le domaine fréquentiel. Cela lui confère une propriété unique : elle minimise simultanément la localisation spatiale et fréquentielle (minimum du produit $\Delta x \cdot \Delta u$ dans le principe d'incertitude). En pratique, cela signifie que le filtre gaussien ne produit **aucun artefact de ringing** et est entièrement sans discontinuités.

---

## 5. Filtre de Butterworth 2D

### 5.1 Définition

La **fonction de transfert de Butterworth passe-bas** d'ordre $n$ et de rayon de coupure $D_0$ est :

$$H_{\text{BW,LP}}(u, v) = \frac{1}{1 + \left(\dfrac{d(u,v)}{D_0}\right)^{2n}}$$

Au rayon de coupure $d = D_0$ : $H_{\text{BW,LP}} = 1/2$, soit $-3\,\text{dB}$ (gain à mi-puissance). À l'origine : $H = 1$. À l'infini : $H \to 0$.

Le filtre passe-haut de Butterworth est son complément :

$$H_{\text{BW,HP}}(u, v) = 1 - H_{\text{BW,LP}}(u, v) = \frac{\left(\dfrac{d(u,v)}{D_0}\right)^{2n}}{1 + \left(\dfrac{d(u,v)}{D_0}\right)^{2n}}$$

### 5.2 Rôle de l'Ordre $n$

L'ordre $n$ contrôle la **pente de la transition** entre bande passante et bande atténuée :

- Pour $n = 1$ : transition très progressive, comparable à un filtre RC du premier ordre.
- Pour $n \to \infty$ : le filtre converge vers le filtre passe-bas idéal à disque.

La pente asymptotique en décibels croît comme $-20n\,\text{dB/décade}$ dans la bande atténuée, identiquement au filtre de Butterworth 1D analogique d'où il est issu. Augmenter $n$ donne une transition plus nette, mais introduit progressivement du ringing (les lobes secondaires de la réponse impulsionnelle spatiale augmentent).

### 5.3 Compromis Butterworth vs Gaussien vs Idéal

Les trois familles de filtres passe-bas représentent un spectre continu de compromis :

| Propriété | Idéal | Butterworth ($n$ élevé) | Gaussien |
|---|---|---|---|
| Transition | Abrupte | Raide mais continue | Progressive |
| Ringing spatial | Fort (Gibbs) | Modéré | Nul |
| Monotonie | Non (oscillations) | Oui | Oui |
| Localisation spatiale | Mauvaise | Bonne | Optimale |

Le filtre de Butterworth est dit à **réponse maximalement plate** (Butterworth, 1930) dans la bande passante : toutes les dérivées de $|H_{\text{BW,LP}}|$ jusqu'à l'ordre $2n-1$ sont nulles en $d = 0$, ce qui garantit l'absence de ripple dans la bande passante.

---

## 6. Filtre à Rejet de Tache (Notch)

### 6.1 Motivation

Les bruits périodiques dans les images (interférence de balayage, trame d'impression offset, artéfacts de compression JPEG) se manifestent dans le spectre de Fourier par des **taches brillantes isolées** (pics) symétriques par rapport à l'origine. Un filtre notch supprime sélectivement ces pics sans affecter le reste du spectre.

### 6.2 Définition

Un filtre notch centré sur les positions $\pm(u_0, v_0)$ de rayon $R$ est défini par :

$$H_{\text{notch}}(u, v) = \begin{cases} 0 & \text{si } d_1(u,v) \leq R \text{ ou } d_2(u,v) \leq R \\ 1 & \text{sinon} \end{cases}$$

où $d_1(u,v) = \sqrt{(u-u_0)^2 + (v-v_0)^2}$ et $d_2(u,v) = \sqrt{(u+u_0)^2 + (v+v_0)^2}$ sont les distances aux deux positions conjuguées. La symétrie $\pm(u_0, v_0)$ est obligatoire pour que la réponse impulsionnelle spatiale reste **réelle** (propriété de symétrie hermitienne de la DFT d'un signal réel).

### 6.3 Lien avec le Bruit Périodique

Un bruit purement sinusoïdal $n(x,y) = A\cos(2\pi(u_0 x/W + v_0 y/H))$ a un spectre de Fourier composé exactement de deux deltas de Dirac en $\pm(u_0, v_0)$. Le filtre notch idéal annule ces deux deltas et supprime parfaitement le bruit sans toucher le reste du signal — à condition que les pics de bruit n'aient pas d'énergie spectrale qui se recouvre avec le signal utile.

---

## 7. Filtres Non-Linéaires Spatiaux

Les filtres de cette section opèrent **directement dans le domaine spatial** et sont intrinsèquement non-linéaires : ils ne peuvent pas être exprimés comme une convolution avec un noyau fixe, et n'ont donc pas de fonction de transfert $H(u,v)$ au sens LTI.

### 7.1 Filtre Médian

Pour chaque pixel $(x_0, y_0)$, le filtre médian de taille $k \times k$ remplace la valeur par la **médiane** des $k^2$ valeurs dans le voisinage :

$$g(x_0, y_0) = \text{médiane}\!\left\{\, f(x_0 + i,\, y_0 + j) : (i,j) \in \left[-\lfloor k/2 \rfloor, \lfloor k/2 \rfloor\right]^2 \right\}$$

La médiane est une **statistique d'ordre** : elle dépend de la valeur centrale parmi les valeurs triées, ce qui la rend robuste aux valeurs aberrantes. Un pixel de bruit impulsionnel (sel et poivre) isolé dans le voisinage est un outlier parmi $k^2 - 1$ valeurs normales ; il n'influence pas la médiane dès que $k \geq 3$. En revanche, une moyenne arithmétique propagerait cette valeur aberrante au pixel de sortie.

**Préservation des contours** : si au plus $\lfloor k^2/2 \rfloor$ pixels du voisinage ont traversé un contour, la médiane reste dans le même côté que le pixel central, préservant la transition spatiale. Un filtre moyen de même taille diluerait le contour sur $k$ pixels.

### 7.2 Filtre Minimum (Érosion)

$$g(x_0, y_0) = \min_{(i,j) \in \mathcal{W}} f(x_0 + i,\, y_0 + j)$$

Le filtre minimum est l'opération morphologique d'**érosion** avec un élément structurant carré de taille $k \times k$. Il contracte les régions claires et dilate les régions sombres. En pratique, il assombrit l'image et élargit les zones sombres. Il est notamment utilisé pour supprimer les pixels brillants isolés (bruit "sel") et pour l'analyse morphologique de forme.

### 7.3 Filtre Maximum (Dilatation)

$$g(x_0, y_0) = \max_{(i,j) \in \mathcal{W}} f(x_0 + i,\, y_0 + j)$$

Le filtre maximum est l'opération morphologique de **dilatation**. Il éclaircit l'image et élargit les zones claires. Il supprime les pixels sombres isolés (bruit "poivre"). La composition dilatation–érosion donne la **fermeture morphologique** ; l'érosion–dilatation donne l'**ouverture morphologique**, opérations fondamentales de la morphologie mathématique.

### 7.4 Filtre Bilatéral

Le filtre bilatéral (Tomasi & Manduchi, 1998) est un filtre de lissage spatial qui préserve les contours en pondérant chaque voisin non seulement par sa **distance spatiale** mais aussi par sa **similitude radiométrique** (différence de valeur) avec le pixel central :

$$g(x_0, y_0) = \frac{1}{W_p} \sum_{(i,j) \in \mathcal{W}} f(x_0+i,\, y_0+j)\, k_s(i, j)\, k_r\!\left(f(x_0+i,\, y_0+j) - f(x_0, y_0)\right)$$

où la **constante de normalisation** est $W_p = \sum_{(i,j)} k_s(i,j)\, k_r(f(x_0+i,y_0+j) - f(x_0,y_0))$, et les deux noyaux sont gaussiens :

$$k_s(i, j) = \exp\!\left(-\frac{i^2 + j^2}{2\sigma_s^2}\right), \qquad k_r(\Delta) = \exp\!\left(-\frac{\Delta^2}{2\sigma_r^2}\right)$$

Le paramètre $\sigma_s$ (sigma spatial) contrôle l'étendue spatiale du filtre. Le paramètre $\sigma_r$ (sigma radiométrique ou sigma couleur) contrôle la sélectivité en intensité : un pixel dont la valeur diffère de plus de $2\sigma_r$ du pixel central reçoit un poids quasi-nul, ce qui empêche le lissage de traverser les contours.

**Propriété clé** : dans les régions homogènes, $k_r \approx 1$ pour tous les voisins, et le filtre bilatéral se réduit à un filtre gaussien ordinaire. Au voisinage d'un contour, les pixels de l'autre côté du contour ont $k_r \approx 0$ et ne participent pas à la moyenne — le contour est ainsi préservé.

---

## 8. Manipulation de la Phase de Fourier

### 8.1 Décomposition Amplitude–Phase

Tout spectre $F(u, v) \in \mathbb{C}$ s'écrit en coordonnées polaires :

$$F(u, v) = A(u, v)\, e^{j\phi(u, v)}$$

où $A(u, v) = |F(u, v)| \geq 0$ est l'**amplitude spectrale** et $\phi(u, v) = \angle F(u, v) \in (-\pi, +\pi]$ est la **phase spectrale**. Les expériences de manipulation de phase révèlent le rôle respectif de ces deux composantes dans la perception visuelle.

### 8.2 Reconstruction Phase Seule

$$G_{\text{phase}}(u, v) = e^{j\phi(u, v)} \qquad \text{(amplitude uniformément remplacée par 1)}$$

$$g_{\text{phase}}(x, y) = \mathcal{F}^{-1}\{G_{\text{phase}}\}(x, y)$$

Toute l'information d'amplitude est supprimée ; seule la structure de phase est conservée. La reconstruction $g_{\text{phase}}$ est normalisée pour l'affichage. Résultat empirique : les **contours et la structure géométrique** des objets sont clairement reconnaissables, bien que les niveaux de gris et les textures soient altérés. Cela démontre que la **phase encode l'information structurelle** (positions des discontinuités, alignements).

### 8.3 Reconstruction par Amplitude Seule

$$G_{\text{amp}}(u, v) = A(u, v) \qquad \text{(phase uniformément mise à zéro)}$$

$$g_{\text{amp}}(x, y) = \mathcal{F}^{-1}\{A(u,v)\}(x, y)$$

En annulant toute la phase, on perd la structure spatiale. Comme $A(u,v) \geq 0$ est réel, sa transformée inverse est à symétrie hermitienne et présente des artefacts caractéristiques (concentration de l'énergie autour du centre spatial). L'image est méconnaissable. Ce résultat — symétrique de 8.2 — confirme que l'amplitude contient les **statistiques spectrales** (distribution d'énergie par fréquence) mais pas l'organisation spatiale.

### 8.4 Reconstruction à Phase Aléatoire

La phase est remplacée par un champ uniforme aléatoire $\phi_{\text{rand}}(u, v) \sim \mathcal{U}(-\pi, +\pi)$, indépendant pour chaque coefficient :

$$G_{\text{rand}}(u, v) = A(u, v)\, e^{j\phi_{\text{rand}}(u, v)}$$

Le spectre d'amplitude original est conservé, donc la **distribution statistique** des niveaux de gris est préservée (même histogramme), mais toute la structure spatiale est détruite. L'image reconstruite ressemble à un bruit coloré dont le spectre de puissance $|A(u,v)|^2$ correspond à l'original.

### 8.5 Quantification de la Phase

La phase est quantifiée sur $L$ niveaux uniformément répartis dans $(-\pi, +\pi]$ :

$$\phi_{\text{quant}}(u, v) = -\pi + \Delta\phi \cdot \text{round}\!\left(\frac{\phi(u,v) + \pi}{\Delta\phi}\right), \qquad \Delta\phi = \frac{2\pi}{L}$$

Pour $L = 2$ : seuls deux niveaux $\{-\pi, 0\}$ sont possibles. Pour $L = 8$ ou plus : la structure visuelle est presque entièrement préservée malgré la forte quantification. Cela indique que la phase n'a pas besoin d'être représentée avec une précision arbitraire pour rester informative — quelques bits de résolution angulaire suffisent pour reconnaître une scène.

### 8.6 Bruit de Phase

Un bruit gaussien est ajouté à la phase :

$$\phi_{\text{noisy}}(u, v) = \phi(u, v) + \sigma_\phi\, \eta(u, v), \qquad \eta(u, v) \overset{\text{iid}}{\sim} \mathcal{N}(0, 1)$$

Pour $\sigma_\phi$ faible devant $\pi$, l'image reste reconnaissable. Pour $\sigma_\phi \gg \pi$, la phase est essentiellement aléatoire et l'image dégénère comme dans la reconstruction à phase aléatoire (section 8.4).

### 8.7 Rampe de Phase Linéaire (Décalage Spatial)

La propriété de décalage de la DFT stipule que multiplier $F(u,v)$ par $e^{-j2\pi(u\Delta x/W + v\Delta y/H)}$ revient à translater $f(x,y)$ de $(\Delta x, \Delta y)$ pixels dans le domaine spatial :

$$\mathcal{F}^{-1}\!\left\{F(u,v)\, e^{-j2\pi\left(\frac{u\,\Delta x}{W} + \frac{v\,\Delta y}{H}\right)}\right\}(x, y) = f(x - \Delta x,\, y - \Delta y)$$

Ajouter une rampe linéaire à la phase, $\phi_{\text{ramp}}(u,v) = \phi(u,v) - 2\pi(u\,\Delta x/W + v\,\Delta y/H)$, produit donc une **translation cyclique** de l'image, sans aucune modification de l'amplitude spectrale. L'image translatée est "enroulée" (ce qui sort d'un bord réapparaît de l'autre). C'est la démonstration la plus directe que la phase encode la **position spatiale** de l'information.

---

## 9. Modèle de Dégradation et Bruit

### 9.1 Modèle Linéaire Général de Dégradation

Le modèle standard de dégradation d'image utilisé en restauration est :

$$f_{\text{deg}}(x, y) = (h * f)(x, y) + n(x, y)$$

où $f(x, y)$ est l'image originale, $h(x, y)$ est la **réponse impulsionnelle du système de dégradation** (PSF — Point Spread Function), $*$ désigne la convolution, et $n(x, y)$ est un bruit additif. Dans le domaine de Fourier :

$$F_{\text{deg}}(u, v) = H_{\text{PSF}}(u, v) \cdot F(u, v) + N(u, v)$$

L'objectif de la restauration est d'estimer $F(u,v)$ à partir de $F_{\text{deg}}(u,v)$, en connaissant (ou estimant) $H_{\text{PSF}}$ et les statistiques de $N$.

### 9.2 Bruit Gaussien Additif

Le bruit gaussien additif modélise le bruit thermique des capteurs :

$$f_{\text{noisy}}(x, y) = f(x, y) + \eta(x, y), \qquad \eta(x, y) \overset{\text{iid}}{\sim} \mathcal{N}(0, \sigma_n^2)$$

En pratique, avec les images normalisées dans $[0,1]$, le paramètre $\sigma_n$ contrôle le niveau de bruit : $\sigma_n = 0.01$ est quasi-imperceptible, $\sigma_n = 0.1$ est fortement bruité.

La densité spectrale de puissance du bruit blanc gaussien est **plate** : la puissance moyenne de chaque coefficient fréquentiel $N(u,v)$ vaut $W H \sigma_n^2$, quelle que soit la fréquence $(u,v)$. Le bruit est uniformément distribué dans tout le spectre — c'est précisément ce qui le rend difficile à éliminer sans dégrader le signal.

### 9.3 Bruit Sel et Poivre

Le bruit sel et poivre modélise des pixels défectueux ou des erreurs de transmission : une fraction $\rho$ des pixels est remplacée par la valeur maximale (sel, $= 1$) ou minimale (poivre, $= 0$), de façon indépendante :

$$f_{\text{sp}}(x, y) = \begin{cases} 1 & \text{avec probabilité } \rho/2 \\ 0 & \text{avec probabilité } \rho/2 \\ f(x, y) & \text{avec probabilité } 1 - \rho \end{cases}$$

Contrairement au bruit gaussien, les pixels corrompus sont des **outliers extrêmes** à valeur connue. Le filtre médian (section 7.1) est le débruiteur optimal pour ce type de bruit.

### 9.4 Point Spread Function (PSF) : Flou Gaussien

La PSF gaussienne modélise le flou de mise au point ou le flou de diffusion optique :

$$h_{\text{Gauss}}(x, y) = \frac{1}{2\pi\sigma^2} \exp\!\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$$

En pratique, elle est discrétisée sur une grille $k \times k$ avec $k$ impair et normalisée pour que $\sum_{x,y} h(x,y) = 1$ (conservation de l'énergie). La convolution avec $h_{\text{Gauss}}$ est appliquée par `fftconvolve` en mode `"same"`, ce qui correspond à une convolution linéaire tronquée à la taille de l'image d'entrée.

### 9.5 Point Spread Function : Flou de Mouvement

La PSF de mouvement modélise le flou cinétique dû au déplacement de la caméra ou du sujet pendant l'exposition. Elle est approximée par un segment de droite de longueur $L$ et d'angle $\theta$ :

$$h_{\text{motion}}(x, y) = \frac{1}{L}\, \mathbf{1}\!\left[(x, y) \text{ sur le segment de longueur } L \text{ et d'angle } \theta\right]$$

Dans le domaine de Fourier, cette PSF a une forme sinc orientée perpendiculairement à la direction de mouvement, avec des zéros régulièrement espacés qui rendent la déconvolution difficile (amplification de bruit aux fréquences annulées par $H_{\text{PSF}}$).

---

## 10. Débruitage : Méthodes Avancées

### 10.1 Variation Totale (TV)

La **régularisation par variation totale** (Rudin, Osher & Fatemi, 1992) estime l'image restaurée $g$ en minimisant un problème variationnel qui équilibre la fidélité aux données et la régularité de la solution :

$$\hat{g} = \arg\min_{g} \left\{ \frac{1}{2}\|g - f_{\text{noisy}}\|_2^2 + \lambda \cdot \text{TV}(g) \right\}$$

La **variation totale isotrope** est définie par :

$$\text{TV}(g) = \sum_{x,y} \|\nabla g(x,y)\|_2 = \sum_{x,y} \sqrt{\left(\frac{\partial g}{\partial x}\right)^2 + \left(\frac{\partial g}{\partial y}\right)^2}$$

Le premier terme $\|g - f_{\text{noisy}}\|_2^2$ est le terme de **fidélité** : il pénalise les solutions qui s'écartent de l'image bruitée. Le second terme $\text{TV}(g)$ est le terme de **régularisation** : il pénalise les variations spatiales importantes. Le paramètre $\lambda$ (weight dans l'app) contrôle le compromis.

La propriété fondamentale de la régularisation TV est la **préservation des contours francs** : contrairement à la régularisation par la somme des carrés du gradient (qui pénalise $\sum_{x,y}\|\nabla g\|^2$ et produit un lissage gaussien progressif), la régularisation par la somme des **normes** du gradient (terme $\text{TV}(g)$) favorise des solutions constantes par morceaux avec des transitions nettes — car annuler le gradient dans une région coûte peu, alors qu'une valeur de gradient élevée est fortement pénalisée. L'algorithme de résolution est Chambolle (2004), une méthode de descente de sous-gradient duale.

### 10.2 Non-Local Means (NLM)

La méthode **Non-Local Means** (Buades, Coll & Morel, 2005) généralise le filtre bilatéral en comparant non plus des pixels individuels mais des **patchs** (voisinages) centrés sur chaque pixel :

$$g(x_0, y_0) = \frac{1}{Z(x_0, y_0)} \sum_{(x, y) \in \Omega} f(x, y)\, \exp\!\left(-\frac{\|P(x_0, y_0) - P(x, y)\|_{2,a}^2}{h^2}\right)$$

où $P(x, y)$ désigne le patch de taille $p \times p$ centré en $(x,y)$, $\|\cdot\|_{2,a}^2$ est la norme $\ell^2$ pondérée par un noyau gaussien de paramètre $a$, $h$ est le paramètre de filtrage (bande passante), et $Z(x_0, y_0) = \sum_{(x,y)} \exp(-\|\ldots\|^2/h^2)$ est la constante de normalisation.

La similitude entre patchs mesure la **ressemblance de texture locale** : deux pixels ayant un voisinage similaire (même texture, même orientation) recevront un poids élevé et se moyenneront, même s'ils sont spatialement éloignés dans l'image. C'est la force de NLM : il exploite la **redondance non-locale** de l'image (le fait que les textures naturelles se répètent).

Les paramètres clés sont : la taille du patch $p$ (patch size), la distance maximale de recherche (patch distance), et $h$ qui contrôle la sélectivité de la comparaison de patchs. L'implémentation naïve compare chaque pixel avec tous les autres pixels de l'image, ce qui représente un coût de calcul proportionnel à $W^2 H^2 p^2$ — prohibitif pour les images de grande taille. La version `fast_mode=True` utilise une approximation par **intégrale de patches** (calcul des distances entre patchs via des sommes cumulées 2D, de façon analogue à l'intégrale d'image décrite pour Kuwahara), ce qui réduit le coût à $W H p^2$ — proportionnel au nombre de pixels, indépendamment de la fenêtre de recherche.

### 10.3 Débruitage par Ondelettes

Les **ondelettes** (wavelets) fournissent une représentation **multi-résolution** de l'image : la transformée en ondelettes 2D décompose $f$ en une série de sous-bandes de détails (horizontaux, verticaux, diagonaux) à différentes échelles, plus une sous-bande d'approximation basse fréquence.

Pour une image bruitée $f = s + \eta$ où $\eta \sim \mathcal{N}(0, \sigma_n^2)$ :

- Les **coefficients d'ondelettes du signal** $s$ sont concentrés (peu nombreux et de grande valeur) pour les images naturelles.
- Les **coefficients d'ondelettes du bruit** $\eta$ sont distribués uniformément sur toutes les sous-bandes avec une variance $\sigma_n^2$.

Cette **parcimonie** des coefficients du signal est le principe fondamental de l'approche ondelettes : un **seuillage** des coefficients (conserver les grands, mettre les petits à zéro) élimine sélectivement le bruit en préservant le signal.

L'app utilise la méthode **BayesShrink** (Chang, Yu & Vetterli, 2000), qui estime adaptativement le seuil optimal pour chaque sous-bande à partir de la variance locale des coefficients :

$$\hat{\sigma}_s = \sqrt{\max\!\left(0, \hat{\sigma}_y^2 - \sigma_n^2\right)}, \qquad \tau_k = \frac{\sigma_n^2}{\hat{\sigma}_{s,k}}$$

où $\hat{\sigma}_y^2$ est la variance des coefficients observés dans la sous-bande $k$ et $\hat{\sigma}_{s,k}$ est l'estimation de la variance du signal dans cette sous-bande. Ce seuil minimise l'erreur quadratique moyenne (MSE) Bayésienne sous une prior de Laplace sur les coefficients du signal.

### 10.4 Filtre de Kuwahara

Le filtre de Kuwahara (Kuwahara et al., 1976) est un filtre de lissage adaptatif qui préserve les contours en sélectionnant le quadrant de voisinage le plus homogène. Pour chaque pixel $(x_0, y_0)$, le voisinage de rayon $r$ est divisé en quatre quadrants qui se chevauchent légèrement :

$$Q_k = \{(x_0 + i,\, y_0 + j) : (i, j) \in \text{quadrant } k\}, \quad k = 1, 2, 3, 4$$

On calcule la **variance** $\sigma_k^2$ et la **moyenne** $\mu_k$ de l'intensité (ici de l'image en niveaux de gris utilisée comme guide) dans chaque quadrant. L'output est la moyenne du quadrant de variance minimale :

$$g(x_0, y_0) = \mu_{k^*}, \qquad k^* = \arg\min_k \sigma_k^2$$

L'intégrale d'image (image intégrale, summed area table) est utilisée pour calculer $\mu_k$ et $\sigma_k^2$ en un nombre **constant** d'opérations par pixel, quel que soit le rayon $r$. En effet, la somme de tout rectangle dans un tableau peut être obtenue par exactement 4 additions/soustractions sur le tableau des sommes cumulées, indépendamment de la taille du rectangle. La complexité totale est donc proportionnelle au nombre de pixels $W \times H$, et n'augmente **pas** avec $r$ — une optimisation cruciale pour les grands rayons (sans cette astuce, le coût serait proportionnel à $W \times H \times r^2$).

**Propriété clé** : du côté du quadrant homogène d'un contour, la variance est faible et ce quadrant est sélectionné ; du côté hétérogène (qui traverse le contour), la variance est élevée et ce quadrant est rejeté. Le résultat est un lissage fort dans les régions homogènes et une préservation des contours.

---

## 11. Déconvolution et Restauration d'Image

### 11.1 Le Problème de Déconvolution

La déconvolution est le problème inverse du floutage. Connaissant l'image floue $f_{\text{deg}} = h * f + n$ et la PSF $h$, on cherche à estimer $f$. Dans le domaine de Fourier, la solution naïve (filtre inverse) serait :

$$\hat{F}(u, v) = \frac{F_{\text{deg}}(u, v)}{H_{\text{PSF}}(u, v)}$$

Ce filtre inverse est **instable** : aux fréquences où $H_{\text{PSF}}(u,v) \approx 0$ (zéros de la PSF), le terme $N(u,v)/H_{\text{PSF}}(u,v)$ diverge — le bruit est amplifié de façon catastrophique. La régularisation est donc indispensable.

### 11.2 Filtre de Wiener pour la Déconvolution

Le filtre de Wiener résout le problème de déconvolution en cherchant le filtre linéaire $W(u,v)$ qui minimise l'erreur quadratique moyenne entre l'estimée $\hat{F}$ et la véritable $F$ :

$$\hat{F}(u, v) = W(u, v) \cdot F_{\text{deg}}(u, v)$$

$$W(u, v) = \frac{H_{\text{PSF}}^*(u, v)}{|H_{\text{PSF}}(u, v)|^2 + \underbrace{S_n(u,v)/S_f(u,v)}_{\text{NSR}(u,v)}}$$

où $H_{\text{PSF}}^*$ est le conjugué complexe de $H_{\text{PSF}}$, $S_n(u,v) = \mathbb{E}[|N(u,v)|^2]$ est le spectre de puissance du bruit et $S_f(u,v) = \mathbb{E}[|F(u,v)|^2]$ est le spectre de puissance du signal. Le rapport $\text{NSR}(u,v) = S_n/S_f$ est le **rapport bruit-signal local**.

Dans l'app, on suppose NSR constant : $\text{NSR}(u,v) = K$ (le paramètre **Wiener balance** $K$). Le filtre de Wiener devient alors :

$$W(u, v) = \frac{H_{\text{PSF}}^*(u, v)}{|H_{\text{PSF}}(u, v)|^2 + K}$$

Pour $K \to 0$ : le filtre de Wiener converge vers le filtre inverse pur (instable, sensible au bruit). Pour $K$ grand : le filtre atténue toutes les fréquences et produit une image floutée mais non amplifiée. La valeur optimale de $K$ équilibre la suppression du flou et la non-amplification du bruit.

### 11.3 Filtre de Wiener Non Supervisé

La version non supervisée (Unsupervised Wiener, Orieux, Giovannelli & Rodet, 2010) estime conjointement l'image restaurée et les hyperparamètres spectraux (les spectres de puissance $S_f$ et $S_n$) à partir des données uniquement, sans connaissance a priori des niveaux de bruit. L'estimation est réalisée par un algorithme MCMC (Monte Carlo par Chaînes de Markov), alternant entre l'échantillonnage de l'image restaurée et la mise à jour des hyperparamètres. Cette méthode est plus robuste que le Wiener supervisé dans les cas où le niveau de bruit n'est pas connu.

### 11.4 Richardson-Lucy (Déconvolution par Maximum de Vraisemblance Poissonien)

L'algorithme de **Richardson-Lucy** (Richardson, 1972 ; Lucy, 1974) est un algorithme itératif de déconvolution issu du modèle de bruit **poissonien**, adapté au bruit photonique. Sous l'hypothèse que chaque pixel $f_{\text{deg}}(x,y)$ est un échantillon d'une loi de Poisson de paramètre $(h * f)(x,y)$, la log-vraisemblance à maximiser est :

$$\ell(f) = \sum_{x,y} \left[ f_{\text{deg}}(x,y) \log\!\left((h * f)(x,y)\right) - (h * f)(x,y) \right]$$

L'algorithme EM (Expectation-Maximization) associé donne la règle de mise à jour itérative de Richardson-Lucy :

$$f^{(t+1)}(x, y) = f^{(t)}(x, y) \cdot \left(h(-\cdot,-\cdot) * \frac{f_{\text{deg}}}{h * f^{(t)}}\right)(x, y)$$

où $h(-x,-y)$ est la PSF retournée (corrélation). En notant $\tilde{h}(x,y) = h(-x,-y)$, l'itération s'écrit :

$$f^{(t+1)} = f^{(t)} \cdot \left[\tilde{h} * \frac{f_{\text{deg}}}{h * f^{(t)}}\right]$$

Chaque itération est une multiplication ponctuelle par un terme correctif. L'algorithme converge vers le maximum de vraisemblance poissonien, qui est l'estimateur du maximum a posteriori (MAP) sous une prior uniforme sur $f$.

**Comportement en nombre d'itérations** : pour un petit nombre d'itérations, l'algorithme agit comme un filtre passe-bas partiel. En augmentant le nombre d'itérations, la résolution s'affine mais le bruit est progressivement amplifié (sur-ajustement aux fluctuations du bruit). Il existe un **nombre d'itérations optimal** qui minimise l'erreur quadratique, au-delà duquel l'image se dégrade.