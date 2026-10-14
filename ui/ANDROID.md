# Sawt AI sur Android

Le site conserve son rendu Nuxt côté serveur. Android embarque une compilation
côté client du même code Vue et utilise la même API FastAPI.

## Génération de l'interface

Depuis `ui`, avec Node 22 ou supérieur :

```bash
npm ci
npm run generate:android
```

Les fichiers mobiles sont générés dans `.output-android/public`, indépendamment
des sorties web `.nuxt` et `.output`. L'API par défaut est
`https://api.sawt-ai.nabster.dev`, vérifiée depuis la configuration du site public.
Pour la remplacer, définir `NUXT_PUBLIC_API_BASE_URL` dans l'environnement de la
commande. Cette adresse est intégrée à l'APK à la compilation ; elle doit être
en HTTPS et ne doit contenir aucun secret.

Dans l'environnement de l'API de production, ajouter `https://localhost` à
`ALLOWED_ORIGINS`, en conservant les origines web existantes :

```dotenv
ALLOWED_ORIGINS=https://sawt-ai.nabster.dev,https://localhost
```

Cette origine correspond au serveur local sécurisé de la WebView Capacitor.
Les CORS ne constituent pas un mécanisme d'authentification. Aucun changement
du modèle de reconnaissance ni des routes FastAPI n'est nécessaire.

Documentation de référence :
[Capacitor](https://capacitorjs.com/docs/getting-started),
[prérequis Android](https://capacitorjs.com/docs/getting-started/environment-setup).

## Projet natif

Capacitor 8.5.3 est installé avec ses plugins officiels App, Browser et
SplashScreen. L'identifiant stable est `dev.nabster.sawtai`. Installer Android
Studio 2025.2.1 ou supérieur et le SDK Android 36, avec Java 21. Le Docker UI
utilise désormais Node 22 ; il ne contient pas le SDK Android.

```bash
npm run android:sync
npm run android:open
```

`android:sync` recompile l'interface avant de synchroniser les fichiers et
plugins natifs. Android Studio permet ensuite de lancer un APK de développement.
La WebView sert les fichiers embarqués depuis `https://localhost` ; aucune
URL distante ni exception HTTP n'est activée dans la configuration Capacitor.
Les fichiers web copiés, caches Gradle, SDK local et clés privées sont ignorés
par Git. Après un checkout, effectuer `npm ci` puis `npm run android:sync`.

## Dépendances et sécurité

L'audit du 14 octobre 2026 a conduit à actualiser les dépendances compatibles,
notamment Nuxt 4.5.2, Vue 3.5.44 et Vitest 4.1.11. Les cinq vulnérabilités
critiques initiales ne sont plus présentes. Des alertes subsistent dans des
dépendances de l'outillage (`braces`, `node-forge`, `xcode/uuid`) sans correctif
compatible disponible pour certaines d'entre elles. Ne pas utiliser
`npm audit fix --force` pour rétrograder Capacitor ou changer les versions
majeures. Le lockfile est versionné ; utiliser `npm ci` pour le reproduire.

## Microphone et interruptions

Capacitor transmet la demande `getUserMedia` à Android. Le manifeste déclare
`RECORD_AUDIO` et `MODIFY_AUDIO_SETTINGS` ; la permission est demandée à la
première interaction avec le micro, pas au lancement. Aucun plugin audio tiers
ni service d'enregistrement en arrière-plan n'est ajouté.

Un refus affiche une instruction pour les paramètres et n'entraîne pas une
deuxième demande immédiate. Le passage en arrière-plan ou la fermeture de
l'écran libère le micro et invalide une autorisation encore en attente. Un
enregistrement interrompu n'est pas envoyé automatiquement à l'API.

Vérifications sur appareil avant promotion de la bêta : accepter/refuser la
permission, réessayer après activation dans les paramètres, enregistrer puis
reconnaître une récitation, verrouiller le téléphone pendant la capture,
changer d'application et tester la recherche vocale de hadiths.

## Navigation

Le bouton Retour ferme d'abord les détails ouverts. Dans le parcours Coran,
il annule une analyse ou revient à l'enregistrement avant de retourner à
l'accueil. Pendant une capture Coran, il est bloqué comme la navigation web.
Dans les hadiths, il annule d'abord une capture ou recherche active. Depuis
l'accueil, il quitte l'application.

Les pages légales sont ouvertes localement sans serveur Nuxt. Les sources
externes et le contact utilisent le plugin Browser (Android Custom Tabs) :
le formulaire et son captcha restent sur leur origine web publique.
Les liens `mailto:` et `tel:` restent traités par Android. Aucun remplacement
des liens du site ni des routes SSR publiques n'est nécessaire.

## Affichage

Le plugin SystemBars intégré à Capacitor 8 active le plein écran immersif.
Le système peut réafficher temporairement les barres lors d'un geste. La gestion
native des insets reste active avec `viewport-fit=cover` et le CSS existant
`env(safe-area-inset-*)`. Le clavier n'est pas bloqué par une hauteur fixe.
L'écran de lancement reste visible jusqu'au montage de Vue puis disparaît.
L'icône micro cyan sur fond Sawt sombre est vectorielle, avec variantes
adaptatives et monochromes Android 13+. Aucune permission caméra n'est ajoutée.

Référence : [SystemBars](https://capacitorjs.com/docs/apis/system-bars).

## APK signé et mises à jour

Avec `JAVA_HOME` pointant vers Java 21 et `ANDROID_HOME` vers le SDK Android 36 :

```bash
npm run android:keygen   # Une seule fois, refuse de remplacer une clé existante.
npm run android:publish
```

La clé RSA 4096 et ses mots de passe aléatoires sont dans `.android-release`,
avec permissions privées. Sauvegarder ce dossier de manière sécurisée, hors
du dépôt et des téléchargements publics. Une restauration sur un autre poste
nécessite d'ajuster `SAWT_ANDROID_KEYSTORE` au chemin restauré. Les variables
`SAWT_ANDROID_KEYSTORE`, `SAWT_ANDROID_STORE_PASSWORD`, `SAWT_ANDROID_KEY_ALIAS`
et `SAWT_ANDROID_KEY_PASSWORD` peuvent remplacer la configuration locale, par
exemple dans un gestionnaire de secrets CI. Aucun mot de passe n'est passé dans
les arguments de commande ou écrit dans les logs. Les fichiers privés et le
projet Android sont exclus du contexte Docker du site.

Le script recompile Nuxt, synchronise Capacitor, compile la release et vérifie
sa signature avec `apksigner`. L'APK et son SHA-256 sont dans `artifacts/android`.
La commande `android:publish` prépare ensuite `public/downloads` : un APK
versionné, un alias `sawt-ai-android.apk`, le checksum et `android-release.json`.
Ces artefacts ne sont pas versionnés et sont exclus des APK suivants.

Avant chaque mise à jour distribuée, augmenter `versionCode` et modifier
`versionName` dans `android/version.properties`. Garder l'identifiant et la
clé pour que les installations existantes puissent être mises à jour.
`npm run android:debug` génère une version de développement, et
`npm run android:bundle` prépare un AAB signé pour le Play Store ultérieur.

Pour héberger l'APK avec Nuxt, préparer les téléchargements **avant** `npm run
build` (ou le build Docker) : Nitro doit inclure les fichiers dans son inventaire
d'assets. Conserver également une copie hors checkout, par exemple dans
`/srv/apps/sawt-ai/data/android`, pour les restaurer avant les builds suivants.
Le Dockerfile de production `Dockerfile.vps` est maintenant versionné et utilise
Node 22. Le wrapper Gradle vérifie le SHA-256 officiel de sa distribution.

Installation bêta : télécharger l'APK, l'ouvrir puis autoriser les installations
depuis le navigateur si Android le demande. Les mises à jour de l'interface
nécessitent un nouvel APK ; les modifications de l'API compatibles profitent
aux versions déjà installées.

## Bouton web

Après publication de l'APK, définir dans l'environnement du serveur Nuxt :

```dotenv
NUXT_PUBLIC_ANDROID_DOWNLOAD_URL=/downloads/sawt-ai-android.apk
```

Le bouton apparaît en haut de l'accueil avec les instructions d'installation.
Il reste masqué sans URL valide et à l'intérieur de l'application Android.
Une URL absolue HTTPS terminant par `.apk` permet aussi un hébergement externe.
En local Docker Compose transmet cette variable depuis le `.env` racine.

Les tests de bout en bout vérifient le lien et les instructions à 320 px. Ils
ne remplacent pas une installation réelle et un test micro sur téléphone.

## Caches Nuxt et TypeScript

Les tests navigateur utilisent `.nuxt-e2e`, Android `.nuxt-android` et le site
peut utiliser un cache dédié avec `SAWT_NUXT_BUILD_DIR=.nuxt-web-release`.
Vite reçoit le tsconfig correspondant au build. Le transformateur Vue de
production lit aussi les références du `tsconfig.json` racine : le script
`prepare-types.mjs` les génère via les API officielles Nuxt Kit, sans vider
le cache de compilation web. Il est appelé avant la génération Android et les
tests de bout en bout. Un ancien cache Docker avec des permissions différentes
a été conservé dans `.cache/nuxt-before-android` sur le poste de travail.
