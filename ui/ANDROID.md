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
