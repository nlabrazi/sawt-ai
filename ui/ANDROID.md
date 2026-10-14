# Application Android

Capacitor embarque une compilation de l'interface Nuxt dans une application
Android. Le web et Android partagent le même code Vue et la même API FastAPI.
La reconnaissance audio et la recherche nécessitent une connexion Internet.

## Prérequis

- Node.js 22.12 ou supérieur.
- Java 21, avec `JAVA_HOME` configuré.
- Android Studio et le SDK Android 36, avec les Build Tools installés.
- `ANDROID_HOME` pointant vers le SDK Android.

L'application prend en charge Android 7.0 et les versions suivantes.
Toutes les commandes ci-dessous s'exécutent depuis le dossier `ui`.

## Développement

```bash
npm ci
npm run android:sync
npm run android:open
```

`android:sync` compile l'interface dans `.output-android/public` et synchronise
le projet natif. `android:open` ouvre Android Studio pour lancer l'application
sur un téléphone ou un émulateur. Après une modification de l'interface,
relancer `android:sync`.

Pour générer un APK de développement en ligne de commande :

```bash
npm run android:debug
```

L'APK est disponible dans `android/app/build/outputs/apk/debug/app-debug.apk`.

## Connexion à l'API

L'adresse de l'API est intégrée à l'APK pendant la compilation. Pour remplacer
l'adresse définie dans `config/mobile.mjs` :

```bash
NUXT_PUBLIC_API_BASE_URL=https://api.example.org npm run android:sync
```

Utiliser une API accessible en HTTPS. Dans son paramètre `ALLOWED_ORIGINS`,
ajouter `https://localhost` en conservant les origines web existantes : c'est
l'origine de la WebView Capacitor.

## APK signé

Créer la clé de signature une seule fois, puis générer l'APK distribuable :

```bash
npm run android:keygen
npm run android:publish
```

La clé et sa configuration sont conservées dans `.android-release`, exclu
de Git. Sauvegarder ce dossier de manière sécurisée : la même clé est requise
pour mettre à jour une application déjà installée. Après restauration sur
un autre poste, ajuster `SAWT_ANDROID_KEYSTORE` au chemin de la clé.

`android:publish` compile l'application, vérifie sa signature et prépare l'APK,
son SHA-256 et les fichiers de téléchargement dans `public/downloads`.
L'APK versionné est également conservé dans `artifacts/android`.

## Distribution et mises à jour

Avant chaque nouvelle version, augmenter `versionCode` et modifier `versionName`
dans `android/version.properties`. Conserver l'identifiant de l'application
et sa clé de signature. Une modification de l'interface nécessite un nouvel APK.

L'APK peut être distribué par un lien HTTPS. Pour l'héberger avec Nuxt, exécuter
`android:publish` avant le build web, puis configurer le lien de téléchargement :

```dotenv
NUXT_PUBLIC_ANDROID_DOWNLOAD_URL=/downloads/sawt-ai-android.apk
```

Une URL absolue HTTPS vers un APK est également acceptée. Le lien est masqué
dans l'application Android. Les APK et les clés privées ne sont pas versionnés.

Pour préparer un Android App Bundle destiné au Google Play Store :

```bash
npm run android:bundle
```

Avant distribution, vérifier sur téléphone le micro (autorisation et refus),
l'arrêt de l'enregistrement en arrière-plan, le bouton Retour et le plein écran.

Référence : [documentation Capacitor Android](https://capacitorjs.com/docs/android).
