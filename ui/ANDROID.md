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
