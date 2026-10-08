<script setup lang="ts">
import LegalPageShell from '~/components/LegalPageShell.vue'
</script>

<template>
  <LegalPageShell
    title="Politique de confidentialité"
    description="Traitement des données personnelles, conservation des enregistrements audio et exercice des droits des utilisateurs de Sawt-AI."
    path="/privacy-policy"
  >
    <section aria-labelledby="scope-title">
      <h2 id="scope-title">1. Objet et champ d’application</h2>
      <p>
        La présente politique de confidentialité précise les modalités de traitement des données
        liées à l’utilisation de Sawt-AI. Elle couvre la reconnaissance de passages du Coran,
        la recherche de hadiths et la consultation des traductions et commentaires.
        Ces fonctionnalités sont accessibles sans création de compte utilisateur.
      </p>
    </section>

    <section aria-labelledby="data-title">
      <h2 id="data-title">2. Données traitées et finalités</h2>
      <ul>
        <li>
          <strong>Enregistrements audio.</strong> Les fichiers enregistrés ou importés par l’utilisateur
          sont transmis au serveur de Sawt-AI pour transcription et analyse. Le traitement est effectué
          sur le serveur du site. L’activation du microphone intervient à la demande de l’utilisateur,
          sous réserve de l’autorisation accordée dans le navigateur.
        </li>
        <li>
          <strong>Recherches et résultats.</strong> Les phrases saisies, transcriptions et résultats
          sont traités pour fournir le service demandé. Ils ne font pas l’objet d’un enregistrement
          automatique dans la base de données, à l’exception des éléments transmis volontairement
          dans le cadre d’un signalement.
        </li>
        <li>
          <strong>Signalements volontaires.</strong> Les transcriptions, résultats, appréciations,
          corrections éventuelles et commentaires transmis au moyen du formulaire de signalement
          sont conservés pour analyser les erreurs et améliorer la qualité du service.
          La transmission de ces informations est facultative.
        </li>
        <li>
          <strong>Informations techniques.</strong> Le fonctionnement du site nécessite le traitement
          d’informations techniques : adresse IP, données de connexion, caractéristiques des fichiers
          audio, durées de traitement, références de versets, scores et erreurs éventuelles.
          Ces informations peuvent être consignées dans les journaux techniques à des fins de
          diagnostic et de sécurité.
        </li>
      </ul>
      <p>
        Les données sont traitées pour assurer les fonctionnalités demandées, analyser les incidents
        et maintenir la sécurité du site. Sawt-AI n’utilise pas les enregistrements audio ou les
        recherches pour établir un profil de croyances religieuses.
      </p>
      <p>
        Les enregistrements, transcriptions et commentaires sont susceptibles de contenir des données
        personnelles. Les contenus transmis doivent se limiter aux informations nécessaires à
        l’utilisation du service et respecter la confidentialité des données concernant des tiers.
      </p>
    </section>

    <section aria-labelledby="providers-title">
      <h2 id="providers-title">3. Prestataires</h2>
      <p>Les prestataires suivants interviennent dans le fonctionnement du site :</p>
      <ul>
        <li>
          <strong>Supabase :</strong> stockage des signalements volontaires et des contenus de tafsir.
          <a href="https://supabase.com/privacy" target="_blank" rel="noreferrer">Politique de confidentialité</a>.
        </li>
        <li>
          <strong>Umami :</strong> mesure de l’audience du site au moyen de l’instance umami.nabster.dev.
          <a href="https://docs.umami.is/docs/faq" target="_blank" rel="noreferrer">Informations sur Umami</a>.
        </li>
        <li>
          <strong>Google Fonts :</strong> téléchargement des polices par le navigateur. Google
          reçoit les informations de connexion nécessaires à leur livraison, notamment l’adresse IP.
          <a href="https://policies.google.com/privacy?hl=fr" target="_blank" rel="noreferrer">Politique de confidentialité</a>.
        </li>
      </ul>
      <p>
        Les prestataires externes peuvent traiter des informations dans d’autres pays, selon
        leurs infrastructures. Les modalités de ces traitements sont précisées dans leurs
        politiques de confidentialité respectives.
      </p>
    </section>

    <section aria-labelledby="analytics-title">
      <h2 id="analytics-title">4. Mesure d’audience et stockage dans le navigateur</h2>
      <p>
        Umami collecte des statistiques relatives aux pages consultées, à la provenance des visites
        et aux caractéristiques générales du navigateur et de l’appareil. Cet outil fonctionne
        sans cookie de suivi. Sawt-AI ne lui transmet pas le contenu des enregistrements audio,
        des recherches ou des formulaires de signalement.
      </p>
      <p>
        Le site n’utilise pas de cookies publicitaires et ne sauvegarde pas d’historique de recherche
        dans le navigateur. L’autorisation d’accès au microphone peut être révoquée à tout moment
        dans les paramètres du navigateur.
      </p>
    </section>

    <section aria-labelledby="retention-title">
      <h2 id="retention-title">5. Conservation</h2>
      <ul>
        <li>
          <strong>Enregistrements audio :</strong> le fichier temporaire reçu par le serveur est
          supprimé à l’issue du traitement de la requête, y compris en cas d’échec. L’application
          ne prévoit pas l’archivage de ces fichiers à des fins d’entraînement des modèles.
        </li>
        <li>
          <strong>Recherches et résultats :</strong> ces données sont conservées dans l’interface
          pendant leur utilisation, puis effacées lors de la réinitialisation ou de la fermeture
          du parcours concerné.
        </li>
        <li>
          <strong>Signalements volontaires :</strong> les données associées sont conservées dans
          Supabase pour l’analyse des erreurs signalées. L’application ne prévoit pas leur suppression
          automatique. Leur effacement peut être demandé par la personne concernée.
        </li>
        <li>
          <strong>Journaux et statistiques :</strong> les modalités de conservation dépendent de
          la configuration de l’hébergement et de l’outil de mesure d’audience. L’application
          ne déclenche pas de suppression automatique des statistiques.
        </li>
      </ul>
    </section>

    <section aria-labelledby="rights-title">
      <h2 id="rights-title">6. Droits des utilisateurs</h2>
      <p>
        L’utilisation du microphone et l’envoi de signalements sont facultatifs. La recherche de
        hadiths est également accessible par saisie textuelle. Dans les conditions prévues par le
        RGPD, les personnes concernées disposent de droits d’accès, de rectification, d’effacement,
        de limitation, de portabilité et d’opposition à certains traitements.
      </p>
      <p>
        Les demandes relatives aux données personnelles sont à adresser à l’éditeur par le moyen
        de contact indiqué sur le site. Une demande concernant un signalement doit préciser sa date
        approximative et les éléments nécessaires à son identification. Les données personnelles
        ne doivent pas être communiquées dans un espace public.
      </p>
      <p>
        Les personnes concernées peuvent également adresser une réclamation à l’autorité de
        protection des données compétente, notamment la
        <a href="https://www.cnil.fr/fr/mes-demarches/les-droits-pour-maitriser-vos-donnees-personnelles" target="_blank" rel="noreferrer">CNIL</a>
        en France.
      </p>
    </section>

    <section aria-labelledby="privacy-updates-title">
      <h2 id="privacy-updates-title">7. Mise à jour de la politique</h2>
      <p>
        La présente politique peut être mise à jour en fonction de l’évolution du site et des
        services utilisés. La date figurant en tête de page indique sa dernière révision.
        Les modalités d’utilisation du service sont précisées dans les
        <a href="/terms-of-service">conditions d’utilisation</a>.
      </p>
    </section>
  </LegalPageShell>
</template>
