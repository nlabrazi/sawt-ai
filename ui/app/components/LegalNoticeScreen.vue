<script setup lang="ts">
import LegalPageShell from '~/components/LegalPageShell.vue'
</script>

<template>
  <LegalPageShell
    title="Mentions légales"
    description="Identification de l’éditeur de Sawt-AI, hébergement, droits sur les contenus et protection des données personnelles."
    path="/legal-notice"
  >
    <section aria-labelledby="legal-purpose-title">
      <h2 id="legal-purpose-title">1. Présentation du site</h2>
      <p>
        Sawt-AI est un service gratuit consacré à la reconnaissance de passages du Coran,
        à la recherche de hadiths et à la consultation des traductions et commentaires disponibles.
        Les informations relatives à l’éditeur et à l’hébergement figurent en tête de page.
      </p>
      <p>
        Les coordonnées et informations légales d’OVHcloud sont consultables sur le
        <a href="https://www.ovhcloud.com/fr/terms-and-conditions/" target="_blank" rel="noreferrer">
          site officiel de l’hébergeur
        </a>.
      </p>
    </section>

    <section aria-labelledby="legal-rights-title">
      <h2 id="legal-rights-title">2. Droits sur les contenus</h2>
      <p>
        Les textes, traductions, commentaires et autres contenus de tiers demeurent soumis aux
        droits et conditions de réutilisation de leurs sources respectives. Leur consultation sur
        Sawt-AI ne confère pas d’autorisation générale de redistribution ou d’exploitation commerciale.
      </p>
      <p>
        Le code du projet est soumis à sa licence propre. La mention « Tous droits réservés »
        ne modifie pas les droits accordés par cette licence ni ceux attachés aux contenus de tiers.
      </p>
    </section>

    <section aria-labelledby="legal-data-title">
      <h2 id="legal-data-title">3. Utilisation du service</h2>
      <p>
        L’accès au service est encadré par les
        <a href="/terms-of-service">conditions générales d’utilisation</a>.
      </p>
      <p>
        Les demandes relatives au site peuvent être adressées à l’éditeur depuis la
        <a href="/contact">page Contact</a>.
      </p>
    </section>
    <section id="privacy" aria-labelledby="privacy-title">
      <h2 id="privacy-title">4. Confidentialité et protection des données</h2>
      <article aria-labelledby="scope-title">
        <h3 id="scope-title">4.1. Objet et champ d’application</h3>
        <p>
          La présente politique de confidentialité précise les modalités de traitement des données
          liées à l’utilisation de Sawt-AI. Elle couvre la reconnaissance de passages du Coran,
          la recherche de hadiths et la consultation des traductions et commentaires.
          Ces fonctionnalités sont accessibles sans création de compte utilisateur. La présente
          politique couvre également les demandes transmises au moyen de la page Contact.
        </p>
      </article>

      <article aria-labelledby="data-title">
        <h3 id="data-title">4.2. Données traitées et finalités</h3>
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
            <strong>Messages de contact.</strong> Le nom, l’adresse e-mail, l’objet et le message
            renseignés dans le formulaire sont transmis à Web3Forms lors de l’envoi, puis à la
            messagerie de l’éditeur. Ces informations servent à traiter la demande et à y répondre.
            L’utilisation du formulaire est facultative ; un contact direct par e-mail est également proposé.
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
      </article>

      <article aria-labelledby="providers-title">
        <h3 id="providers-title">4.3. Prestataires</h3>
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
            <strong>Web3Forms :</strong> traitement et transmission des messages envoyés depuis la page Contact.
            <a href="https://web3forms.com/privacy" target="_blank" rel="noreferrer">Politique de confidentialité</a>.
          </li>
          <li>
            <strong>hCaptcha :</strong> vérification antispam du formulaire de contact. Le service est chargé
            lors de l’interaction avec le formulaire et traite les informations techniques nécessaires
            à cette vérification, notamment l’adresse IP et les interactions avec le module.
            <a href="https://www.hcaptcha.com/privacy" target="_blank" rel="noreferrer">Politique de confidentialité</a>.
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
      </article>

      <article aria-labelledby="analytics-title">
        <h3 id="analytics-title">4.4. Mesure d’audience et stockage dans le navigateur</h3>
        <p>
          Umami collecte des statistiques relatives aux pages consultées, à la provenance des visites
          et aux caractéristiques générales du navigateur et de l’appareil. Cet outil fonctionne
          sans cookie de suivi. Sawt-AI ne lui transmet pas le contenu des enregistrements audio,
          des recherches, des formulaires de signalement ou des messages de contact.
        </p>
        <p>
          Le site n’utilise pas de cookies publicitaires et ne sauvegarde pas d’historique de recherche
          dans le navigateur. La vérification antispam du formulaire de contact repose sur hCaptcha,
          dont les modalités de stockage sont précisées dans sa politique de confidentialité.
          L’autorisation d’accès au microphone peut être révoquée à tout moment
          dans les paramètres du navigateur.
        </p>
      </article>

      <article aria-labelledby="retention-title">
        <h3 id="retention-title">4.5. Conservation</h3>
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
            <strong>Messages de contact :</strong> les messages reçus sont conservés dans la messagerie
            de l’éditeur pour le suivi des demandes. Les données traitées par Web3Forms sont soumises
            aux modalités de conservation décrites dans sa politique de confidentialité et à la
            configuration du service. Leur effacement peut être demandé à l’éditeur.
          </li>
          <li>
            <strong>Journaux et statistiques :</strong> les modalités de conservation dépendent de
            la configuration de l’hébergement et de l’outil de mesure d’audience. L’application
            ne déclenche pas de suppression automatique des statistiques.
          </li>
        </ul>
      </article>

      <article aria-labelledby="rights-title">
        <h3 id="rights-title">4.6. Droits des utilisateurs</h3>
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
      </article>

      <article aria-labelledby="privacy-updates-title">
        <h3 id="privacy-updates-title">4.7. Mise à jour de la politique</h3>
        <p>
          La présente politique peut être mise à jour en fonction de l’évolution du site et des
          services utilisés. La date figurant en tête de page indique sa dernière révision.
          Les modalités d’utilisation du service sont précisées dans les
          <a href="/terms-of-service">conditions d’utilisation</a>.
        </p>
      </article>
    </section>
  </LegalPageShell>
</template>
