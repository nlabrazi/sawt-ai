<script setup lang="ts">
import LegalPageShell from '~/components/LegalPageShell.vue'
</script>

<template>
  <LegalPageShell
    title="Conditions d’utilisation"
    description="Conditions d’accès à Sawt-AI, limites de la reconnaissance audio, sources religieuses et règles d’utilisation du service."
    path="/terms-of-service"
  >
    <section aria-labelledby="service-title">
      <h2 id="service-title">1. Objet et accès au service</h2>
      <p>
        Les présentes conditions définissent les modalités d’accès et d’utilisation des
        fonctionnalités publiques de Sawt-AI.
      </p>
      <p>
        Sawt-AI propose un service gratuit de reconnaissance de passages du Coran à partir
        d’enregistrements audio, de recherche de hadiths et de consultation des traductions
        françaises et des tafsirs disponibles. L’accès à ces fonctionnalités ne nécessite pas
        de création de compte utilisateur.
      </p>
    </section>

    <section aria-labelledby="accuracy-title">
      <h2 id="accuracy-title">2. Reconnaissance et exactitude des résultats</h2>
      <p>
        Les résultats de reconnaissance dépendent notamment de la qualité de l’enregistrement,
        du niveau de bruit et des caractéristiques de la récitation. Ils peuvent être inexacts,
        incertains ou absents. L’identification éventuelle d’un récitant constitue une estimation.
      </p>
      <p>
        Il appartient à l’utilisateur de vérifier les sourates, les versets et les références
        avant toute réutilisation ou diffusion d’un résultat. Les recherches de hadiths peuvent
        être incomplètes et ne se substituent pas à la consultation des recueils et de leurs explications.
      </p>
      <p>
        Les contenus proposés constituent une aide à la recherche et à la consultation.
        Le service ne délivre pas de fatwa et ne se substitue pas à l’avis d’une personne
        qualifiée en matière religieuse.
      </p>
    </section>

    <section aria-labelledby="sources-title">
      <h2 id="sources-title">3. Traductions, tafsirs et sources</h2>
      <p>
        La traduction française du Coran, attribuée à Rachid Maach, est importée depuis QuranEnc.
        Les textes de hadiths proviennent de HadeethEnc. Les références et attributions disponibles
        accompagnent les contenus publiés.
      </p>
      <p>
        Sawt-AI utilise l’API de contenu de Quran Foundation pour l’import des textes sources
        arabes des tafsirs d’Ibn Kathir et d’As-Sa‘di.
      </p>
      <p>
        Les tafsirs d’Ibn Kathir et d’As-Sa‘di font l’objet d’une présentation distincte. Certains
        textes sources peuvent être traduits en français à l’aide d’un outil de traduction.
        Leur publication est soumise à une relecture et à une validation manuelles, par comparaison
        avec les éditions françaises. Les textes en attente de validation ne sont pas accessibles au public.
      </p>
      <p>
        La validation manuelle réduit le risque d’erreur sans en garantir l’absence. Une traduction
        restitue une interprétation du sens et ne remplace pas le texte arabe. Toute erreur identifiée
        peut être signalée à l’éditeur par le moyen de contact indiqué sur le site.
      </p>
    </section>

    <section aria-labelledby="usage-title">
      <h2 id="usage-title">4. Utilisation des enregistrements et du service</h2>
      <p>L’utilisation du service implique le respect des règles suivantes :</p>
      <ul>
        <li>L’utilisateur doit disposer des droits et autorisations nécessaires à l’utilisation et à la transmission des fichiers.</li>
        <li>Les contenus transmis doivent respecter la vie privée des personnes enregistrées et la confidentialité de leurs informations.</li>
        <li>Les fichiers doivent respecter les limites de format, de taille et de durée indiquées dans l’interface.</li>
        <li>La transmission de contenus illicites ou portant atteinte aux droits de tiers est interdite.</li>
        <li>Le contournement des contrôles d’accès et les requêtes abusives susceptibles de perturber le service sont interdits.</li>
      </ul>
      <p>
        Le formulaire de signalement est destiné aux erreurs de reconnaissance. Son utilisation
        est facultative et les informations transmises doivent être limitées au problème signalé.
        La transmission d’un enregistrement ou de données personnelles par un mineur doit être
        effectuée avec l’accompagnement d’un responsable.
      </p>
    </section>

    <section aria-labelledby="ownership-title">
      <h2 id="ownership-title">5. Droits sur les contenus</h2>
      <p>
        Les traductions, commentaires et contenus importés demeurent soumis aux attributions
        et conditions de réutilisation de leurs sources respectives. Leur mise à disposition
        sur Sawt-AI n’emporte pas d’autorisation générale de redistribution ou d’exploitation
        commerciale. Il appartient à l’utilisateur de vérifier les conditions applicables avant
        toute réutilisation.
      </p>
      <p>
        La licence du code de Sawt-AI est consultable dans le dépôt du projet. Elle ne s’étend pas
        automatiquement aux textes religieux, traductions, enregistrements ou autres contenus de tiers.
      </p>
    </section>

    <section aria-labelledby="availability-title">
      <h2 id="availability-title">6. Disponibilité et responsabilité</h2>
      <p>
        L’accès au service peut être interrompu pour maintenance, en cas d’incident technique
        ou d’indisponibilité des ressources nécessaires à son fonctionnement. Il peut également
        être limité en cas d’utilisation abusive. La disponibilité permanente du service et
        l’obtention d’un résultat pour chaque enregistrement ne sont pas garanties.
      </p>
      <p>
        L’utilisation des résultats doit tenir compte des limites exposées dans les présentes
        conditions. Celles-ci s’appliquent sous réserve des droits et garanties reconnus aux
        utilisateurs par la législation applicable.
      </p>
    </section>

    <section aria-labelledby="terms-privacy-title">
      <h2 id="terms-privacy-title">7. Données personnelles</h2>
      <p>
        Les traitements des enregistrements audio, recherches, signalements volontaires et
        statistiques sont décrits
        dans la <a href="/privacy-policy">politique de confidentialité</a>.
        Celle-ci précise les finalités des traitements, les modalités de conservation et
        les droits des personnes concernées.
      </p>
    </section>

    <section aria-labelledby="terms-updates-title">
      <h2 id="terms-updates-title">8. Évolution du service et contact</h2>
      <p>
        Le service et les présentes conditions peuvent faire l’objet de mises à jour. La version
        applicable est publiée sur cette page avec sa date de révision. Toute demande, réclamation
        ou signalement relatif au service peut être adressé à l’éditeur par le moyen de contact
        indiqué en tête de page.
      </p>
    </section>
  </LegalPageShell>
</template>
