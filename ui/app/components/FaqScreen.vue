<script setup lang="ts">
import { ChevronDown } from '@lucide/vue'

const questions = [
  {
    question: 'Comment retrouver un passage du Coran ?',
    answer:
      'Dans Coran, touchez le micro pour enregistrer une récitation, puis touchez le bouton d’arrêt pour lancer l’analyse. Vous pouvez aussi importer un fichier audio.',
  },
  {
    question: 'Comment rechercher un hadith ?',
    answer:
      'Saisissez un sujet ou une phrase en français, ou touchez le micro à côté de la recherche pour dicter votre demande. Touchez le bouton d’arrêt pour rechercher ; l’enregistrement s’arrête aussi après 30 secondes. Vous pouvez corriger la phrase reconnue. Ouvrez un résultat pour lire le texte arabe, sa traduction et son explication.',
  },
  {
    question: 'D’où viennent les hadiths ?',
    answer:
      'Les textes proviennent de la collection HadeethEnc. Chaque résultat donne accès à sa source ; les textes ne sont pas générés par Sawt AI.',
  },
  {
    question: 'Les résultats sont-ils toujours exacts ?',
    answer:
      'Une correspondance peut rester incertaine. Vérifiez le passage et sa source. Pour le Coran, un extrait plus long ou un environnement plus calme peut aider ; pour les hadiths, essayez une formulation plus précise.',
  },
  {
    question: 'Pourquoi le micro ne fonctionne-t-il pas ?',
    answer:
      'Autorisez l’accès au microphone dans votre navigateur et ouvrez Sawt AI en HTTPS. Pour le Coran, vous pouvez aussi importer un audio ; pour les hadiths, saisissez votre recherche.',
  },
]
</script>

<template>
  <section class="faq-screen" aria-labelledby="faq-title">
    <h1 id="faq-title">Questions fréquentes</h1>
    <div class="questions">
      <details v-for="item in questions" :key="item.question" class="question">
        <summary>{{ item.question }}<ChevronDown :size="18" aria-hidden="true" /></summary>
        <p>{{ item.answer }}</p>
      </details>
    </div>
  </section>
</template>

<style scoped>
.faq-screen {
  flex: 1;
  width: min(100%, 720px);
  margin: 0 auto;
  padding: 48px 20px 32px;
}

h1 {
  margin: 0 0 32px;
  font-size: clamp(28px, 4vw, 40px);
  line-height: 1.2;
  letter-spacing: -.045em;
  text-align: center;
}

.questions { border-top: 1px solid #26384f; }
.question { border-bottom: 1px solid #26384f; }

summary {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  min-height: 64px;
  padding: 18px 0;
  color: #dce9fc;
  font-size: 15px;
  line-height: 1.5;
  list-style: none;
  cursor: pointer;
}

summary::-webkit-details-marker { display: none; }
summary svg { flex-shrink: 0; transition: transform 160ms ease; }
.question[open] summary svg { transform: rotate(180deg); }

summary:focus-visible {
  outline: 2px solid #93c5fd;
  outline-offset: 4px;
  border-radius: 4px;
}

p {
  margin: 0;
  padding: 0 28px 20px 0;
  color: #aebed3;
  font-size: 14px;
  line-height: 1.8;
}

@media (max-width: 640px) {
  .faq-screen { padding: 32px 16px 24px; }
}

@media (prefers-reduced-motion: reduce) {
  summary svg { transition: none; }
}
</style>
