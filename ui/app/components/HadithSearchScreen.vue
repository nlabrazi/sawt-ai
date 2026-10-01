<script setup lang="ts">
import { Search } from '@lucide/vue'
import { defineAsyncComponent, nextTick, onBeforeUnmount, ref, watch } from 'vue'
import HadithResultCard from '~/components/HadithResultCard.vue'
import type { useHadithSearch } from '~/composables/useHadithSearch'
import type { HadithResult } from '~/types/hadith'

const HadithDetailsDialog = defineAsyncComponent(
  () => import('~/components/HadithDetailsDialog.vue'),
)
const props = defineProps<{ searchState: ReturnType<typeof useHadithSearch> }>()
const { query, response, loading, error, validationError, search, cancel } = props.searchState
const selected = ref<HadithResult | null>(null)
const resultsTitle = ref<HTMLElement | null>(null)

watch(response, async (value) => {
  if (!value) return
  await nextTick()
  resultsTitle.value?.focus({ preventScroll: true })
})

onBeforeUnmount(() => {
  selected.value = null
  cancel()
})
</script>

<template>
  <section class="hadith-screen" :class="{ 'is-idle': !loading && !response && !error }" aria-labelledby="hadith-title">
    <div class="search-shell">
      <div class="search-intro">
        <h1 id="hadith-title">Retrouvez un hadith</h1>
        <p>Un sujet, quelques mots ou un extrait.</p>
      </div>
      <form class="search-form" novalidate @submit.prevent="search">
        <label for="hadith-query" class="sr-only">Que recherchez-vous ?</label>
        <div class="search-controls">
          <input id="hadith-query" v-model="query" type="search" maxlength="300" placeholder="Rechercher un hadith" enterkeyhint="search" :aria-invalid="!!validationError" :aria-describedby="validationError ? 'hadith-query-error' : undefined" @input="validationError = null" />
          <button class="search-button" type="submit" :disabled="loading">
            <Search :size="20" aria-hidden="true" /><span class="sr-only">{{ loading ? 'Recherche…' : 'Rechercher' }}</span>
          </button>
        </div>
        <p v-if="validationError" id="hadith-query-error" class="validation-error" role="alert">{{ validationError }}</p>
      </form>
    </div>
    <div class="search-feedback sr-only" role="status" aria-live="polite" aria-atomic="true">
      <span v-if="loading">Recherche des hadiths en cours…</span>
      <span v-else-if="response">{{ response.results.length }} proposition{{ response.results.length === 1 ? '' : 's' }} disponible{{ response.results.length === 1 ? '' : 's' }}.</span>
    </div>
    <div v-if="loading" class="loading-panel" :aria-busy="true">
      <div class="loading-copy"><span class="loading-dot" aria-hidden="true" /><p>Recherche en cours…</p><button type="button" @click="cancel">Annuler</button></div>
      <div v-for="position in 3" :key="position" class="skeleton-card" aria-hidden="true"><span /><span /><span /></div>
    </div>
    <div v-else-if="error" class="status-panel" role="alert">
      <h2>La recherche n’a pas abouti</h2><p>{{ error }}</p>
      <button type="button" @click="search">Réessayer</button>
    </div>
    <section v-else-if="response" class="results" aria-labelledby="hadith-results-title">
      <div class="results-heading"><h2 id="hadith-results-title" ref="resultsTitle" tabindex="-1">Hadiths proposés</h2><p>Pour « {{ response.query }} »</p></div>
      <div v-if="!response.results.length" class="empty-state">
        <p v-if="response.search_mode === 'keywords'">Aucun résultat pour ces mots-clés dans la collection française indexée. Essayez un autre mot ou décrivez le hadith dans une phrase.</p>
        <p v-else>Aucun résultat exploitable n’a été retourné. Essayez une autre formulation.</p>
        <a href="https://hadeethenc.com/fr" target="_blank" rel="noopener noreferrer">Consulter la collection HadeethEnc <span class="sr-only">(nouvel onglet)</span></a>
      </div>
      <HadithResultCard v-for="(hadith, index) in response.results" :key="hadith.id" :hadith="hadith" :position="index + 1" @read="selected = $event" />
      <details class="result-context">
        <summary>À propos des résultats <span class="beta-badge">Bêta</span></summary>
        <p class="search-method" v-if="response.search_mode === 'keywords'">Recherche par mots-clés<span v-if="response.search_terms.length"> : {{ response.search_terms.map(term => `« ${term} »`).join(', ') }}</span>.<br />Chaque résultat contient ces mots, au singulier ou au pluriel, dans son titre, son texte ou son explication en français.</p>
        <p class="search-method" v-else>Recherche par sens. Les propositions peuvent être proches du sujet sans répondre exactement à votre demande.</p>
        <p class="source-note">Collection HadeethEnc · Vérifiez le texte et sa source.</p>
      </details>
    </section>
    <HadithDetailsDialog v-if="selected" :hadith="selected" @close="selected = null" />
  </section>
</template>

<style scoped>
.hadith-screen {
  flex: 1;
  display: flex;
  flex-direction: column;
  width: min(100%, 780px);
  margin: 0 auto;
  padding: 48px 20px 32px;
}

.hadith-screen.is-idle {
  justify-content: center;
}

.search-shell {
  width: min(100%, 620px);
  margin-inline: auto;
}

.search-intro p {
  margin: 16px 0 0;
  color: #aebed3;
  font-size: 15px;
  line-height: 1.6;
}

.search-intro {
  margin: 0 0 28px;
  text-align: center;
}

h1 {
  margin: 0;
  font-size: clamp(28px, 4vw, 40px);
  line-height: 1.1;
  letter-spacing: -.045em;
}

.search-form {
  width: 100%;
}

.search-controls {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px;
  border: 1px solid #3c506f;
  border-radius: 18px;
  background: #0e1b2d;
}

.search-controls:focus-within {
  border-color: #93c5fd;
}

input {
  flex: 1;
  min-width: 0;
  min-height: 52px;
  padding: 14px;
  border: 0;
  border-radius: 12px;
  background: transparent;
  color: #f1f5fb;
  font: inherit;
  font-size: 16px;
}

input::placeholder {
  color: #92a4be;
}

.search-controls:has(input[aria-invalid='true']) {
  border-color: #f4a2a2;
}

button {
  min-height: 44px;
  border: 1px solid #3c506f;
  border-radius: 12px;
  background: #1a2e49;
  color: #dce9fc;
  padding: 10px 14px;
  font: inherit;
  font-size: 14px;
  cursor: pointer;
}

button:disabled {
  opacity: .6;
  cursor: default;
}

.search-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  width: 48px;
  height: 48px;
  padding: 0;
  background: #2563eb;
  border-color: #4982f8;
  color: #fff;
  font-weight: 600;
}

.search-button:hover:not(:disabled) {
  background: #3472f2;
}

input:focus-visible, button:focus-visible {
  outline: 2px solid #93c5fd;
  outline-offset: 3px;
}

.validation-error {
  color: #fca5a5;
  font-size: 14px;
}

.results, .loading-panel {
  display: grid;
  gap: 16px;
  margin-top: 28px;
}

.results-heading h2 {
  margin: 0;
  font-size: 22px;
}

.results-heading p {
  color: #b6c4d8;
  font-size: 14px;
  overflow-wrap: anywhere;
}

.result-context {
  color: #a7b6cb;
  font-size: 12px;
}

.result-context summary {
  display: flex;
  align-items: center;
  gap: 8px;
  width: fit-content;
  min-height: 44px;
  cursor: pointer;
  list-style: none;
}

.result-context summary::before {
  content: '+';
  font-size: 16px;
}

.result-context[open] summary::before {
  content: '−';
}

.result-context summary:focus-visible {
  outline: 2px solid #93c5fd;
  outline-offset: 3px;
}

.beta-badge {
  padding: 2px 6px;
  border: 1px solid #3c506f;
  border-radius: 6px;
  font-size: 10px;
}

.source-note {
  margin: 10px 0 0;
  line-height: 1.7;
}

.search-method {
  margin: 0;
  color: #b6c4d8;
  font-size: 13px;
  line-height: 1.7;
  overflow-wrap: anywhere;
}

.empty-state p {
  margin: 0 0 12px;
}

.empty-state a {
  display: inline-block;
  padding: 8px 0;
  color: #93c5fd;
  text-underline-offset: 3px;
}

.empty-state a:focus-visible {
  outline: 2px solid #93c5fd;
  outline-offset: 3px;
}

.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip-path: inset(50%);
  white-space: nowrap;
  border: 0;
}

.status-panel, .empty-state {
  padding: 22px;
  border: 1px solid #394b65;
  border-radius: 18px;
  background: #132238;
  color: #bdcde2;
  line-height: 1.7;
}

.status-panel h2 {
  margin: 0;
  font-size: 20px;
  color: #f0d4d4;
}

.loading-copy {
  display: flex;
  align-items: center;
  gap: 10px;
  color: #bed0e7;
  font-size: 14px;
}

.loading-copy p {
  flex: 1;
  line-height: 1.6;
}

.loading-dot {
  width: 8px;
  height: 8px;
  flex-shrink: 0;
  border-radius: 50%;
  background: #79aaff;
  animation: pulse 1.4s ease-in-out infinite;
}

.skeleton-card {
  padding: 24px;
  border: 1px solid #26384f;
  border-radius: 22px;
  background: #112036;
}

.skeleton-card span {
  display: block;
  height: 14px;
  margin-bottom: 16px;
  border-radius: 5px;
  background: #243952;
  animation: pulse 1.4s ease-in-out infinite;
}

.skeleton-card span:first-child {
  width: 65%;
  height: 20px;
}

.skeleton-card span:last-child {
  width: 45%;
  margin-bottom: 0;
}

@keyframes pulse {
  50% {
    opacity: .45;
  }
}

@media (max-width: 640px) {
  .hadith-screen {
    padding: 36px 16px 24px;
  }
  .loading-copy {
    flex-wrap: wrap;
  }
}

@media (prefers-reduced-motion: reduce) {
  .loading-dot, .skeleton-card span {
    animation: none;
  }
}
</style>
