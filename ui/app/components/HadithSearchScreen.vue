<script setup lang="ts">
import { BookOpen, Search } from '@lucide/vue'
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
const input = ref<HTMLInputElement | null>(null)
const resultsTitle = ref<HTMLElement | null>(null)
const examples = ['Ne pas se mettre en colère', 'Les actes et les intentions', 'La miséricorde']

watch(response, async (value) => {
  if (!value) return
  await nextTick()
  resultsTitle.value?.focus({ preventScroll: true })
})

onBeforeUnmount(() => {
  selected.value = null
  cancel()
})

function chooseExample(example: string) {
  query.value = example
  validationError.value = null
  input.value?.focus()
}
</script>

<template>
  <section class="hadith-screen" aria-labelledby="hadith-title">
    <div class="search-intro">
      <span class="eyebrow"><BookOpen :size="16" aria-hidden="true" /> Recherche de hadiths · Bêta</span>
      <h1 id="hadith-title">Retrouvez un hadith</h1>
      <p>Décrivez un sujet ou quelques mots du texte.<br />Sawt AI vous propose des hadiths à consulter.</p>
    </div>
    <form class="search-form" novalidate @submit.prevent="search">
      <label for="hadith-query">Que recherchez-vous ?</label>
      <div class="search-controls">
        <input id="hadith-query" ref="input" v-model="query" type="search" maxlength="300" placeholder="Un hadith sur la colère, les intentions…" :aria-invalid="!!validationError" :aria-describedby="validationError ? 'hadith-query-error' : 'hadith-query-hint'" @input="validationError = null" />
        <button class="search-button" type="submit" :disabled="loading">
          <Search :size="19" aria-hidden="true" /> {{ loading ? 'Recherche…' : 'Rechercher' }}
        </button>
      </div>
      <p v-if="validationError" id="hadith-query-error" class="validation-error" role="alert">{{ validationError }}</p>
      <p id="hadith-query-hint" class="input-hint">Une phrase en français · 300 caractères maximum</p>
      <div class="examples" aria-label="Exemples de recherche">
        <span>Essayez :</span>
        <button v-for="example in examples" :key="example" type="button" :disabled="loading" @click="chooseExample(example)">{{ example }}</button>
      </div>
    </form>
    <div class="search-feedback" role="status" aria-live="polite" aria-atomic="true">
      <span v-if="loading">Recherche des hadiths en cours…</span>
      <span v-else-if="response">{{ response.results.length }} proposition{{ response.results.length === 1 ? '' : 's' }} disponible{{ response.results.length === 1 ? '' : 's' }}.</span>
    </div>
    <div v-if="loading" class="loading-panel" :aria-busy="true">
      <div class="loading-copy"><span class="loading-dot" aria-hidden="true" /><p>Nous recherchons les passages proches de votre demande.</p><button type="button" @click="cancel">Annuler</button></div>
      <div v-for="position in 3" :key="position" class="skeleton-card" aria-hidden="true"><span /><span /><span /></div>
    </div>
    <div v-else-if="error" class="status-panel" role="alert">
      <h2>La recherche n’a pas abouti</h2><p>{{ error }}</p>
      <button type="button" @click="search">Réessayer</button>
    </div>
    <section v-else-if="response" class="results" aria-labelledby="hadith-results-title">
      <div class="results-heading"><h2 id="hadith-results-title" ref="resultsTitle" tabindex="-1">Hadiths proposés</h2><p>Pour « {{ response.query }} »</p></div>
      <p v-if="!response.results.length" class="empty-state">Aucun résultat exploitable n’a été retourné. Essayez une autre formulation ou consultez HadeethEnc.</p>
      <HadithResultCard v-for="(hadith, index) in response.results" :key="hadith.id" :hadith="hadith" :position="index + 1" @read="selected = $event" />
    </section>
    <p class="source-note">Recherche dans la collection HadeethEnc. Les propositions peuvent être proches du sujet sans répondre exactement à votre demande. Vérifiez les textes et leur source.</p>
    <HadithDetailsDialog v-if="selected" :hadith="selected" @close="selected = null" />
  </section>
</template>

<style scoped>
.hadith-screen {
  flex: 1;
  width: min(100%, 780px);
  margin: 0 auto;
  padding: 26px 20px 42px;
}

.search-intro {
  margin: 48px 0 32px;
  text-align: center;
}

.eyebrow {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  color: #93c5fd;
  font-size: 13px;
}

h1 {
  margin: 18px 0 16px;
  font-size: clamp(36px, 6vw, 54px);
  line-height: 1.1;
  letter-spacing: -.045em;
}

.search-intro p {
  margin: 0;
  color: #aebed3;
  line-height: 1.7;
}

.search-form {
  padding: 24px;
  background: #112035b8;
  border: 1px solid #2a3d58;
  border-radius: 22px;
}

label {
  display: block;
  margin-bottom: 12px;
  font-weight: 600;
  font-size: 14px;
  color: #d4e0f1;
}

.search-controls {
  display: flex;
  gap: 10px;
}

input {
  flex: 1;
  min-width: 0;
  min-height: 52px;
  padding: 14px;
  border: 1px solid #3c506f;
  border-radius: 12px;
  background: #091323;
  color: #f1f5fb;
  font: inherit;
  font-size: 16px;
}

input::placeholder {
  color: #92a4be;
}

input[aria-invalid='true'] {
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
  gap: 8px;
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

.input-hint {
  margin: 10px 0 18px;
  color: #a7b6cb;
  font-size: 12px;
}

.validation-error {
  color: #fca5a5;
  font-size: 14px;
}

.examples {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
  color: #a7b6cb;
  font-size: 12px;
}

.examples button {
  padding: 7px 10px;
  font-size: 12px;
  border-radius: 999px;
  background: #142740;
}

.examples button:hover:not(:disabled) {
  border-color: #719acc;
}

.search-feedback {
  min-height: 32px;
  padding-top: 14px;
  color: #b0c8e7;
  font-size: 13px;
}

.results, .loading-panel {
  display: grid;
  gap: 16px;
  margin-top: 18px;
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

.source-note {
  margin: 28px auto 0;
  max-width: 640px;
  color: #98abc5;
  font-size: 12px;
  line-height: 1.8;
  text-align: center;
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
  .search-intro {
    margin-top: 34px;
  }
  .search-form {
    padding: 18px;
  }
  .search-controls {
    flex-direction: column;
  }
  .search-button {
    min-height: 48px;
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
