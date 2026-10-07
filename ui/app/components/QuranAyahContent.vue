<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import type { QuranAyahContent } from '~/composables/useQuranContent'
import type { TafsirSource } from '~/composables/useTafsirReview'

const props = defineProps<{ content: QuranAyahContent }>()
const sources: { id: TafsirSource; label: string }[] = [
  { id: 'ibn_kathir', label: 'Ibn Kathir' },
  { id: 'as_saadi', label: 'As-Sa‘di' },
]
const selectedSource = ref<TafsirSource>('ibn_kathir')
const selectedTafsir = computed(() =>
  props.content.tafsirs.find((tafsir) => tafsir.source === selectedSource.value),
)
const translationUrl = computed(() => {
  try {
    const url = new URL(props.content.translation?.source_url ?? '')
    return ['https:', 'http:'].includes(url.protocol) ? url.href : null
  } catch {
    return null
  }
})

watch(
  () => props.content.tafsirs,
  (tafsirs) => {
    if (!tafsirs.some((tafsir) => tafsir.source === selectedSource.value)) {
      selectedSource.value =
        sources.find((source) => tafsirs.some((tafsir) => tafsir.source === source.id))?.id ??
        'ibn_kathir'
    }
  },
  { immediate: true },
)
</script>

<template>
  <article class="french-ayah" :data-ayah="content.ayah" lang="fr">
    <h3 class="ayah-heading">Verset {{ content.ayah }}</h3>
    <section aria-label="Traduction française">
      <h4 class="section-heading">Traduction française</h4>
      <template v-if="content.translation">
        <p class="french-text translation-text">{{ content.translation.text }}</p>
        <p class="source-metadata">
          {{ content.translation.translator }} ·
          <a v-if="translationUrl" :href="translationUrl" target="_blank" rel="noopener noreferrer">
            {{ content.translation.source }}
          </a>
          <span v-else>{{ content.translation.source }}</span>
          · version {{ content.translation.version }}
        </p>
        <details v-if="content.translation.footnotes" class="translation-notes">
          <summary>Notes de la traduction</summary>
          <p class="french-text">{{ content.translation.footnotes }}</p>
        </details>
      </template>
      <p v-else class="unavailable-text">Traduction indisponible pour ce verset.</p>
    </section>

    <section v-if="content.tafsirs.length" class="tafsir-section" aria-label="Tafsir">
      <h4 class="section-heading">Tafsir</h4>
      <div class="tafsir-sources" role="group" :aria-label="`Source du tafsir du verset ${content.ayah}`">
        <button
          v-for="source in sources"
          :key="source.id"
          type="button"
          class="source-button"
          :aria-pressed="selectedSource === source.id"
          :disabled="!content.tafsirs.some((tafsir) => tafsir.source === source.id)"
          @click="selectedSource = source.id"
        >
          {{ source.label }}
        </button>
      </div>
      <template v-if="selectedTafsir">
        <p class="french-text tafsir-text">{{ selectedTafsir.text_fr }}</p>
        <p class="source-metadata">
          {{ selectedTafsir.source_reference }} · version {{ selectedTafsir.version }}
        </p>
      </template>
    </section>
  </article>
</template>

<style scoped>
.french-ayah {
  min-width: 0;
}

.french-ayah + .french-ayah {
  border-top: 1px solid #b8c4d1;
  padding-top: 22px;
}

.ayah-heading {
  margin: 0 0 16px;
  font-size: 18px;
  color: #172033;
}

.section-heading {
  margin: 0 0 10px;
  font-size: 15px;
  color: #274b9f;
}

.french-text {
  margin: 0;
  line-height: 1.8;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}

.source-metadata,
.unavailable-text {
  margin: 10px 0 0;
  font-size: 13px;
  line-height: 1.6;
  color: #526274;
  overflow-wrap: anywhere;
}

.source-metadata a {
  color: #274b9f;
  text-decoration: underline;
}

.translation-notes {
  margin-top: 12px;
  font-size: 14px;
}

.translation-notes summary {
  cursor: pointer;
  color: #274b9f;
}

.translation-notes .french-text {
  margin-top: 10px;
}

.tafsir-section {
  margin-top: 22px;
}

.tafsir-sources {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 14px;
}

.source-button {
  min-height: 42px;
  padding: 0 16px;
  border: 1px solid #9fb0ce;
  border-radius: 999px;
  background: transparent;
  color: #1f3473;
  font-weight: 700;
  cursor: pointer;
}

.source-button[aria-pressed='true'] {
  background: #bccae3;
}

.source-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.source-button:focus-visible,
.source-metadata a:focus-visible,
.translation-notes summary:focus-visible {
  outline: 3px solid rgba(49, 88, 183, 0.24);
  outline-offset: 2px;
}
</style>
