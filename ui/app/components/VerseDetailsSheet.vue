<script setup lang="ts">
import BookOpen from '@lucide/vue/dist/esm/icons/book-open.mjs'
import Copy from '@lucide/vue/dist/esm/icons/copy.mjs'
import X from '@lucide/vue/dist/esm/icons/x.mjs'
import { computed, nextTick, onBeforeUnmount, ref, watch } from 'vue'

import MiniToast from '~/components/MiniToast.vue'
import TajwidLegend from '~/components/TajwidLegend.vue'
import TajwidText from '~/components/TajwidText.vue'
import { useMiniToast } from '~/composables/useMiniToast'
import type { RecognizeResponse } from '~/composables/useRecognition'
import { type TajwidResponse, useTajwid } from '~/composables/useTajwid'
import { parseTajwidToTokens } from '~/utils/parseTajwid'
import { TAJWID_READING_SURFACE_COLOR } from '~/utils/tajwidRules'

const props = defineProps<{
  open: boolean
  result: RecognizeResponse
}>()

const emit = defineEmits<{
  close: []
}>()

const { loading, fetchTajwid, error } = useTajwid()
const tajwidResponse = ref<TajwidResponse | null>(null)
const sheetRef = ref<HTMLElement | null>(null)
const closeButtonRef = ref<HTMLButtonElement | null>(null)
const { message: toastMessage, visible: toastVisible, show: showToast } = useMiniToast()
let previouslyFocusedElement: HTMLElement | null = null

const focusableSelector = [
  'a[href]',
  'button:not([disabled])',
  'input:not([disabled])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
].join(',')

const verseLabel = computed(() => {
  if (!props.result.verse) return ''

  const { start_verse, end_verse } = props.result.verse
  return start_verse === end_verse
    ? `Verset ${start_verse}`
    : `Versets ${start_verse} à ${end_verse}`
})

const topImam = computed(() => props.result.imam_predictions?.[0] ?? null)
const otherImams = computed(() => props.result.imam_predictions?.slice(1) ?? [])
const tajwidAyahs = computed(() =>
  (tajwidResponse.value?.ayahs ?? []).map((ayah) => ({
    number: ayah.number,
    tokens: parseTajwidToTokens(ayah.tajwid_text),
  })),
)
const tajwidTokens = computed(() => tajwidAyahs.value.flatMap((ayah) => ayah.tokens))

watch(
  () => props.open,
  async (isOpen) => {
    document.body.style.overflow = isOpen ? 'hidden' : ''

    if (isOpen) {
      previouslyFocusedElement =
        document.activeElement instanceof HTMLElement ? document.activeElement : null
      await nextTick()

      if (props.open) {
        closeButtonRef.value?.focus()
      }

      return
    }

    restorePreviousFocus()
    tajwidResponse.value = null
  },
  { immediate: true },
)

onBeforeUnmount(() => {
  document.body.style.overflow = ''
  restorePreviousFocus()
})

function restorePreviousFocus() {
  previouslyFocusedElement?.focus()
  previouslyFocusedElement = null
}

function onSheetKeydown(event: KeyboardEvent) {
  if (event.key === 'Escape') {
    event.preventDefault()
    emit('close')
    return
  }

  if (event.key !== 'Tab' || !sheetRef.value) return

  const focusableElements = Array.from(
    sheetRef.value.querySelectorAll<HTMLElement>(focusableSelector),
  )
  const firstElement = focusableElements[0]
  const lastElement = focusableElements.at(-1)

  if (!firstElement || !lastElement) {
    event.preventDefault()
    sheetRef.value.focus()
    return
  }

  if (!sheetRef.value.contains(document.activeElement)) {
    event.preventDefault()
    const nextElement = event.shiftKey ? lastElement : firstElement
    nextElement.focus()
    return
  }

  if (event.shiftKey && document.activeElement === firstElement) {
    event.preventDefault()
    lastElement.focus()
    return
  }

  if (!event.shiftKey && document.activeElement === lastElement) {
    event.preventDefault()
    firstElement.focus()
  }
}

async function toggleTajwid() {
  if (!props.result.verse) return

  if (tajwidResponse.value) {
    tajwidResponse.value = null
    return
  }

  const response = await fetchTajwid(
    props.result.verse.sourate_id,
    props.result.verse.start_verse,
    props.result.verse.end_verse,
  )

  tajwidResponse.value = response
}

async function copyVerse() {
  if (!props.result.verse) return

  const payload = [
    `${props.result.verse.sourate_name} — ${props.result.verse.transliteration}`,
    verseLabel.value,
    props.result.verse.text,
  ].join('\n')

  try {
    await navigator.clipboard.writeText(payload)
    showToast('Verset copié')
  } catch (copyError) {
    console.error(copyError)
  }
}
</script>

<template>
  <teleport to="body">
    <div v-if="open" class="sheet-overlay" @click.self="$emit('close')">
      <section
        ref="sheetRef"
        class="sheet"
        :style="{ '--tajwid-reading-surface': TAJWID_READING_SURFACE_COLOR }"
        role="dialog"
        aria-modal="true"
        aria-label="Passage détecté"
        tabindex="-1"
        @keydown="onSheetKeydown"
      >
        <div class="sheet-handle" />
        <div class="sheet-header">
          <div>
            <p class="sheet-kicker">Passage détecté</p>
            <h2 class="sheet-title">{{ result.verse?.sourate_name }}</h2>
            <p class="sheet-subtitle">
              {{ result.verse?.transliteration }} · {{ verseLabel }}
            </p>
          </div>

          <button
            ref="closeButtonRef"
            class="close-btn"
            type="button"
            aria-label="Fermer"
            @click="$emit('close')"
          >
            <X class="close-icon" :stroke-width="2" aria-hidden="true" />
          </button>
        </div>

        <div class="sheet-scroll">
          <section v-if="result.verse" class="content-card">
            <p class="content-label">Texte coranique</p>
            <p class="arabic-text">{{ result.verse.text }}</p>
          </section>

          <section class="action-card">
            <button class="sheet-btn" type="button" @click="copyVerse">
              <Copy class="sheet-btn-icon" :stroke-width="2" aria-hidden="true" />
              Copier
            </button>

            <button class="sheet-btn" type="button" :disabled="loading" @click="toggleTajwid">
              <BookOpen class="sheet-btn-icon" :stroke-width="2" aria-hidden="true" />
              <span v-if="loading">Chargement du tajwid…</span>
              <span v-else-if="tajwidResponse">Masquer le tajwid</span>
              <span v-else>Afficher le tajwid</span>
            </button>
          </section>

          <section v-if="tajwidResponse" class="content-card tajwid-reading-card">
            <header class="tajwid-reading-header">
              <p class="content-label">Affichage tajwid</p>
              <div class="surah-cartouche">
                <span class="cartouche-line" aria-hidden="true" />
                <h3 class="surah-cartouche-title" lang="ar" dir="rtl">
                  {{ result.verse?.sourate_name }}
                </h3>
                <span class="cartouche-line" aria-hidden="true" />
              </div>
              <p class="tajwid-reading-subtitle">
                {{ result.verse?.transliteration }} · {{ verseLabel }}
              </p>
            </header>

            <div class="mushaf-reading-panel">
              <TajwidText :ayahs="tajwidAyahs" />
            </div>

            <TajwidLegend :tokens="tajwidTokens" />
          </section>

          <p v-if="error" class="error-text">
            {{ error }}
          </p>

          <section v-if="result.transcription_text" class="content-card">
            <p class="content-label">Transcription brute</p>
            <p class="transcription-text">
              {{ result.transcription_text }}
            </p>
          </section>

          <section v-if="topImam || otherImams.length" class="content-card">
            <p class="content-label">Récitateur</p>
            <p v-if="topImam" class="imam-text">
              {{ topImam.name.replace(/_/g, ' ').trim() }}
            </p>

            <div v-if="otherImams.length" class="imam-list">
              <span v-for="imam in otherImams" :key="imam.name" class="imam-chip">
                {{ imam.name.replace(/_/g, ' ').trim() }}
              </span>
            </div>
          </section>
        </div>
      </section>
    </div>
  </teleport>

  <MiniToast :open="toastVisible" :message="toastMessage" />
</template>

<style scoped>
.sheet-overlay {
  position: fixed;
  inset: 0;
  z-index: 9999;
  background: rgba(2, 6, 23, 0.72);
  backdrop-filter: blur(12px);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 18px;
  box-sizing: border-box;
}

.sheet {
  width: min(980px, 100%);
  max-height: min(88vh, 960px);
  display: flex;
  flex-direction: column;
  border-radius: 28px;
  border: 1px solid rgba(148, 163, 184, 0.32);
  background: var(--tajwid-reading-surface, #dce3ea);
  color: #172033;
  box-shadow: 0 30px 100px rgba(2, 6, 23, 0.5);
  overflow: hidden;
}

.sheet-handle {
  align-self: center;
  width: 56px;
  height: 5px;
  border-radius: 999px;
  margin-top: 12px;
  background: #aab7c5;
}

.sheet-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 18px;
  padding: 22px 22px 18px;
  border-bottom: 1px solid #b8c4d1;
}

.sheet-kicker,
.content-label {
  margin: 0;
  font-size: 13px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: #274b9f;
}

.sheet-title {
  margin: 10px 0 0;
  font-size: 34px;
  line-height: 1.1;
  font-family: 'Amiri Quran', 'Amiri', serif;
  color: #111827;
}

.sheet-subtitle {
  margin: 8px 0 0;
  color: #526274;
}

.close-btn,
.sheet-btn {
  border: none;
  border-radius: 999px;
  min-height: 46px;
  padding: 0 18px;
  font-weight: 700;
  cursor: pointer;
  transition: transform 0.2s ease, background 0.2s ease, opacity 0.2s ease;
}

.close-btn,
.sheet-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
}

.close-btn:hover,
.sheet-btn:hover {
  transform: translateY(-1px);
}

.close-btn {
  width: 46px;
  padding: 0;
  border: 1px solid #b8c4d1;
  background: #e8edf2;
  color: #27364a;
}

.close-btn:hover {
  background: #d2dae3;
}

.close-btn:focus-visible,
.sheet-btn:focus-visible {
  outline: 3px solid rgba(49, 88, 183, 0.24);
  outline-offset: 2px;
}

.close-icon,
.sheet-btn-icon {
  width: 18px;
  height: 18px;
  fill: none;
  stroke: currentColor;
  stroke-width: 2;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.sheet-scroll {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding: 18px 22px 22px;
  scrollbar-width: thin;
  overscroll-behavior: contain;
}

.content-card,
.action-card {
  border: 1px solid #b8c4d1;
  border-radius: 8px;
  background: #e8edf2;
  padding: 18px;
}

.content-card + .content-card,
.content-card + .action-card,
.action-card + .content-card {
  margin-top: 18px;
}

.arabic-text {
  margin: 16px 0 0;
  font-size: 38px;
  line-height: 2;
  direction: rtl;
  text-align: right;
  font-family: 'Amiri Quran', 'Amiri', serif;
  color: #172033;
}

.transcription-text {
  margin: 14px 0 0;
  line-height: 1.8;
  direction: rtl;
  text-align: right;
  color: #3f4f62;
}

.action-card {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  background: #ced8e4;
}

.sheet-btn {
  background: #bccae3;
  border: 1px solid #9fb0ce;
  color: #1f3473;
}

.sheet-btn:hover:not(:disabled) {
  background: #afc0dd;
}

.sheet-btn:disabled {
  opacity: 0.7;
  cursor: not-allowed;
}

.imam-text {
  margin: 14px 0 0;
  font-size: 20px;
  font-weight: 700;
  color: #172033;
}

.imam-list {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 14px;
}

.imam-chip {
  display: inline-flex;
  align-items: center;
  min-height: 36px;
  padding: 0 14px;
  border-radius: 999px;
  background: #cbd5e1;
  color: #334155;
}

.error-text {
  margin: 18px 0 0;
  color: #8f1d1d;
}

.tajwid-reading-card {
  position: relative;
  display: grid;
  gap: 22px;
  padding: 26px 28px 22px;
  border: 1px solid #b9a87f;
  border-radius: 18px;
  background: var(--tajwid-reading-surface, #dce3ea);
  box-shadow:
    inset 0 0 0 5px rgba(255, 252, 242, 0.76),
    inset 0 0 0 6px rgba(185, 168, 127, 0.34),
    0 12px 32px rgba(75, 59, 31, 0.08);
}

.tajwid-reading-header {
  display: grid;
  justify-items: center;
  gap: 8px;
  text-align: center;
}

.tajwid-reading-header .content-label {
  color: #806a35;
}

.surah-cartouche {
  width: min(520px, 100%);
  display: grid;
  grid-template-columns: minmax(24px, 1fr) auto minmax(24px, 1fr);
  align-items: center;
  gap: 16px;
}

.cartouche-line {
  height: 1px;
  background: linear-gradient(90deg, transparent, #a68d54);
}

.cartouche-line:last-child {
  background: linear-gradient(90deg, #a68d54, transparent);
}

.surah-cartouche-title {
  margin: 0;
  padding: 4px 18px 6px;
  border: 1px solid rgba(146, 119, 58, 0.7);
  border-radius: 999px;
  color: #3d3423;
  font-family: 'Amiri Quran', 'Amiri', serif;
  font-size: 26px;
  font-weight: 700;
  line-height: 1.25;
}

.tajwid-reading-subtitle {
  margin: 0;
  color: #75694f;
  font-size: 13px;
}

.mushaf-reading-panel {
  padding: 22px 18px;
  border-block: 1px solid rgba(166, 141, 84, 0.34);
  background: rgba(255, 252, 244, 0.38);
}

@media (max-width: 768px) {
  .sheet-overlay {
    align-items: flex-end;
    padding: 0;
  }

  .sheet {
    width: 100%;
    max-height: 92vh;
    border-radius: 28px 28px 0 0;
  }

  .sheet-header {
    padding: 18px 18px 14px;
  }

  .sheet-title {
    font-size: 30px;
  }

  .sheet-scroll {
    padding: 16px 18px 18px;
  }

  .action-card {
    flex-direction: column;
  }

  .arabic-text {
    font-size: 30px;
    line-height: 1.9;
  }

  .tajwid-reading-card {
    gap: 18px;
    padding: 22px 18px 18px;
    border-radius: 14px;
  }

  .surah-cartouche {
    gap: 10px;
  }

  .surah-cartouche-title {
    padding-inline: 14px;
    font-size: 23px;
  }

  .mushaf-reading-panel {
    padding: 18px 4px;
  }
}
</style>
