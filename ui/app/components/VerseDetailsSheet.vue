<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, ref, watch } from 'vue'

import { useTajwid } from '~/composables/useTajwid'
import type { RecognizeResponse } from '~/composables/useRecognition'
import { parseTajwidToHtml } from '~/utils/parseTajwid'

const props = defineProps<{
  open: boolean
  result: RecognizeResponse
}>()

const emit = defineEmits<{
  close: []
}>()

const { loading, fetchTajwid, error } = useTajwid()
const tajwidText = ref<string | null>(null)
const sheetRef = ref<HTMLElement | null>(null)
const closeButtonRef = ref<HTMLButtonElement | null>(null)
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
const tajwidHtml = computed(() => {
  if (!tajwidText.value) return ''
  return parseTajwidToHtml(tajwidText.value)
})

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
    tajwidText.value = null
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

  if (tajwidText.value) {
    tajwidText.value = null
    return
  }

  const response = await fetchTajwid(
    props.result.verse.sourate_id,
    props.result.verse.start_verse,
    props.result.verse.end_verse,
  )

  tajwidText.value = response.text
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
            <svg class="close-icon" viewBox="0 0 24 24" aria-hidden="true">
              <path d="M18 6 6 18" />
              <path d="m6 6 12 12" />
            </svg>
          </button>
        </div>

        <div class="sheet-scroll">
          <section v-if="result.verse" class="content-card">
            <p class="content-label">Texte coranique</p>
            <p class="arabic-text">{{ result.verse.text }}</p>
          </section>

          <section class="action-card">
            <button class="sheet-btn" type="button" @click="copyVerse">
              <svg class="sheet-btn-icon" viewBox="0 0 24 24" aria-hidden="true">
                <rect x="9" y="9" width="11" height="11" rx="2" />
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
              </svg>
              Copier
            </button>

            <button class="sheet-btn" type="button" :disabled="loading" @click="toggleTajwid">
              <svg class="sheet-btn-icon" viewBox="0 0 24 24" aria-hidden="true">
                <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
                <path d="M4 4.5A2.5 2.5 0 0 1 6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5z" />
              </svg>
              <span v-if="loading">Chargement du tajwid…</span>
              <span v-else-if="tajwidText">Masquer le tajwid</span>
              <span v-else>Afficher le tajwid</span>
            </button>
          </section>

          <section v-if="tajwidText" class="content-card">
            <p class="content-label">Affichage tajwid</p>
            <p class="arabic-text tajwid-text" v-html="tajwidHtml" />
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
  border: 1px solid rgba(148, 163, 184, 0.14);
  background: linear-gradient(180deg, rgba(7, 16, 30, 0.98) 0%, rgba(6, 14, 27, 0.98) 100%);
  box-shadow: 0 30px 100px rgba(2, 6, 23, 0.50);
  overflow: hidden;
}

.sheet-handle {
  align-self: center;
  width: 56px;
  height: 5px;
  border-radius: 999px;
  margin-top: 12px;
  background: rgba(148, 163, 184, 0.4);
}

.sheet-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 18px;
  padding: 22px 22px 18px;
  border-bottom: 1px solid rgba(148, 163, 184, 0.10);
}

.sheet-kicker,
.content-label {
  margin: 0;
  font-size: 13px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: #93c5fd;
}

.sheet-title {
  margin: 10px 0 0;
  font-size: 34px;
  line-height: 1.1;
  font-family: 'Amiri', serif;
}

.sheet-subtitle {
  margin: 8px 0 0;
  color: #b8c6d8;
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
  background: rgba(255, 255, 255, 0.08);
  color: #e2e8f0;
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
  border: 1px solid rgba(148, 163, 184, 0.10);
  border-radius: 24px;
  background: rgba(8, 18, 34, 0.72);
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
  font-family: 'Amiri', serif;
  color: #f8fafc;
}

.transcription-text {
  margin: 14px 0 0;
  line-height: 1.8;
  direction: rtl;
  text-align: right;
  color: #cbd5e1;
}

.action-card {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
}

.sheet-btn {
  background: rgba(30, 64, 175, 0.22);
  border: 1px solid rgba(96, 165, 250, 0.20);
  color: #eff6ff;
}

.sheet-btn:disabled {
  opacity: 0.7;
  cursor: not-allowed;
}

.imam-text {
  margin: 14px 0 0;
  font-size: 20px;
  font-weight: 700;
  color: #eff6ff;
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
  background: rgba(255, 255, 255, 0.05);
  color: #dbeafe;
}

.error-text {
  margin: 18px 0 0;
  color: #fecaca;
}

:deep(.tajwid-fragment) {
  color: #f8fafc;
}

:deep(.tajwid-rule-ghn),
:deep(.tajwid-rule-g),
:deep(.tajwid-rule-idgham),
:deep(.tajwid-rule-i) {
  color: #f59e0b;
}

:deep(.tajwid-rule-ikhf),
:deep(.tajwid-rule-k) {
  color: #22c55e;
}

:deep(.tajwid-rule-qlq),
:deep(.tajwid-rule-q) {
  color: #38bdf8;
}

:deep(.tajwid-rule-m),
:deep(.tajwid-rule-p) {
  color: #f472b6;
}

:deep(.tajwid-rule-n) {
  color: #a78bfa;
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
}
</style>
