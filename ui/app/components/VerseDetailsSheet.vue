<script setup lang="ts">
import { computed, ref, watch } from 'vue'
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

const showTajwid = ref(false)
const tajwidText = ref<string | null>(null)
const copied = ref(false)

const { loading, fetchTajwid, error } = useTajwid()

const otherImams = computed(() => props.result.imam_predictions?.slice(1) ?? [])

const verseLabel = computed(() => {
  const verse = props.result.verse
  if (!verse) return ''
  return verse.start_verse === verse.end_verse
    ? `Verset ${verse.start_verse}`
    : `Versets ${verse.start_verse} à ${verse.end_verse}`
})

const tajwidHtml = computed(() => {
  if (!tajwidText.value) return ''
  return parseTajwidToHtml(tajwidText.value)
})

watch(() => props.open, (isOpen) => {
  if (!isOpen) {
    showTajwid.value = false
  }
})

async function toggleTajwid() {
  const verse = props.result.verse
  if (!verse) return

  if (showTajwid.value) {
    showTajwid.value = false
    return
  }

  if (!tajwidText.value) {
    const response = await fetchTajwid(verse.sourate_id, verse.start_verse, verse.end_verse)
    tajwidText.value = response.text
  }

  showTajwid.value = true
}

async function copyVerse() {
  const verse = props.result.verse
  if (!verse) return

  const text = [
    `${verse.sourate_name} — ${verse.transliteration}`,
    verseLabel.value,
    verse.text,
  ].join('\n')

  try {
    await navigator.clipboard.writeText(text)
    copied.value = true
    window.setTimeout(() => {
      copied.value = false
    }, 1200)
  } catch (err) {
    console.error(err)
  }
}
</script>

<template>
  <transition name="sheet-fade">
    <div v-if="open" class="sheet-backdrop" @click.self="$emit('close')">
      <transition name="sheet-slide">
        <section class="sheet" role="dialog" aria-modal="true">
          <div class="sheet-handle" />
          <div class="sheet-head">
            <div>
              <p class="sheet-label">Passage détecté</p>
              <h2 class="sheet-title">
                {{ result.verse?.sourate_name }}
              </h2>
              <p class="sheet-subtitle">
                {{ result.verse?.transliteration }} · {{ verseLabel }}
              </p>
            </div>

            <button class="close-btn" type="button" @click="$emit('close')">
              Fermer
            </button>
          </div>

          <div class="sheet-body">
            <section class="block">
              <p class="block-label">Texte coranique</p>
              <p class="arabic-text">
                {{ result.verse?.text }}
              </p>
            </section>

            <section class="block actions-inline">
              <button class="chip-btn" type="button" @click="copyVerse">
                {{ copied ? 'Copié' : 'Copier le verset' }}
              </button>

              <button class="chip-btn" type="button" :disabled="loading" @click="toggleTajwid">
                <span v-if="loading">Chargement…</span>
                <span v-else-if="showTajwid">Masquer le tajwid</span>
                <span v-else>Afficher le tajwid</span>
              </button>
            </section>

            <section v-if="showTajwid && tajwidHtml" class="block">
              <p class="block-label">Affichage tajwid</p>
              <p class="arabic-text tajwid-text" v-html="tajwidHtml" />
            </section>

            <p v-if="error" class="error-text">
              {{ error }}
            </p>

            <section v-if="result.transcription_text" class="block">
              <p class="block-label">Transcription brute</p>
              <p class="transcription-text">
                {{ result.transcription_text }}
              </p>
            </section>

            <section v-if="otherImams.length" class="block">
              <p class="block-label">Autres récitateurs suggérés</p>
              <div class="imam-list">
                <span v-for="imam in otherImams" :key="imam.name" class="imam-chip">
                  {{ imam.name.replace(/_/g, ' ').trim() }}
                </span>
              </div>
            </section>
          </div>
        </section>
      </transition>
    </div>
  </transition>
</template>

<style scoped>
.sheet-backdrop {
  position: fixed;
  inset: 0;
  z-index: 25;
  background: rgba(2, 6, 23, 0.56);
  backdrop-filter: blur(10px);
  display: flex;
  align-items: flex-end;
  justify-content: center;
  padding: 0 14px 14px;
}

.sheet {
  width: min(860px, 100%);
  max-height: 88vh;
  overflow: auto;
  border-radius: 30px;
  border: 1px solid rgba(148, 163, 184, 0.12);
  background: linear-gradient(180deg, rgba(4, 11, 28, 0.98), rgba(1, 7, 19, 0.98));
  box-shadow: 0 -18px 60px rgba(2, 6, 23, 0.42);
  padding: 14px 18px 18px;
}

.sheet-handle {
  width: 54px;
  height: 5px;
  margin: 0 auto 14px;
  border-radius: 999px;
  background: rgba(148, 163, 184, 0.26);
}

.sheet-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 18px;
}

.sheet-label,
.block-label {
  margin: 0;
  font-size: 12px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: #93c5fd;
}

.sheet-title {
  margin: 10px 0 0;
  font-size: 30px;
  line-height: 1;
}

.sheet-subtitle {
  margin: 10px 0 0;
  color: #94a3b8;
}

.close-btn,
.chip-btn {
  border: none;
  border-radius: 999px;
  min-height: 42px;
  padding: 0 16px;
  font-weight: 700;
  cursor: pointer;
}

.close-btn {
  background: rgba(255, 255, 255, 0.06);
  color: #e2e8f0;
}

.sheet-body {
  margin-top: 20px;
  display: grid;
  gap: 18px;
}

.block {
  padding: 18px;
  border-radius: 24px;
  background: rgba(7, 18, 43, 0.56);
  border: 1px solid rgba(148, 163, 184, 0.08);
}

.actions-inline {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.chip-btn {
  background: rgba(59, 130, 246, 0.14);
  color: #dbeafe;
  box-shadow: inset 0 0 0 1px rgba(96, 165, 250, 0.16);
}

.arabic-text {
  margin: 10px 0 0;
  font-family: "Amiri", serif;
  direction: rtl;
  text-align: right;
  font-size: 32px;
  line-height: 1.95;
  color: #f8fafc;
}

.transcription-text {
  margin: 10px 0 0;
  direction: rtl;
  text-align: right;
  line-height: 1.8;
  color: #cbd5e1;
}

.imam-list {
  margin-top: 12px;
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.imam-chip {
  display: inline-flex;
  align-items: center;
  min-height: 34px;
  padding: 0 12px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.06);
  color: #e2e8f0;
}

.error-text {
  margin: 0;
  color: #fca5a5;
}

.sheet-fade-enter-active,
.sheet-fade-leave-active,
.sheet-slide-enter-active,
.sheet-slide-leave-active {
  transition: all 0.22s ease;
}

.sheet-fade-enter-from,
.sheet-fade-leave-to {
  opacity: 0;
}

.sheet-slide-enter-from,
.sheet-slide-leave-to {
  opacity: 0;
  transform: translateY(18px);
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
  .sheet-backdrop {
    padding: 0;
  }

  .sheet {
    width: 100%;
    max-height: 92vh;
    border-bottom-left-radius: 0;
    border-bottom-right-radius: 0;
    padding: 12px 14px 18px;
  }

  .sheet-head {
    flex-direction: column;
  }

  .close-btn,
  .chip-btn {
    width: 100%;
  }

  .actions-inline {
    flex-direction: column;
  }

  .arabic-text {
    font-size: 29px;
  }
}
</style>
