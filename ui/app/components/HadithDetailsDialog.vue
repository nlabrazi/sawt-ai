<script setup lang="ts">
import { ArrowUpRight, X } from '@lucide/vue'
import { onBeforeUnmount, onMounted, ref } from 'vue'
import type { HadithResult } from '~/types/hadith'

defineProps<{ hadith: HadithResult }>()
const emit = defineEmits<{ close: [] }>()
const dialog = ref<HTMLDialogElement | null>(null)
let previousFocus: HTMLElement | null = null
let previousOverflow = ''

onMounted(() => {
  previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null
  previousOverflow = document.body.style.overflow
  document.body.style.overflow = 'hidden'
  dialog.value?.showModal()
})

onBeforeUnmount(() => {
  dialog.value?.close()
  document.body.style.overflow = previousOverflow
  if (previousFocus?.isConnected) previousFocus.focus()
})
</script>

<template>
  <Teleport to="body">
    <dialog
      ref="dialog"
      aria-labelledby="hadith-reading-title"
      @cancel.prevent="emit('close')"
      @click.self="emit('close')"
    >
      <div class="reading-sheet">
        <header class="reading-header">
          <span>{{ hadith.provider }}</span>
          <button type="button" aria-label="Fermer la lecture du hadith" autofocus @click="emit('close')">
            <X :size="22" aria-hidden="true" />
          </button>
        </header>
        <div class="reading-content">
          <h2 id="hadith-reading-title">{{ hadith.title }}</h2>
          <p v-if="hadith.grade" class="grade">{{ hadith.grade }}</p>
          <section aria-label="Texte arabe" class="arabic-surface">
            <p lang="ar" dir="rtl" class="arabic-text">{{ hadith.arabic }}</p>
          </section>
          <section aria-labelledby="hadith-translation-title">
            <h3 id="hadith-translation-title">Traduction française</h3>
            <p class="source-text">{{ hadith.translation }}</p>
          </section>
          <section v-if="hadith.explanation" aria-labelledby="hadith-explanation-title">
            <h3 id="hadith-explanation-title">Explication</h3>
            <p class="source-text">{{ hadith.explanation }}</p>
          </section>
          <section v-if="hadith.attribution" aria-labelledby="hadith-attribution-title">
            <h3 id="hadith-attribution-title">Attribution</h3>
            <p class="source-text">{{ hadith.attribution }}</p>
          </section>
          <a class="source-link" :href="hadith.source_url" target="_blank" rel="noopener noreferrer">
            Consulter HadeethEnc <ArrowUpRight :size="18" aria-hidden="true" />
            <span class="sr-only"> (nouvel onglet)</span>
          </a>
        </div>
      </div>
    </dialog>
  </Teleport>
</template>

<style scoped>
dialog {
  width: min(760px, calc(100% - 32px));
  max-height: calc(100dvh - 48px);
  margin: auto;
  padding: 0;
  border: 1px solid #34465f;
  border-radius: 24px;
  background: #101c2d;
  color: #eef4fc;
  box-shadow: 0 30px 100px #0008;
}

dialog[open] {
  animation: reading-enter 180ms ease-out;
}

dialog::backdrop {
  background: #030812bd;
  backdrop-filter: blur(5px);
}

.reading-sheet {
  overflow-wrap: anywhere;
}

.reading-header {
  position: sticky;
  top: 0;
  z-index: 1;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 14px 22px;
  background: #101c2df5;
  border-bottom: 1px solid #293a51;
  color: #b9c9df;
  font-size: 13px;
}

button {
  display: grid;
  place-items: center;
  width: 44px;
  height: 44px;
  flex-shrink: 0;
  border: 1px solid #334762;
  border-radius: 12px;
  color: #dce9fc;
  background: #1b2b41;
  cursor: pointer;
}

.reading-content {
  padding: 28px;
}

h2 {
  margin: 0;
  font-size: 25px;
  line-height: 1.45;
}

h3 {
  margin: 28px 0 12px;
  font-size: 16px;
  color: #c9ddfa;
}

.grade {
  color: #b6d8c6;
  line-height: 1.6;
}

.arabic-surface {
  margin-top: 24px;
  padding: 24px;
  border-radius: 18px;
  background: #eaf0e8;
  color: #142d24;
}

.arabic-text {
  margin: 0;
  font-family: Amiri, serif;
  font-size: 28px;
  line-height: 2;
  white-space: pre-line;
}

.source-text {
  margin: 0;
  color: #c3cfdf;
  line-height: 1.85;
  white-space: pre-line;
}

.source-link {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  min-height: 44px;
  margin-top: 28px;
  color: #93c5fd;
}

button:focus-visible, a:focus-visible {
  outline: 2px solid #93c5fd;
  outline-offset: 3px;
}

.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  overflow: hidden;
  clip-path: inset(50%);
  white-space: nowrap;
}

@keyframes reading-enter {
  from {
    opacity: 0;
    transform: translateY(12px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@media (max-width: 640px) {
  dialog {
    width: 100%;
    max-width: 100%;
    max-height: calc(100dvh - env(safe-area-inset-top, 0px) - 12px);
    margin: auto 0 0;
    border-radius: 24px 24px 0 0;
  }
  .reading-content {
    padding: 22px max(22px, env(safe-area-inset-right, 0px)) calc(22px + env(safe-area-inset-bottom, 0px)) max(22px, env(safe-area-inset-left, 0px));
  }
  h2 {
    font-size: 22px;
  }
  .arabic-surface {
    padding: 18px;
  }
}

@media (prefers-reduced-motion: reduce) {
  dialog[open] {
    animation: none;
  }
}
</style>
