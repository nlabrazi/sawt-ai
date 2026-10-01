<script setup lang="ts">
import { BookOpen, Mic } from '@lucide/vue'
import { defineAsyncComponent, nextTick, onMounted, ref } from 'vue'
import AppFooter from '~/components/AppFooter.vue'
import QuranRecognitionScreen from '~/components/QuranRecognitionScreen.vue'
import { useHadithSearch } from '~/composables/useHadithSearch'

const HadithSearchScreen = defineAsyncComponent(() => import('~/components/HadithSearchScreen.vue'))
const mode = ref<'quran' | 'hadith'>('quran')
// Server-rendered navigation becomes interactive once Vue has hydrated it.
const ready = ref(false)
onMounted(() => {
  ready.value = true
})
const hadithSearch = useHadithSearch()
const navigationLocked = ref(false)
const content = ref<HTMLElement | null>(null)

async function selectMode(nextMode: 'quran' | 'hadith') {
  if (navigationLocked.value || mode.value === nextMode) return
  if (mode.value === 'hadith') hadithSearch.cancel()
  mode.value = nextMode
  await nextTick()
  content.value?.focus({ preventScroll: true })
}
</script>

<template>
  <main class="page">
    <div class="page-content">
      <div class="app-header">
        <header class="brand" aria-label="Sawt AI">
          <span>Sawt</span><span class="brand-mark">AI</span>
        </header>
        <nav aria-label="Choisir un mode" class="mode-selector">
          <button type="button" :disabled="!ready" :aria-pressed="mode === 'quran'" @click="selectMode('quran')">
            <Mic :size="17" aria-hidden="true" /> Coran
          </button>
          <button
            type="button"
            :aria-pressed="mode === 'hadith'"
            :disabled="!ready || navigationLocked"
            :aria-describedby="navigationLocked ? 'mode-lock-hint' : undefined"
            @click="selectMode('hadith')"
          >
            <BookOpen :size="17" aria-hidden="true" /> Hadiths
          </button>
        </nav>
        <p v-if="navigationLocked" id="mode-lock-hint" class="mode-hint" role="status">
          Terminez l’enregistrement pour changer de mode.
        </p>
      </div>
      <div ref="content" class="experience" tabindex="-1">
        <Transition name="screen-transition" mode="out-in">
          <QuranRecognitionScreen
            v-if="mode === 'quran'"
            key="quran"
            @navigation-lock="navigationLocked = $event"
          />
          <HadithSearchScreen v-else key="hadith" :search-state="hadithSearch" />
        </Transition>
      </div>
      <AppFooter />
    </div>
  </main>
</template>

<style scoped>
:global(html, body, #__nuxt) {
  min-height: 100%;
}

:global(body) {
  margin: 0;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: #07101d;
  color: #fff;
}

:global(*) {
  box-sizing: border-box;
}

.page {
  position: relative;
  min-height: 100vh;
  min-height: 100svh;
  overflow-x: hidden;
  color: #fff;
  background: linear-gradient(180deg, #07111f 0%, #08101c 56%, #060d17 100%);
  isolation: isolate;
}

.page::before {
  content: '';
  position: absolute;
  z-index: -1;
  top: -260px;
  left: 50%;
  width: min(820px, 120vw);
  height: 620px;
  border-radius: 999px;
  background: rgba(37, 99, 235, 0.18);
  filter: blur(120px);
  transform: translateX(-50%);
  pointer-events: none;
}

.page-content {
  min-height: 100vh;
  min-height: 100svh;
  display: flex;
  flex-direction: column;
}

:global(.screen-transition-enter-active) {
  transition:
    opacity 180ms ease-out,
    transform 180ms ease-out;
}

:global(.screen-transition-leave-active) {
  transition:
    opacity 120ms ease-in,
    transform 120ms ease-in;
}

:global(.screen-transition-enter-from) {
  opacity: 0;
  transform: translateY(6px);
}

:global(.screen-transition-leave-to) {
  opacity: 0;
  transform: translateY(-4px);
}

@media (max-width: 640px) {
  .page::before {
    top: -220px;
    height: 520px;
    filter: blur(96px);
  }
}

@media (prefers-reduced-motion: reduce) {
  :global(.screen-transition-enter-active),
  :global(.screen-transition-leave-active) {
    transition: none;
  }
  :global(.screen-transition-enter-from),
  :global(.screen-transition-leave-to) {
    transform: none;
  }
}

.app-header {
  position: relative;
  z-index: 1;
  padding: 24px 20px 0;
  display: grid;
  justify-items: center;
  gap: 18px;
}

.brand {
  display: flex;
  align-items: baseline;
  gap: 5px;
  font-size: 17px;
  line-height: 1;
  font-weight: 800;
  letter-spacing: -.02em;
}

.brand-mark {
  color: #60a5fa;
  font-size: 12px;
  letter-spacing: .04em;
}

.mode-selector {
  display: inline-flex;
  padding: 5px;
  gap: 4px;
  border: 1px solid #2a3d58;
  border-radius: 999px;
  background: #101d30;
}

.mode-selector button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  min-height: 44px;
  min-width: 112px;
  padding: 10px 20px;
  border: 1px solid transparent;
  border-radius: 999px;
  background: transparent;
  color: #a9bdd8;
  font: inherit;
  font-size: 14px;
  cursor: pointer;
  transition: background 160ms, color 160ms;
}

.mode-selector button[aria-pressed='true'] {
  background: #244c7f;
  border-color: #3f6598;
  color: #f4f8ff;
}

.mode-selector button:disabled {
  opacity: .5;
  cursor: default;
}

.mode-selector button:focus-visible {
  outline: 2px solid #93c5fd;
  outline-offset: 3px;
}

.mode-hint {
  margin: 0;
  color: #b4c9e3;
  font-size: 12px;
}

.experience {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
}

.experience:focus {
  outline: none;
}

@media (prefers-reduced-motion: reduce) {
  .mode-selector button {
    transition: none;
  }
}
</style>
