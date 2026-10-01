<script setup lang="ts">
import { BookOpen, CircleHelp, Mic } from '@lucide/vue'
import { computed, defineAsyncComponent, nextTick, onMounted, ref } from 'vue'
import AppFooter from '~/components/AppFooter.vue'
import LandingScreen from '~/components/LandingScreen.vue'
import QuranRecognitionScreen from '~/components/QuranRecognitionScreen.vue'
import { useHadithSearch } from '~/composables/useHadithSearch'

const HadithSearchScreen = defineAsyncComponent(() => import('~/components/HadithSearchScreen.vue'))
const FaqScreen = defineAsyncComponent(() => import('~/components/FaqScreen.vue'))
type Screen = 'home' | 'quran' | 'hadith' | 'faq'
const mode = ref<Screen>('home')
const navigationItems = [
  { mode: 'quran', label: 'Coran', icon: Mic },
  { mode: 'hadith', label: 'Hadiths', icon: BookOpen },
  { mode: 'faq', label: 'FAQ', icon: CircleHelp },
] as const
const activeIndex = computed(() => navigationItems.findIndex((item) => item.mode === mode.value))
// Server-rendered navigation becomes interactive once Vue has hydrated it.
const ready = ref(false)
onMounted(() => {
  ready.value = true
})
const hadithSearch = useHadithSearch()
const navigationLocked = ref(false)
const content = ref<HTMLElement | null>(null)

async function selectMode(nextMode: Screen) {
  if (navigationLocked.value || mode.value === nextMode) return
  if (mode.value === 'hadith') hadithSearch.reset()
  mode.value = nextMode
  await nextTick()
  content.value?.focus({ preventScroll: true })
}

async function returnHome() {
  if (navigationLocked.value) return
  hadithSearch.reset()
  await selectMode('home')
}
</script>

<template>
  <main class="page">
    <div class="page-content">
      <header class="app-header">
        <button
          class="brand"
          type="button"
          aria-label="Sawt AI — Accueil et réinitialisation"
          :disabled="!ready || navigationLocked"
          :aria-describedby="navigationLocked ? 'mode-lock-hint' : undefined"
          @click="returnHome"
        >
          <span>Sawt</span><span class="brand-mark">AI</span>
        </button>
        <nav
          aria-label="Navigation principale"
          class="mode-selector"
          :class="{ 'has-active-mode': activeIndex >= 0 }"
          :style="{ '--active-index': Math.max(0, activeIndex) }"
        >
          <button
            v-for="item in navigationItems"
            :key="item.mode"
            type="button"
            :aria-pressed="mode === item.mode"
            :disabled="!ready || (navigationLocked && mode !== item.mode)"
            :aria-describedby="navigationLocked && mode !== item.mode ? 'mode-lock-hint' : undefined"
            @click="selectMode(item.mode)"
          >
            <component :is="item.icon" :size="17" aria-hidden="true" /> {{ item.label }}
          </button>
        </nav>
        <p v-if="navigationLocked" id="mode-lock-hint" class="sr-only" role="status">
          Terminez l’enregistrement pour changer d’écran.
        </p>
      </header>
      <div ref="content" class="experience" tabindex="-1">
        <Transition name="screen-transition" mode="out-in">
          <LandingScreen v-if="mode === 'home'" key="home" :disabled="!ready" @navigate="selectMode" />
          <QuranRecognitionScreen
            v-else-if="mode === 'quran'"
            key="quran"
            @navigation-lock="navigationLocked = $event"
          />
          <HadithSearchScreen v-else-if="mode === 'hadith'" key="hadith" :search-state="hadithSearch" />
          <FaqScreen v-else key="faq" />
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

:global(html) {
  color-scheme: dark;
}

:global(button, summary, a) {
  touch-action: manipulation;
}

:global(.sr-only) {
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

:global(*) {
  box-sizing: border-box;
}

.page {
  position: relative;
  min-height: 100vh;
  min-height: 100dvh;
  overflow-x: clip;
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
  min-height: 100dvh;
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
  width: min(100%, 980px);
  margin: 0 auto;
  padding: calc(20px + env(safe-area-inset-top, 0px)) max(20px, env(safe-area-inset-right, 0px)) 0 max(20px, env(safe-area-inset-left, 0px));
  display: grid;
  justify-items: center;
  gap: 12px;
}

.brand {
  display: flex;
  align-items: baseline;
  justify-content: center;
  gap: 5px;
  min-height: 44px;
  padding: 8px 16px;
  border: 0;
  border-radius: 12px;
  background: transparent;
  color: #fff;
  font-family: inherit;
  font-size: 23px;
  cursor: pointer;
  line-height: 1;
  font-weight: 800;
  letter-spacing: -.02em;
}

.brand-mark {
  color: #60a5fa;
  font-size: 15px;
  letter-spacing: .04em;
}

.mode-selector {
  position: relative;
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  width: min(100%, 330px);
  isolation: isolate;
  padding: 3px;
  gap: 0;
  border: 1px solid #2a3d58;
  border-radius: 999px;
  background: #101d30;
}

.mode-selector::before {
  content: '';
  box-sizing: border-box;
  position: absolute;
  z-index: -1;
  top: 3px;
  bottom: 3px;
  left: 3px;
  width: calc((100% - 6px) / 3);
  border: 1px solid #3f6598;
  border-radius: 999px;
  background: #244c7f;
  opacity: 0;
  transform: translateX(calc(var(--active-index) * 100%));
  transition: transform 220ms ease, opacity 160ms ease;
}

.mode-selector.has-active-mode::before {
  opacity: 1;
}

.mode-selector button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  min-height: 44px;
  min-width: 0;
  padding: 10px 12px;
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
  color: #f4f8ff;
}

.mode-selector button svg {
  flex-shrink: 0;
}

.mode-selector button:disabled,
.brand:disabled {
  opacity: .5;
  cursor: default;
}

.mode-selector button:focus-visible,
.brand:focus-visible {
  outline: 2px solid #93c5fd;
  outline-offset: 3px;
}

.experience {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
  padding-left: env(safe-area-inset-left, 0px);
  padding-right: env(safe-area-inset-right, 0px);
}

.experience:focus {
  outline: none;
}

@media (max-width: 640px) {
  .app-header {
    padding-inline: max(16px, env(safe-area-inset-left, 0px)) max(16px, env(safe-area-inset-right, 0px));
    gap: 12px;
  }

  .mode-selector button {
    padding-inline: 8px;
    gap: 6px;
  }
}

@media (prefers-reduced-motion: reduce) {
  .mode-selector button,
  .mode-selector::before {
    transition: none;
  }
}
</style>
