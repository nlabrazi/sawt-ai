<script setup lang="ts">
import { useHead, useRequestURL } from '#app'
import { BookOpen, CircleHelp, Mic } from '@lucide/vue'
import { Motion, MotionConfig } from 'motion-v'
import { computed, defineAsyncComponent, nextTick, onMounted, ref } from 'vue'
import AppFooter from '~/components/AppFooter.vue'
import LandingScreen from '~/components/LandingScreen.vue'
import MotionScreenTransition from '~/components/MotionScreenTransition.vue'
import { useUiMotion } from '~/composables/useUiMotion'
import QuranRecognitionScreen from '~/components/QuranRecognitionScreen.vue'
import { useHadithSearch } from '~/composables/useHadithSearch'

const HadithSearchScreen = defineAsyncComponent(() => import('~/components/HadithSearchScreen.vue'))
const FaqScreen = defineAsyncComponent(() => import('~/components/FaqScreen.vue'))
const TafsirReviewScreen = defineAsyncComponent(() => import('~/components/TafsirReviewScreen.vue'))
const TermsOfServiceScreen = defineAsyncComponent(
  () => import('~/components/TermsOfServiceScreen.vue'),
)
const LegalNoticeScreen = defineAsyncComponent(() => import('~/components/LegalNoticeScreen.vue'))
const ContactScreen = defineAsyncComponent(() => import('~/components/ContactScreen.vue'))
const pathname = useRequestURL().pathname.replace(/\/$/, '')
const internalReview = pathname === '/internal/tafsir'
useHead(
  internalReview
    ? {
        title: 'Review des tafsirs — Sawt AI',
        meta: [{ name: 'robots', content: 'noindex, nofollow' }],
      }
    : {
        script: [
          {
            src: 'https://umami.nabster.dev/script.js',
            defer: true,
            'data-website-id': '6c8e5246-8c6c-4964-a20d-f8e66169aed6',
          },
        ],
      },
)
type Screen = 'home' | 'quran' | 'hadith' | 'faq'
const mode = ref<Screen>('home')
const { canAnimate, spring } = useUiMotion()
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
  <TafsirReviewScreen v-if="internalReview" />
  <TermsOfServiceScreen v-else-if="pathname === '/terms-of-service'" />
  <LegalNoticeScreen v-else-if="pathname === '/legal-notice'" />
  <ContactScreen v-else-if="pathname === '/contact'" />
  <MotionConfig v-else reduced-motion="user">
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
          <nav aria-label="Navigation principale" class="mode-selector">
            <Motion
              as="span"
              v-if="activeIndex >= 0"
              class="mode-highlight"
              aria-hidden="true"
              :initial="false"
              :animate="{ x: `${activeIndex * 100}%`, opacity: 1 }"
              :transition="spring"
            />
            <Motion
              as="button"
              v-for="item in navigationItems"
              :key="item.mode"
              type="button"
              :aria-pressed="mode === item.mode"
              :disabled="!ready || (navigationLocked && mode !== item.mode)"
              :aria-describedby="navigationLocked && mode !== item.mode ? 'mode-lock-hint' : undefined"
              :while-press="canAnimate && !navigationLocked ? { scale: 0.94 } : undefined"
              :transition="spring"
              @click="selectMode(item.mode)"
            >
              <component :is="item.icon" :size="17" aria-hidden="true" /> {{ item.label }}
            </Motion>
          </nav>
          <p v-if="navigationLocked" id="mode-lock-hint" class="sr-only" role="status">
            Terminez l’enregistrement pour changer d’écran.
          </p>
        </header>
        <div ref="content" class="experience" tabindex="-1">
          <MotionScreenTransition>
            <LandingScreen v-if="mode === 'home'" key="home" :disabled="!ready" @navigate="selectMode" />
            <QuranRecognitionScreen
              v-else-if="mode === 'quran'"
              key="quran"
              @navigation-lock="navigationLocked = $event"
            />
            <HadithSearchScreen v-else-if="mode === 'hadith'" key="hadith" :search-state="hadithSearch" />
            <FaqScreen v-else key="faq" />
          </MotionScreenTransition>
        </div>
        <AppFooter />
      </div>
    </main>
  </MotionConfig>
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

.page::after {
  content: '';
  position: absolute;
  z-index: -1;
  inset: 0;
  pointer-events: none;
  background-image: radial-gradient(rgba(147, 197, 253, .22) .7px, transparent .7px);
  background-size: 28px 28px;
  mask-image: linear-gradient(transparent, #000 30%, transparent 85%);
  opacity: .3;
}

.page-content {
  min-height: 100vh;
  min-height: 100dvh;
  display: flex;
  flex-direction: column;
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
  background: rgba(16, 29, 48, .8);
  box-shadow: 0 8px 32px rgba(0, 0, 0, .12);
}

.mode-highlight {
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
  pointer-events: none;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, .12), 0 2px 12px rgba(37, 99, 235, .15);
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
  .mode-highlight {
    transition: none;
  }
}
</style>
