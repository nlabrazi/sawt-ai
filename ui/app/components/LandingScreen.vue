<script setup lang="ts">
import { ArrowRight, BookOpen, Mic } from '@lucide/vue'
import { Motion } from 'motion-v'
import MotionReveal from '~/components/MotionReveal.vue'
import { useUiMotion } from '~/composables/useUiMotion'

defineProps<{ disabled?: boolean }>()
defineEmits<{ navigate: [mode: 'quran' | 'hadith'] }>()
const { canAnimate, spring } = useUiMotion()
const bars = [14, 24, 38, 28, 48, 60, 42, 68, 42, 60, 48, 28, 38, 24, 14]
</script>

<template>
  <section class="landing-screen" aria-labelledby="landing-title">
    <div class="landing-content">
      <MotionReveal class="product-animation" aria-hidden="true" :distance="10">
        <div class="signal-halo" />
        <Motion
          class="signal-orbit"
          :initial="false"
          :animate="{ rotate: canAnimate ? 360 : 0 }"
          :transition="{ type: 'tween', duration: canAnimate ? 36 : 0, repeat: canAnimate ? Infinity : 0, ease: 'linear' }"
        >
          <span class="orbit-dot" />
        </Motion>
        <div class="signal-disc">
          <div class="sound-wave">
            <Motion
              v-for="(height, index) in bars"
              :key="index"
              as="span"
              :style="{ height: `${height}px` }"
              :initial="false"
              :animate="{ scaleY: canAnimate ? [0.45, 1, 0.45] : 0.65, opacity: canAnimate ? [0.55, 1, 0.55] : 1 }"
              :transition="{ type: 'tween', duration: canAnimate ? 2.6 : 0, delay: canAnimate ? index * 0.09 : 0, repeat: canAnimate ? Infinity : 0, ease: 'easeInOut' }"
            />
          </div>
        </div>
        <Motion
          class="symbol-badge microphone-badge"
          :initial="false"
          :animate="{ y: canAnimate ? [0, -7, 0] : 0 }"
          :transition="{ type: 'tween', duration: canAnimate ? 5 : 0, repeat: canAnimate ? Infinity : 0, ease: 'easeInOut' }"
        ><Mic :size="25" :stroke-width="1.6" /></Motion>
        <Motion
          class="symbol-badge book-badge"
          :initial="false"
          :animate="{ y: canAnimate ? [0, 7, 0] : 0 }"
          :transition="{ type: 'tween', duration: canAnimate ? 6 : 0, repeat: canAnimate ? Infinity : 0, ease: 'easeInOut' }"
        ><BookOpen :size="25" :stroke-width="1.6" /></Motion>
      </MotionReveal>
      <MotionReveal :delay="0.08">
        <h1 id="landing-title">Retrouvez les mots<br /><span>qui vous inspirent.</span></h1>
      </MotionReveal>
      <MotionReveal :delay="0.16">
        <p class="landing-description">
        <span>Une récitation, un passage du Coran.</span>
        <span>Quelques mots, un hadith à retrouver.</span>
        <span>Sawt AI vous accompagne dans votre recherche.</span>
        </p>
      </MotionReveal>
      <MotionReveal class="landing-actions" :delay="0.24">
        <Motion
          as="button"
          class="primary-action"
          type="button"
          :disabled="disabled"
          :while-hover="canAnimate && !disabled ? { y: -3, scale: 1.025 } : undefined"
          :while-press="canAnimate && !disabled ? { scale: 0.97 } : undefined"
          :transition="spring"
          @click="$emit('navigate', 'quran')"
        >
          <Mic :size="18" aria-hidden="true" /> Explorer le Coran
          <ArrowRight :size="17" aria-hidden="true" />
        </Motion>
        <Motion
          as="button"
          class="secondary-action"
          type="button"
          :disabled="disabled"
          :while-hover="canAnimate && !disabled ? { y: -3 } : undefined"
          :while-press="canAnimate && !disabled ? { scale: 0.97 } : undefined"
          :transition="spring"
          @click="$emit('navigate', 'hadith')"
        >
          <BookOpen :size="18" aria-hidden="true" /> Rechercher un hadith
        </Motion>
      </MotionReveal>
    </div>
  </section>
</template>

<style scoped>
.landing-screen {
  flex: 1;
  display: grid;
  place-items: center;
  padding: 36px 20px;
}

.landing-content {
  width: min(100%, 680px);
  text-align: center;
}

.product-animation {
  position: relative;
  display: grid;
  place-items: center;
  width: 280px;
  height: 218px;
  margin: 0 auto 26px;
  color: #93c5fd;
}

.signal-halo {
  position: absolute;
  inset: -30px;
  background: radial-gradient(ellipse, rgba(56, 189, 248, .14), rgba(37, 99, 235, .08) 40%, transparent 68%);
}

.signal-orbit {
  position: absolute;
  width: 208px;
  height: 208px;
  border: 1px solid rgba(147, 197, 253, .16);
  border-radius: 50%;
}

.orbit-dot {
  position: absolute;
  top: 20px;
  left: 34px;
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #7dd3fc;
  box-shadow: 0 0 16px rgba(125, 211, 252, .7);
}

.signal-disc {
  width: 152px;
  height: 152px;
  display: grid;
  place-items: center;
  border: 1px solid rgba(147, 197, 253, .3);
  border-radius: 50%;
  background: radial-gradient(circle at 35% 20%, #183c63, #0c1c32 70%);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, .12), 0 20px 60px rgba(0, 0, 0, .25);
}

.symbol-badge {
  position: absolute;
  display: grid;
  place-items: center;
  width: 52px;
  height: 52px;
  border: 1px solid rgba(147, 197, 253, .24);
  border-radius: 17px;
  background: linear-gradient(145deg, #183552, #0b1b2f);
  box-shadow: 0 12px 30px rgba(0, 0, 0, .25);
}

.microphone-badge { left: 8px; top: 42px; }
.book-badge { right: 8px; bottom: 34px; color: #a5b4fc; }

.sound-wave {
  display: flex;
  align-items: center;
  gap: 4px;
  height: 68px;
}

.sound-wave span {
  width: 4px;
  border-radius: 999px;
  background: linear-gradient(180deg, #a5f3fc, #60a5fa 65%, #818cf8);
}

h1 {
  margin: 0;
  font-size: clamp(34px, 5vw, 60px);
  line-height: 1.12;
  letter-spacing: -.045em;
  text-wrap: balance;
}

h1 span {
  color: #93c5fd;
  background: linear-gradient(100deg, #bfdbfe, #7dd3fc 45%, #a5b4fc);
  background-clip: text;
  -webkit-text-fill-color: transparent;
}

.landing-description {
  display: grid;
  gap: 6px;
  margin: 24px 0 30px;
  color: #aebed3;
  font-size: 15px;
  line-height: 1.6;
  text-wrap: balance;
}

.landing-actions {
  display: flex;
  justify-content: center;
  flex-wrap: wrap;
  gap: 12px;
}

button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  min-height: 48px;
  padding: 12px 18px;
  border: 1px solid #3c506f;
  border-radius: 14px;
  color: #e5efff;
  font: inherit;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: background 160ms ease;
}

.primary-action {
  background: linear-gradient(135deg, #3b82f6, #2563eb);
  border-color: #4982f8;
  color: #fff;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, .2), 0 8px 28px rgba(37, 99, 235, .25);
}

.primary-action:hover {
  background: #3472f2;
}

.secondary-action {
  background: #101d30;
}

.secondary-action:hover {
  background: #1a2e49;
}

button:focus-visible {
  outline: 2px solid #93c5fd;
  outline-offset: 4px;
}

button:disabled {
  opacity: .6;
  cursor: default;
}

@media (max-width: 640px) {
  .landing-screen { padding: 24px 16px; }
  .product-animation { height: 190px; margin-bottom: 20px; }
  .signal-orbit { width: 186px; height: 186px; }
  .landing-description { font-size: 14px; margin: 20px 0 24px; }
  .landing-actions { flex-direction: column; align-items: center; }
  button { width: min(100%, 280px); }
}

@media (prefers-reduced-motion: reduce) {
  button { transition: none; }
}
</style>
