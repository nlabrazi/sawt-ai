<script setup lang="ts">
import { ArrowRight, BookOpen, Mic } from '@lucide/vue'

defineProps<{ disabled?: boolean }>()
defineEmits<{ navigate: [mode: 'quran' | 'hadith'] }>()
</script>

<template>
  <section class="landing-screen" aria-labelledby="landing-title">
    <div class="landing-content">
      <div class="product-animation" aria-hidden="true">
        <Mic :size="28" :stroke-width="1.6" />
        <div class="sound-wave">
          <span v-for="bar in 7" :key="bar" :style="{ '--bar': bar }" />
        </div>
        <BookOpen :size="28" :stroke-width="1.6" />
      </div>
      <h1 id="landing-title">Retrouvez les mots<br />qui vous inspirent.</h1>
      <p class="landing-description">
        <span>Une récitation, un passage du Coran.</span>
        <span>Quelques mots, un hadith à retrouver.</span>
        <span>Sawt AI vous accompagne dans votre recherche.</span>
      </p>
      <div class="landing-actions">
        <button class="primary-action" type="button" :disabled="disabled" @click="$emit('navigate', 'quran')">
          <Mic :size="18" aria-hidden="true" /> Explorer le Coran
          <ArrowRight :size="17" aria-hidden="true" />
        </button>
        <button class="secondary-action" type="button" :disabled="disabled" @click="$emit('navigate', 'hadith')">
          <BookOpen :size="18" aria-hidden="true" /> Rechercher un hadith
        </button>
      </div>
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
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 24px;
  height: 72px;
  margin-bottom: 24px;
  color: #93c5fd;
}

.sound-wave {
  display: flex;
  align-items: center;
  gap: 5px;
  height: 40px;
}

.sound-wave span {
  width: 4px;
  height: 28px;
  border-radius: 999px;
  background: #60a5fa;
  transform: scaleY(.35);
  animation: sound-wave 2.4s ease-in-out infinite;
  animation-delay: calc(var(--bar) * -180ms);
}

h1 {
  margin: 0;
  font-size: clamp(32px, 5vw, 54px);
  line-height: 1.12;
  letter-spacing: -.045em;
  text-wrap: balance;
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
  background: #2563eb;
  border-color: #4982f8;
  color: #fff;
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

@keyframes sound-wave {
  0%, 100% { transform: scaleY(.35); opacity: .6; }
  50% { transform: scaleY(1); opacity: 1; }
}

@media (max-width: 640px) {
  .landing-screen { padding: 24px 16px; }
  .product-animation { margin-bottom: 16px; }
  .landing-description { font-size: 14px; margin: 20px 0 24px; }
  .landing-actions { flex-direction: column; align-items: center; }
  button { width: min(100%, 280px); }
}

@media (prefers-reduced-motion: reduce) {
  .sound-wave span { animation: none; }
  .sound-wave span:nth-child(even) { transform: scaleY(.7); }
  button { transition: none; }
}
</style>
