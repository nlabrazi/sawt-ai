<script setup lang="ts">
import { ArrowUpRight, BookOpen } from '@lucide/vue'
import type { HadithResult } from '~/types/hadith'

defineProps<{ hadith: HadithResult; position: number }>()
defineEmits<{ read: [hadith: HadithResult] }>()
</script>

<template>
  <article class="hadith-card">
    <div class="card-meta">
      <span class="result-number">Proposition {{ position }}</span>
      <span class="source-label">{{ hadith.provider }}</span>
    </div>
    <h3>{{ hadith.title }}</h3>
    <p class="excerpt">{{ hadith.translation }}</p>
    <p v-if="hadith.grade" class="grade">{{ hadith.grade }}</p>
    <div class="card-actions">
      <button type="button" @click="$emit('read', hadith)">
        <BookOpen :size="17" aria-hidden="true" /> Lire le hadith
        <span class="sr-only"> : {{ hadith.title }}</span>
      </button>
      <a :href="hadith.source_url" target="_blank" rel="noopener noreferrer">
        Source <ArrowUpRight :size="16" aria-hidden="true" />
        <span class="sr-only"> HadeethEnc : {{ hadith.title }} (nouvel onglet)</span>
      </a>
    </div>
  </article>
</template>

<style scoped>
.hadith-card {
  padding: 24px;
  border: 1px solid #243349;
  border-radius: 22px;
  background: linear-gradient(140deg, #111f32, #0d1728);
  transition: border-color 160ms, transform 160ms;
}

.hadith-card:hover {
  border-color: #48668d;
  transform: translateY(-2px);
}

.card-meta, .card-actions {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.card-meta {
  color: #a7bbd5;
  font-size: 12px;
}

.result-number {
  color: #93c5fd;
}

h3 {
  margin: 16px 0 12px;
  font-size: 20px;
  line-height: 1.45;
  overflow-wrap: anywhere;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.excerpt {
  margin: 0;
  color: #b7c5d8;
  line-height: 1.7;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
  white-space: pre-line;
}

.grade {
  color: #b6d8c6;
  font-size: 13px;
  line-height: 1.5;
}

.card-actions {
  margin-top: 20px;
  flex-wrap: wrap;
}

button, a {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  min-height: 44px;
  border-radius: 12px;
  font: inherit;
  font-size: 14px;
}

button {
  padding: 10px 14px;
  border: 1px solid #355276;
  background: #193354;
  color: #e1eeff;
  cursor: pointer;
}

a {
  color: #b0c9eb;
  text-decoration: none;
  padding: 8px;
}

button:hover {
  background: #23476f;
}

button:focus-visible, a:focus-visible {
  outline: 2px solid #93c5fd;
  outline-offset: 4px;
}

.sr-only {
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

@media (max-width: 640px) {
  .hadith-card {
    padding: 20px;
  }
}

@media (prefers-reduced-motion: reduce) {
  .hadith-card {
    transition: none;
  }
  .hadith-card:hover {
    transform: none;
  }
}
</style>
