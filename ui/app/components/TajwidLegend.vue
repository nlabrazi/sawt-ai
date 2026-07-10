<script setup lang="ts">
import ChevronDown from '@lucide/vue/dist/esm/icons/chevron-down.mjs'
import Palette from '@lucide/vue/dist/esm/icons/palette.mjs'
import { computed } from 'vue'

import type { TajwidToken } from '~/utils/parseTajwid'
import type { TajwidRuleId } from '~/utils/tajwidRules'
import { getTajwidRuleBySourceCode } from '~/utils/tajwidRules'

const props = defineProps<{
  tokens: TajwidToken[]
}>()

type TajwidLegendItem = {
  id: TajwidRuleId
  label: string
  labelArabic: string
  description: string
  swatchStyle: Record<'--tajwid-legend-color', string>
}

const visibleRules = computed<TajwidLegendItem[]>(() => {
  const seenRuleIds = new Set<TajwidRuleId>()
  const rules: TajwidLegendItem[] = []

  for (const token of props.tokens) {
    if (!token.rule || !token.sourceCode || seenRuleIds.has(token.rule)) continue

    const rule = getTajwidRuleBySourceCode(token.sourceCode)
    if (!rule || rule.id !== token.rule) continue

    seenRuleIds.add(rule.id)
    rules.push({
      id: rule.id,
      label: rule.label,
      labelArabic: rule.labelArabic,
      description: rule.description,
      swatchStyle: { '--tajwid-legend-color': rule.displayColor },
    })
  }

  return rules
})
</script>

<template>
  <details v-if="visibleRules.length" class="tajwid-legend">
    <summary class="tajwid-legend-summary">
      <span class="tajwid-legend-heading">
        <Palette class="tajwid-legend-icon" :stroke-width="2" aria-hidden="true" />
        <span>Légende</span>
        <span class="tajwid-legend-count">
          {{ visibleRules.length }} {{ visibleRules.length > 1 ? 'règles' : 'règle' }}
        </span>
      </span>
      <ChevronDown class="tajwid-legend-chevron" :stroke-width="2" aria-hidden="true" />
    </summary>

    <ul class="tajwid-legend-list">
      <li v-for="rule in visibleRules" :key="rule.id" class="tajwid-legend-item">
        <span class="tajwid-legend-swatch" :style="rule.swatchStyle" aria-hidden="true" />
        <span class="tajwid-legend-copy">
          <span class="tajwid-legend-name-row">
            <span class="tajwid-legend-name">{{ rule.label }}</span>
            <span class="tajwid-legend-name-arabic" lang="ar" dir="rtl">
              {{ rule.labelArabic }}
            </span>
          </span>
          <span class="tajwid-legend-description">{{ rule.description }}</span>
        </span>
      </li>
    </ul>
  </details>
</template>

<style scoped>
.tajwid-legend {
  border-top: 1px solid rgba(166, 141, 84, 0.4);
  padding-top: 6px;
  color: #29251d;
}

.tajwid-legend-summary {
  min-height: 44px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  cursor: pointer;
  list-style: none;
  font-size: 14px;
  font-weight: 800;
  color: #4c422f;
}

.tajwid-legend-summary::-webkit-details-marker {
  display: none;
}

.tajwid-legend-heading {
  min-width: 0;
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.tajwid-legend-count {
  font-size: 12px;
  font-weight: 700;
  color: #75694f;
}

.tajwid-legend-icon,
.tajwid-legend-chevron {
  width: 18px;
  height: 18px;
  flex: 0 0 auto;
}

.tajwid-legend-icon {
  color: #806a35;
}

.tajwid-legend-chevron {
  color: #75694f;
  transition: transform 160ms ease;
}

.tajwid-legend[open] .tajwid-legend-chevron {
  transform: rotate(180deg);
}

.tajwid-legend-list {
  margin: 0;
  padding: 12px 0 4px;
  list-style: none;
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px 24px;
}

.tajwid-legend-item {
  min-width: 0;
  display: grid;
  grid-template-columns: 14px minmax(0, 1fr);
  align-items: start;
  gap: 10px;
}

.tajwid-legend-swatch {
  width: 14px;
  height: 14px;
  margin-top: 3px;
  border-radius: 3px;
  background: var(--tajwid-legend-color);
  box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.12);
}

.tajwid-legend-copy {
  min-width: 0;
  display: grid;
  gap: 3px;
}

.tajwid-legend-name-row {
  display: flex;
  align-items: baseline;
  flex-wrap: wrap;
  gap: 6px;
}

.tajwid-legend-name {
  font-size: 14px;
  font-weight: 800;
  color: #29251d;
}

.tajwid-legend-name-arabic {
  font-family: 'Amiri', serif;
  font-size: 17px;
  line-height: 1;
  color: #4c422f;
}

.tajwid-legend-description {
  font-size: 13px;
  line-height: 1.5;
  color: #75694f;
}

@media (max-width: 768px) {
  .tajwid-legend-list {
    grid-template-columns: 1fr;
    gap: 14px;
  }
}

@media (prefers-reduced-motion: reduce) {
  .tajwid-legend-chevron {
    transition: none;
  }
}
</style>
