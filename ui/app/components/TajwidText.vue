<script setup lang="ts">
import { computed } from 'vue'

import type { TajwidToken } from '~/utils/parseTajwid'
import { formatAyahNumber } from '~/utils/formatAyahNumber'
import { getTajwidRuleBySourceCode } from '~/utils/tajwidRules'

type RenderableTajwidAyah = {
  number: number
  tokens: TajwidToken[]
}

const props = defineProps<{
  ayahs: RenderableTajwidAyah[]
}>()

const renderedAyahs = computed(() => {
  return props.ayahs.map((ayah) => ({
    ...ayah,
    formattedNumber: formatAyahNumber(ayah.number),
    tokens: ayah.tokens.map((token, index) => {
      const ruleDefinition = token.sourceCode ? getTajwidRuleBySourceCode(token.sourceCode) : null
      const stateClass = token.rule
        ? 'tajwid-token--rule'
        : token.sourceCode
          ? 'tajwid-token--unknown'
          : 'tajwid-token--plain'

      return {
        ...token,
        key: `${ayah.number}-${index}-${token.sourceCode ?? 'plain'}-${token.annotationId ?? 'none'}`,
        className: [stateClass, token.rule ? `tajwid-rule--${token.rule}` : null],
        style: ruleDefinition ? { '--tajwid-token-color': ruleDefinition.displayColor } : undefined,
      }
    }),
  }))
})
</script>

<template>
  <p class="tajwid-text" dir="rtl" lang="ar">
    <span v-for="(ayah, ayahIndex) in renderedAyahs" :key="ayah.number" class="tajwid-ayah">
      <span v-if="ayahIndex > 0">{{ ' ' }}</span>
      <span
        v-for="token in ayah.tokens"
        :key="token.key"
        class="tajwid-token"
        :class="token.className"
        :style="token.style"
        :data-rule="token.rule ?? (token.sourceCode ? 'unknown' : undefined)"
        :data-source-code="token.sourceCode ?? undefined"
        :data-annotation-id="token.annotationId ?? undefined"
      >{{ token.text }}</span>
      <span class="ayah-marker-separator">{{ ' ' }}</span>
      <span
        class="ayah-marker"
        :aria-label="`نهاية الآية ${ayah.number}`"
      >۝{{ ayah.formattedNumber }}</span>
    </span>
  </p>
</template>

<style scoped>
.tajwid-text {
  max-width: 100%;
  margin: 0;
  color: #29251d;
  font-family: 'Amiri Quran', 'Amiri', serif;
  font-size: 36px;
  line-height: 2.15;
  letter-spacing: 0;
  direction: rtl;
  text-align: center;
  unicode-bidi: plaintext;
  white-space: pre-wrap;
  overflow-wrap: break-word;
}

.tajwid-token {
  color: inherit;
}

.tajwid-token--rule {
  color: var(--tajwid-token-color);
}

.ayah-marker {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 1.75em;
  color: #806a35;
  font-size: 0.72em;
  font-weight: 700;
  line-height: 1;
  vertical-align: 0.08em;
  white-space: nowrap;
}

@media (max-width: 768px) {
  .tajwid-text {
    font-size: 31px;
    line-height: 2.05;
  }
}

@media (max-width: 380px) {
  .tajwid-text {
    font-size: 28px;
  }
}
</style>
