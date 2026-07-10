<script setup lang="ts">
import { computed } from 'vue'

import type { TajwidToken } from '~/utils/parseTajwid'
import { getTajwidRuleBySourceCode } from '~/utils/tajwidRules'

const props = defineProps<{
  tokens: TajwidToken[]
}>()

const renderedTokens = computed(() => {
  return props.tokens.map((token, index) => {
    const ruleDefinition = token.sourceCode ? getTajwidRuleBySourceCode(token.sourceCode) : null
    const stateClass = token.rule
      ? 'tajwid-token--rule'
      : token.sourceCode
        ? 'tajwid-token--unknown'
        : 'tajwid-token--plain'

    return {
      ...token,
      key: `${index}-${token.sourceCode ?? 'plain'}-${token.annotationId ?? 'none'}`,
      className: [stateClass, token.rule ? `tajwid-rule--${token.rule}` : null],
      style: ruleDefinition ? { '--tajwid-token-color': ruleDefinition.displayColor } : undefined,
    }
  })
})
</script>

<template>
  <p class="tajwid-text" dir="rtl" lang="ar">
    <span
      v-for="token in renderedTokens"
      :key="token.key"
      class="tajwid-token"
      :class="token.className"
      :style="token.style"
      :data-rule="token.rule ?? (token.sourceCode ? 'unknown' : undefined)"
      :data-source-code="token.sourceCode ?? undefined"
      :data-annotation-id="token.annotationId ?? undefined"
    >{{ token.text }}</span>
  </p>
</template>

<style scoped>
.tajwid-text {
  max-width: 100%;
  margin: 0;
  color: #172033;
  font-family: 'Amiri', serif;
  font-size: 34px;
  line-height: 1.95;
  letter-spacing: 0;
  direction: rtl;
  text-align: right;
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

@media (max-width: 768px) {
  .tajwid-text {
    font-size: 30px;
    line-height: 1.9;
  }
}

@media (max-width: 380px) {
  .tajwid-text {
    font-size: 28px;
  }
}
</style>
