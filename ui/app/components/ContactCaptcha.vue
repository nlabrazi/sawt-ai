<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from 'vue'
import { type CaptchaApi, loadCaptcha } from '~/utils/hcaptcha'

const emit = defineEmits<{ verified: [token: string] }>()
const container = ref<HTMLElement | null>(null)
const failed = ref(false)
let api: CaptchaApi | undefined
let widget: string | undefined
let mounted = false

function reset() {
  emit('verified', '')
  if (widget !== undefined) api?.reset(widget)
}

async function start() {
  failed.value = false
  try {
    api = await loadCaptcha()
    if (!mounted || !container.value) return
    if (widget !== undefined) api.remove(widget)
    emit('verified', '')
    widget = api.render(container.value, {
      // Clé de site partagée officielle pour Web3Forms Free.
      sitekey: '50b2fe65-b00b-4b9e-ad62-3ba471098be2',
      theme: 'dark',
      size: 'compact',
      hl: 'fr',
      callback: (token) => {
        failed.value = false
        emit('verified', token)
      },
      'expired-callback': () => emit('verified', ''),
      'error-callback': () => {
        failed.value = true
        emit('verified', '')
      },
    })
  } catch {
    if (mounted) failed.value = true
  }
}

onMounted(() => {
  mounted = true
  void start()
})
onBeforeUnmount(() => {
  mounted = false
  if (widget !== undefined) api?.remove(widget)
})
defineExpose({ reset })
</script>

<template>
  <div class="captcha-wrapper">
    <div ref="container" class="contact-captcha" />
    <div v-if="failed" class="captcha-error" role="alert">
      <p>La vérification antispam est indisponible. Le contact par e-mail reste accessible.</p>
      <button type="button" @click="start">Réessayer</button>
    </div>
  </div>
</template>

<style scoped>
.captcha-wrapper {
  margin-top: 18px;
}

.captcha-error {
  margin-top: 12px;
  color: #c4cfdf;
  font-size: 14px;
}

.captcha-error p {
  margin: 0 0 10px;
}

.captcha-error button {
  min-height: 44px;
  padding: 8px 16px;
  border: 1px solid rgba(255, 255, 255, .18);
  border-radius: 8px;
  background: rgba(255, 255, 255, .05);
  color: inherit;
  font: inherit;
  cursor: pointer;
}

.captcha-error button:focus-visible {
  outline: 2px solid #93c5fd;
  outline-offset: 3px;
}
</style>
