<script setup lang="ts">
import RecognitionActionButton from '~/components/RecognitionActionButton.vue'

const emit = defineEmits<{
  'micro-click': []
  'select-file': [file: File]
}>()

const fileInput = ref<HTMLInputElement | null>(null)

function openFilePicker() {
  fileInput.value?.click()
}

function onFileChange(event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]

  if (!file) return

  emit('select-file', file)
}
</script>

<template>
  <section class="screen idle-screen">
    <div class="brand">
      <p class="brand-kicker">Sawt AI</p>
    </div>

    <div class="center-stack">
      <RecognitionActionButton @click="emit('micro-click')" />

      <h1 class="main-title">Touchez pour réciter</h1>
      <p class="main-subtitle">
        Détection du verset en quelques secondes
      </p>

      <div class="secondary-action">
        <span class="secondary-text">Vous préférez importer un fichier audio ?</span>
        <button class="secondary-link" type="button" @click="openFilePicker">
          Choisir un fichier
        </button>
      </div>

      <input
        ref="fileInput"
        class="hidden-input"
        type="file"
        accept=".wav,.mp3,.m4a,.ogg"
        @change="onFileChange"
      />
    </div>
  </section>
</template>

<style scoped>
.screen {
  position: relative;
  z-index: 1;
  min-height: 100vh;
  padding: 24px 16px 36px;
  box-sizing: border-box;
}

.idle-screen {
  display: flex;
  flex-direction: column;
}

.brand {
  display: flex;
  justify-content: center;
}

.brand-kicker {
  margin: 0;
  font-size: 13px;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: #93c5fd;
}

.center-stack {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
}

.main-title {
  margin: 28px 0 0;
  font-size: 34px;
  line-height: 1.08;
  font-weight: 800;
  letter-spacing: -0.03em;
}

.main-subtitle {
  margin: 12px 0 0;
  max-width: 420px;
  font-size: 16px;
  line-height: 1.6;
  color: #cbd5e1;
}

.secondary-action {
  margin-top: 34px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
}

.secondary-text {
  color: #94a3b8;
  font-size: 14px;
}

.secondary-link {
  border: none;
  background: transparent;
  color: #93c5fd;
  font-weight: 700;
  font-size: 15px;
  cursor: pointer;
  transition: opacity 0.2s ease, transform 0.2s ease, color 0.2s ease;
}

.secondary-link:hover {
  opacity: 0.9;
  transform: translateY(-1px);
  color: #bfdbfe;
}

.hidden-input {
  display: none;
}

@media (max-width: 768px) {
  .screen {
    padding: 18px 14px 28px;
  }

  .main-title {
    font-size: 28px;
  }
}
</style>
