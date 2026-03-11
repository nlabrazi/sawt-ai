<script setup lang="ts">
import RecognitionActionButton from '~/components/RecognitionActionButton.vue'
import RecognitionLoader from '~/components/RecognitionLoader.vue'

defineProps<{
  loading: boolean
}>()

defineEmits<{
  cancel: []
}>()
</script>

<template>
  <section class="screen loading-screen">
    <div class="top-bar">
      <button class="top-action" type="button" @click="$emit('cancel')">
        Annuler
      </button>
    </div>

    <div class="center-stack compact-center">
      <RecognitionActionButton disabled loading />

      <h2 class="main-title loading-title">Détection en cours</h2>
      <p class="main-subtitle">
        Analyse de la récitation et recherche du verset
      </p>

      <div class="loader-shell">
        <RecognitionLoader :loading="loading" />
      </div>
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

.loading-screen {
  display: flex;
  flex-direction: column;
}

.top-bar {
  display: flex;
  justify-content: flex-end;
}

.top-action {
  border: 1px solid rgba(255, 255, 255, 0.12);
  background: rgba(15, 23, 42, 0.6);
  color: #e2e8f0;
  border-radius: 999px;
  padding: 10px 14px;
  cursor: pointer;
  font-weight: 600;
  backdrop-filter: blur(10px);
  transition: transform 0.2s ease, background 0.2s ease, border-color 0.2s ease;
}

.top-action:hover {
  transform: translateY(-1px);
  background: rgba(30, 41, 59, 0.78);
  border-color: rgba(147, 197, 253, 0.28);
}

.center-stack {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
}

.compact-center {
  max-width: 760px;
  margin: 0 auto;
}

.main-title {
  margin: 28px 0 0;
  font-size: 34px;
  line-height: 1.08;
  font-weight: 800;
  letter-spacing: -0.03em;
}

.loading-title {
  margin-top: 22px;
}

.main-subtitle {
  margin: 12px 0 0;
  max-width: 420px;
  font-size: 16px;
  line-height: 1.6;
  color: #cbd5e1;
}

.loader-shell {
  width: 100%;
  max-width: 760px;
  margin-top: 28px;
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
