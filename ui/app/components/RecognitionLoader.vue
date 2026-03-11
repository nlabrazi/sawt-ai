<script setup lang="ts">
// ROLE
// ----
// Loader centré pour l'état d'analyse.
// Inspiré d'un flow type Shazam : peu de texte, étapes lisibles.

defineProps<{
  loading: boolean
}>()
</script>

<template>
  <div v-if="loading" class="loader-card">
    <div class="steps">
      <div class="step active">
        <span class="step-icon">
          <span class="spinner" />
        </span>
        <span class="step-label">Détection du verset en cours</span>
      </div>

      <div class="step muted">
        <span class="step-icon pending">○</span>
        <span class="step-label">Sourate détectée ensuite</span>
      </div>

      <div class="step muted">
        <span class="step-icon pending">○</span>
        <span class="step-label">Détection imam bientôt disponible</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.loader-card {
  width: 100%;
  border: 1px solid rgba(148, 163, 184, 0.14);
  border-radius: 28px;
  padding: 22px;
  background: rgba(15, 23, 42, 0.46);
  backdrop-filter: blur(14px);
  box-sizing: border-box;
}

.steps {
  display: grid;
  gap: 12px;
}

.step {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 14px 16px;
  border-radius: 18px;
  border: 1px solid rgba(148, 163, 184, 0.1);
  background: rgba(2, 6, 23, 0.36);
}

.step.active {
  border-color: rgba(96, 165, 250, 0.24);
}

.step.muted {
  opacity: 0.72;
}

.step-icon {
  width: 20px;
  height: 20px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.pending {
  color: #64748b;
}

.step-label {
  color: #e2e8f0;
  font-size: 15px;
}

.spinner {
  width: 18px;
  height: 18px;
  border-radius: 999px;
  border: 2px solid rgba(255, 255, 255, 0.16);
  border-top-color: #60a5fa;
  animation: spin 0.9s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}
</style>
