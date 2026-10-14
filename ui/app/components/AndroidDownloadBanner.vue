<script setup lang="ts">
import { useRuntimeConfig } from '#app'
import { Capacitor } from '@capacitor/core'
import { Download } from '@lucide/vue'
import { resolveAndroidDownloadUrl } from '~/utils/androidDownload'

const downloadUrl = resolveAndroidDownloadUrl(useRuntimeConfig().public.androidDownloadUrl)
const showDownload = downloadUrl && Capacitor.getPlatform() !== 'android'
</script>

<template>
  <aside v-if="showDownload" class="android-download" aria-label="Application Android">
    <a class="download-link" :href="downloadUrl ?? undefined" download="sawt-ai-android.apk">
      <Download :size="16" aria-hidden="true" />
      Télécharger pour Android <span class="beta-label">Bêta</span>
    </a>
    <details class="install-help">
      <summary>Comment installer ?</summary>
      <p>Téléchargez l’APK, ouvrez-le puis autorisez l’installation depuis votre navigateur si Android le demande. Les prochaines versions seront disponibles ici.</p>
    </details>
  </aside>
</template>

<style scoped>
.android-download {
  width: min(100%, 1120px);
  margin: 0 auto;
  padding: 12px max(20px, env(safe-area-inset-right, 0px)) 0 max(20px, env(safe-area-inset-left, 0px));
  box-sizing: border-box;
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 6px;
}
.download-link {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  min-height: 44px;
  padding: 0 14px;
  box-sizing: border-box;
  border: 1px solid #2b6173;
  border-radius: 12px;
  background: #0b2635;
  color: #d6faff;
  font-size: 13px;
  font-weight: 600;
  text-decoration: none;
}
.download-link:hover { background: #12394b; }
.download-link:focus-visible, summary:focus-visible {
  outline: 2px solid #67e8f9;
  outline-offset: 4px;
}
.beta-label { font-size: 10px; color: #8bbcc8; }
.install-help { max-width: 360px; font-size: 12px; color: #aebed3; }
summary { cursor: pointer; text-align: right; }
.install-help p { line-height: 1.6; margin: 8px 0 0; }
@media (max-width: 420px) {
  .android-download { align-items: center; padding-inline: 16px; }
  summary { text-align: center; }
}
</style>
