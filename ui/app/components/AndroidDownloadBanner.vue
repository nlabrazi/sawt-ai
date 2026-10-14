<script setup lang="ts">
import { useRuntimeConfig } from '#app'
import { Capacitor } from '@capacitor/core'
import { resolveAndroidDownloadUrl } from '~/utils/androidDownload'

const downloadUrl = resolveAndroidDownloadUrl(useRuntimeConfig().public.androidDownloadUrl)
const showDownload = downloadUrl && Capacitor.getPlatform() !== 'android'
</script>

<template>
  <aside v-if="showDownload" class="android-download" aria-label="Application Android">
    <a class="download-link" :href="downloadUrl ?? undefined" download="sawt-ai-android.apk"
      aria-label="Télécharger l’application Android" aria-describedby="android-beta-label">
      <img class="download-logo" src="/assets/icons/google-play.svg" width="14" height="14" alt="" />
      <span class="download-copy">
        <span class="download-caption">Télécharger l’application</span>
        <span class="download-platform">Android<sup aria-hidden="true">™</sup></span>
      </span>
    </a>
    <span id="android-beta-label" class="beta-label">Bêta</span>
  </aside>
</template>

<style scoped>
@font-face {
  font-family: 'Roboto';
  font-style: normal;
  font-weight: 400 500;
  font-display: swap;
  src: url('/fonts/Roboto-Latin.woff2') format('woff2');
}

.android-download {
  width: min(100%, 1120px);
  margin: 0 auto;
  padding: 12px max(20px, env(safe-area-inset-right, 0px)) 0 max(20px, env(safe-area-inset-left, 0px));
  box-sizing: border-box;
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 8px;
}

.download-link {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 4px;
  width: 128px;
  min-height: 30px;
  max-width: 100%;
  padding: 1px 6px;
  box-sizing: border-box;
  border: 1px solid #a6a6a6;
  border-radius: 5px;
  background: #000;
  color: #fff;
  font-family: Roboto, Arial, sans-serif;
  text-decoration: none;
  text-align: left;
  transition: border-color 160ms ease, background-color 160ms ease;
}

.download-link:hover {
  border-color: #fff;
  background: #111;
}

.download-link:focus-visible {
  outline: 2px solid #67e8f9;
  outline-offset: 4px;
}

.download-logo {
  flex-shrink: 0;
}

.download-copy {
  display: flex;
  flex-direction: column;
  gap: 1px;
}

.download-caption {
  font-size: 8px;
  font-weight: 400;
  line-height: 1.35;
  white-space: nowrap;
}

.download-platform {
  font-size: 13px;
  font-weight: 500;
  line-height: 1.1;
  letter-spacing: -.25px;
}

.download-platform sup {
  margin-left: 2px;
  font-size: 6px;
  letter-spacing: 0;
  vertical-align: super;
}

.beta-label {
  flex-shrink: 0;
  padding: 3px 6px;
  border: 1px solid #536178;
  border-radius: 999px;
  color: #d7e1ee;
  font-size: 10px;
  font-weight: 500;
  line-height: 1.2;
}

@media (max-width: 420px) {
  .android-download {
    justify-content: center;
    padding-inline: 16px;
  }
}

@media (prefers-reduced-motion: reduce) {
  .download-link {
    transition: none;
  }
}
</style>
