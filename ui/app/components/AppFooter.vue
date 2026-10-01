<script setup lang="ts">
import { useRuntimeConfig } from '#app'

const contactEmail = String(useRuntimeConfig().public.contactEmail ?? '').trim()
const contactHref = contactEmail
  ? `mailto:${contactEmail}`
  : 'https://github.com/nlabrazi/sawt-ai/issues/new'
</script>

<template>
  <footer class="app-footer">
    <div class="footer-inner">
      <p class="footer-brand">
        © 2026 <a href="/" aria-label="Accueil Sawt AI">Sawt AI</a>
      </p>

      <nav class="footer-links" aria-label="Liens utiles">
        <a href="https://github.com/nlabrazi/sawt-ai" target="_blank" rel="noreferrer">
          GitHub
        </a>
        <a href="https://nabster.dev" target="_blank" rel="noreferrer">
          Portfolio
        </a>
        <a
          :href="contactHref"
          :target="contactEmail ? undefined : '_blank'"
          :rel="contactEmail ? undefined : 'noreferrer'"
        >
          Contact
        </a>
      </nav>
    </div>
  </footer>
</template>

<style scoped>
.app-footer {
  position: relative;
  z-index: 1;
  padding: 0 max(20px, env(safe-area-inset-right, 0px)) calc(12px + env(safe-area-inset-bottom, 0px)) max(20px, env(safe-area-inset-left, 0px));
}

.footer-inner {
  max-width: 980px;
  margin: 0 auto;
  padding: 8px 0 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}

.footer-brand {
  margin: 0;
  font-size: 13px;
  color: #8c9db4;
}

.footer-brand a {
  color: inherit;
  text-decoration: none;
}

.footer-links {
  display: flex;
  flex-wrap: wrap;
  gap: 14px;
}

.footer-links a {
  display: inline-flex;
  align-items: center;
  min-height: 44px;
  color: #a1b0c5;
  text-decoration: none;
  font-size: 12px;
  transition: color 0.2s ease, opacity 0.2s ease;
}

.footer-links a:hover {
  color: #fff;
}

@media (max-width: 640px) {
  .footer-inner {
    justify-content: center;
    flex-wrap: wrap;
    gap: 4px 16px;
  }
}

@media (prefers-reduced-motion: reduce) {
  .footer-links a {
    transition: none;
  }
}
</style>
