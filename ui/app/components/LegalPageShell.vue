<script setup lang="ts">
import { useHead, useRuntimeConfig } from '#app'
import AppFooter from '~/components/AppFooter.vue'
import { legalInformation } from '~/content/legal'

const props = defineProps<{
  title: string
  description: string
  path: '/privacy-policy' | '/terms-of-service'
}>()
const config = useRuntimeConfig()
const contactEmail = String(config.public.contactEmail ?? '').trim()
const siteUrl = String(config.public.siteUrl || 'https://sawt-ai.nabster.dev').replace(/\/$/, '')
const canonicalUrl = `${siteUrl}${props.path}`
const pageTitle = `${props.title} — Sawt AI`

useHead({
  title: pageTitle,
  meta: [
    { name: 'description', content: props.description },
    { property: 'og:title', content: pageTitle },
    { property: 'og:description', content: props.description },
    { property: 'og:url', content: canonicalUrl },
    { name: 'twitter:title', content: pageTitle },
    { name: 'twitter:description', content: props.description },
  ],
  link: [{ rel: 'canonical', href: canonicalUrl }],
})
</script>

<template>
  <div class="legal-page">
    <header class="legal-header">
      <a href="/">← Retour à Sawt AI</a>
      <nav aria-label="Documents légaux">
        <a href="/privacy-policy" :aria-current="path === '/privacy-policy' ? 'page' : undefined">
          Confidentialité
        </a>
        <a href="/terms-of-service" :aria-current="path === '/terms-of-service' ? 'page' : undefined">
          Conditions d’utilisation
        </a>
      </nav>
    </header>

    <main id="legal-content" class="legal-content">
      <h1>{{ title }}</h1>
      <p class="updated-at">
        Dernière mise à jour :
        <time :datetime="legalInformation.updatedAt">{{ legalInformation.updatedAtLabel }}</time>.
      </p>
      <section aria-labelledby="site-information-title">
        <h2 id="site-information-title">Informations sur le site</h2>
        <p v-if="legalInformation.publisher.name">
          Éditeur et responsable des traitements :
          <strong>{{ legalInformation.publisher.name }}</strong>.
        </p>
        <p v-if="legalInformation.publisher.postalAddress" class="postal-address">
          Adresse postale : {{ legalInformation.publisher.postalAddress }}
        </p>
        <p>
          Sawt-AI est hébergé par
          <a href="https://www.ovhcloud.com/fr/personal-data-protection/" target="_blank" rel="noreferrer">
            {{ legalInformation.hosting.provider }}
          </a>, sur des serveurs situés en {{ legalInformation.hosting.country }}.
        </p>
        <p v-if="contactEmail">
          Les demandes relatives au service ou aux données personnelles peuvent être adressées à :
          <a :href="`mailto:${contactEmail}`">{{ contactEmail }}</a>.
        </p>
        <p v-else>L’éditeur peut être contacté au moyen du lien « Contact » figurant en pied de page.</p>
      </section>

      <slot />
    </main>

    <AppFooter />
  </div>
</template>

<style scoped>
.legal-page {
  min-height: 100vh;
  min-height: 100dvh;
  display: flex;
  flex-direction: column;
  background: linear-gradient(180deg, #07111f, #060d17);
}

.legal-header,
.legal-content {
  width: min(100%, 820px);
  margin: 0 auto;
  padding-inline: max(20px, env(safe-area-inset-left, 0px)) max(20px, env(safe-area-inset-right, 0px));
}

.legal-header {
  padding-top: calc(24px + env(safe-area-inset-top, 0px));
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 12px 24px;
}

.legal-header nav {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 20px;
}

.legal-page :deep(a) {
  color: #93c5fd;
  text-underline-offset: 4px;
  overflow-wrap: anywhere;
}

.legal-header a {
  display: inline-flex;
  align-items: center;
  min-height: 44px;
  font-size: 14px;
}

.legal-header a[aria-current='page'] {
  color: #fff;
  font-weight: 600;
}

.legal-page :deep(a:focus-visible) {
  outline: 2px solid #93c5fd;
  outline-offset: 4px;
  border-radius: 2px;
}

.legal-content {
  flex: 1;
  padding-top: 32px;
  padding-bottom: 48px;
  color: #bdcbe0;
  font-size: 15px;
  line-height: 1.8;
  overflow-wrap: anywhere;
}

.legal-content h1 {
  margin: 0;
  color: #f4f8ff;
  font-size: clamp(26px, 5vw, 36px);
  line-height: 1.25;
  letter-spacing: -.025em;
}

.updated-at {
  margin-top: 12px;
  color: #92a6c2;
  font-size: 13px;
}

.legal-content :deep(section) {
  margin-top: 32px;
}

.legal-content :deep(h2) {
  margin: 0 0 12px;
  color: #f4f8ff;
  font-size: 19px;
  line-height: 1.4;
}

.legal-content :deep(p) {
  margin-bottom: 12px;
}

.legal-content :deep(li) {
  margin-bottom: 10px;
}

.legal-content :deep(ul) {
  padding-left: 22px;
}

.postal-address {
  white-space: pre-line;
}
</style>
