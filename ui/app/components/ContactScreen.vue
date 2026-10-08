<script setup lang="ts">
import { useRuntimeConfig } from '#app'
import { Mail } from '@lucide/vue'
import { $fetch } from 'ofetch'
import { onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import ContactCaptcha from '~/components/ContactCaptcha.vue'
import LegalPageShell from '~/components/LegalPageShell.vue'
import { legalInformation } from '~/content/legal'

// Structure et présentation reprises du composant Contact.vue du portfolio.
const config = useRuntimeConfig()
const contactEmail = String(config.public.contactEmail ?? '').trim() || legalInformation.contactEmail
const accessKey = String(config.public.web3formsAccessKey ?? '').trim()
const form = ref<HTMLFormElement | null>(null)
const formState = reactive({ name: '', email: '', subject: '', message: '' })
const captcha = ref<{ reset: () => void } | null>(null)
const captchaEnabled = ref(false)
const captchaToken = ref('')
const ready = ref(false)
const isSending = ref(false)
const errorMessage = ref('')
const successMessage = ref('')
let requestController: AbortController | null = null

const contactItems = [
  { label: 'E-mail', href: `mailto:${contactEmail}`, value: contactEmail, icon: '' },
  { label: 'LinkedIn', href: 'https://fr.linkedin.com/in/nabil-labrazi', value: 'fr.linkedin.com/in/nabil-labrazi', icon: 'linkedin' },
  { label: 'GitHub', href: 'https://github.com/nlabrazi', value: 'github.com/nlabrazi', icon: 'github' },
  { label: 'X', href: 'https://x.com/Nabil71405502', value: 'x.com/Nabil', icon: 'x-twitter' },
]

async function submitMessage() {
  if (isSending.value || !ready.value || !accessKey || !form.value || !form.value.reportValidity()) return
  captchaEnabled.value = true
  errorMessage.value = ''
  successMessage.value = ''

  if (!Object.values(formState).every((value) => value.trim())) {
    errorMessage.value = 'Veuillez renseigner tous les champs obligatoires.'
    return
  }
  if (!captchaToken.value) {
    errorMessage.value = 'Veuillez compléter la vérification antispam.'
    return
  }
  if (new FormData(form.value).get('botcheck')) return

  isSending.value = true
  requestController = new AbortController()
  try {
    const response = await $fetch<{ success: boolean }>('https://api.web3forms.com/submit', {
      method: 'POST',
      body: {
        access_key: accessKey,
        name: formState.name.trim(),
        email: formState.email.trim(),
        subject: `Sawt AI — ${formState.subject.trim()}`,
        message: formState.message.trim(),
        from_name: 'Sawt AI',
        'h-captcha-response': captchaToken.value,
      },
      retry: 0,
      timeout: 15_000,
      signal: requestController.signal,
    })
    if (response.success !== true) throw new Error('Contact submission rejected')
    Object.assign(formState, { name: '', email: '', subject: '', message: '' })
    successMessage.value = 'Votre message a été transmis.'
  } catch {
    errorMessage.value = 'L’envoi n’a pas pu être confirmé. Veuillez réessayer ultérieurement ou utiliser l’adresse e-mail indiquée sur cette page.'
  } finally {
    isSending.value = false
    requestController = null
    captchaToken.value = ''
    captcha.value?.reset()
  }
}

onMounted(() => { ready.value = true })
onBeforeUnmount(() => requestController?.abort())
</script>

<template>
  <LegalPageShell class="contact-page" title="Contact"
    description="Contacter Sawt-AI pour une question, un problème technique ou une remarque sur les contenus."
    path="/contact" :show-document-details="false">
    <p class="contact-intro">Pour toute question, remarque ou suggestion concernant Sawt-AI.</p>
    <div class="contact-layout">
      <article class="contact-card" aria-labelledby="contact-details-title">
        <h2 id="contact-details-title">Coordonnées</h2>
        <ul class="contact-list">
          <li v-for="item in contactItems" :key="item.label" class="contact-item">
            <span class="contact-icon" aria-hidden="true">
              <img v-if="item.icon" :src="'/assets/icons/' + item.icon + '.svg'" alt="" width="16" height="16" />
              <Mail v-else :size="16" />
            </span>
            <div>
              <p class="contact-label">{{ item.label }}</p>
              <a :href="item.href" :target="item.icon ? '_blank' : undefined"
                :rel="item.icon ? 'noopener noreferrer' : undefined" class="contact-link" dir="ltr">{{ item.value }}</a>
            </div>
          </li>
        </ul>
        <div class="contact-guidance">
          <p>Pour un signalement relatif à un verset ou à un hadith, préciser sa référence facilite le traitement de la
            demande.</p>
        </div>
      </article>

      <article class="contact-card contact-form-panel" aria-labelledby="contact-form-title">
        <h2 id="contact-form-title">Envoyer un message</h2>
        <form ref="form" :aria-busy="isSending" @focusin="captchaEnabled = true" @submit.prevent="submitMessage">
          <fieldset :disabled="isSending || !accessKey || !ready">
            <legend class="sr-only">Envoyer un message</legend>
            <input class="botcheck" type="checkbox" name="botcheck" tabindex="-1" autocomplete="off"
              aria-hidden="true" />
            <div class="identity-fields">
              <div class="form-field">
                <label for="contact-name">Nom complet <span aria-hidden="true">*</span></label>
                <input id="contact-name" v-model="formState.name" name="name" autocomplete="name" maxlength="120"
                  placeholder="Votre nom" required />
              </div>
              <div class="form-field">
                <label for="contact-email">Adresse e-mail <span aria-hidden="true">*</span></label>
                <input id="contact-email" v-model="formState.email" type="email" name="email" autocomplete="email"
                  maxlength="254" placeholder="vous@exemple.com" required />
              </div>
            </div>
            <div class="form-field">
              <label for="contact-subject">Objet <span aria-hidden="true">*</span></label>
              <input id="contact-subject" v-model="formState.subject" name="subject" maxlength="200"
                placeholder="Question / Signalement / Suggestion…" required />
            </div>
            <div class="form-field">
              <label for="contact-message">Message <span aria-hidden="true">*</span></label>
              <textarea id="contact-message" v-model="formState.message" name="message" rows="5" maxlength="5000"
                placeholder="Précisez votre demande…" required />
            </div>
            <ContactCaptcha v-if="captchaEnabled && accessKey" ref="captcha" @verified="captchaToken = $event" />
            <div class="send-actions">
              <button type="submit" aria-describedby="contact-send-help">
                {{ isSending ? 'Envoi en cours…' : 'Envoyer le message' }}
              </button>
              <span id="contact-send-help">Votre message est transmis via Web3Forms.</span>
            </div>
          </fieldset>
          <p v-if="!accessKey" class="form-unavailable">
            Le formulaire est temporairement indisponible. Le contact par e-mail reste accessible.
          </p>
          <noscript>
            <p class="form-unavailable">L’envoi par formulaire nécessite JavaScript. Le contact par e-mail reste
              accessible.</p>
          </noscript>
          <p class="privacy-notice">
            Les champs marqués d’un astérisque sont obligatoires.
            <a href="/legal-notice#privacy">Informations sur la protection des données</a>.
          </p>
          <p v-if="errorMessage" class="form-message is-error" role="alert">{{ errorMessage }}</p>
          <p v-if="successMessage" class="form-message is-success" role="status">{{ successMessage }}</p>
        </form>
      </article>
    </div>
  </LegalPageShell>
</template>

<style scoped>
.contact-page :deep(.legal-header),
.contact-page :deep(.legal-content) {
  width: min(100%, 1280px);
}

.contact-page :deep(h1) {
  position: relative;
  display: inline-block;
}

.contact-page :deep(h1)::after {
  content: '';
  position: absolute;
  left: 0;
  bottom: -7px;
  width: 42px;
  height: 2px;
  border-radius: 999px;
  background: linear-gradient(90deg, #38bdf8, #f472b6);
}

.contact-page .contact-intro {
  margin: 16px 0 32px;
  color: rgba(255, 255, 255, .6);
  font-size: 14px;
}

.contact-layout {
  display: grid;
  gap: 16px;
}

.contact-card {
  min-width: 0;
  padding: 24px;
  border: 1px solid rgba(255, 255, 255, .1);
  border-radius: 16px;
  background: linear-gradient(160deg, rgba(255, 255, 255, .06), rgba(255, 255, 255, .02));
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, .04), 0 0 0 1px rgba(255, 255, 255, .02);
  backdrop-filter: blur(8px);
  transition: box-shadow 220ms ease;
}

@media (hover: hover) {
  .contact-card:hover {
    box-shadow: 0 0 0 1px rgba(255, 255, 255, .12), 0 0 50px rgba(56, 189, 248, .14), 0 0 90px rgba(244, 114, 182, .08);
  }
}

.contact-card h2 {
  margin: 0;
  font-size: 24px;
  font-weight: 600;
}

.contact-form-panel h2 {
  font-size: 18px;
}

.contact-card .contact-list {
  display: grid;
  gap: 20px;
  margin: 24px 0 0;
  padding: 0;
  list-style: none;
}

.contact-card .contact-item {
  display: flex;
  align-items: flex-start;
  gap: 16px;
  margin: 0;
}

.contact-icon {
  display: inline-flex;
  flex: 0 0 auto;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 36px;
  border: 1px solid rgba(255, 255, 255, .14);
  border-radius: 14px;
  background: rgba(255, 255, 255, .06);
  color: rgba(255, 255, 255, .82);
}

.contact-card .contact-label {
  margin: 0;
  color: #94a3b8;
  font-size: 13px;
  line-height: 1.5;
}

.contact-card .contact-link {
  color: #e2e8f0;
  font-size: 15px;
  text-decoration: none;
  transition: color 200ms ease;
}

.contact-card .contact-link:hover {
  color: #7dd3fc;
}

.contact-guidance {
  margin-top: 28px;
  padding-top: 20px;
  border-top: 1px solid rgba(148, 163, 184, .18);
  color: rgba(255, 255, 255, .7);
  font-size: 14px;
}

.contact-guidance p {
  margin: 0;
}

form {
  margin-top: 24px;
}

fieldset {
  min-width: 0;
  margin: 0;
  padding: 0;
  border: 0;
}

.identity-fields {
  display: grid;
  gap: 16px;
}

fieldset>.form-field {
  margin-top: 16px;
}

.form-field label {
  display: block;
  margin-bottom: 8px;
  color: rgba(255, 255, 255, .6);
  font-size: 14px;
}

.form-field input,
.form-field textarea {
  display: block;
  width: 100%;
  min-height: 44px;
  padding: 12px 16px;
  border: 1px solid rgba(255, 255, 255, .1);
  border-radius: 12px;
  background: rgba(255, 255, 255, .05);
  color: #fff;
  font: inherit;
  font-size: 16px;
  line-height: 1.5;
}

.form-field input::placeholder,
.form-field textarea::placeholder {
  color: rgba(255, 255, 255, .4);
}

.form-field textarea {
  resize: vertical;
}

.form-field input:focus-visible,
.form-field textarea:focus-visible,
button:focus-visible {
  outline: 2px solid #93c5fd;
  outline-offset: 3px;
}

.send-actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 12px;
  margin-top: 16px;
}

.send-actions span {
  color: rgba(255, 255, 255, .6);
  font-size: 13px;
}

.send-actions button {
  min-height: 44px;
  padding: 12px 20px;
  border: 0;
  border-radius: 12px;
  background: linear-gradient(135deg, rgba(94, 234, 212, .95), rgba(99, 102, 241, .9));
  box-shadow: 0 10px 25px rgba(56, 189, 248, .22), inset 0 1px 0 rgba(255, 255, 255, .4);
  color: #0b0f1a;
  font: inherit;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
}

.send-actions button:hover {
  filter: brightness(1.05);
}

fieldset:disabled {
  opacity: .65;
}

fieldset:disabled button {
  cursor: not-allowed;
}

.privacy-notice,
.form-unavailable {
  margin-top: 16px;
  color: #a6b8d0;
  font-size: 12px;
  line-height: 1.7;
}

.form-message {
  margin-top: 16px;
  font-size: 14px;
}

.is-error {
  color: #fca5a5;
}

.is-success {
  color: #a7f3d0;
}

.botcheck {
  display: none;
}

@media (min-width: 768px) {
  .identity-fields {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (min-width: 1024px) {
  .contact-layout {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 480px) {
  .contact-card {
    padding: 20px;
  }
}

@media (prefers-reduced-motion: reduce) {

  .contact-card,
  .contact-link {
    transition: none;
  }
}
</style>
