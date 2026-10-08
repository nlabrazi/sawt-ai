import { flushPromises, mount } from '@vue/test-utils'
import { $fetch } from 'ofetch'
import { defineComponent } from 'vue'
import ContactScreen from '~/components/ContactScreen.vue'
import { setRuntimeConfig } from '../mocks/nuxt-app'

vi.mock('ofetch', () => ({ $fetch: vi.fn() }))
const resetCaptcha = vi.fn()
const CaptchaStub = defineComponent({
  name: 'ContactCaptcha',
  emits: ['verified'],
  setup(_props, { expose }) {
    expose({ reset: resetCaptcha })
  },
  template: '<div class="captcha-stub" />',
})

function mountContact() {
  return mount(ContactScreen, { global: { stubs: { ContactCaptcha: CaptchaStub } } })
}

beforeEach(() => {
  vi.mocked($fetch).mockReset()
  setRuntimeConfig({
    public: {
      apiBaseUrl: 'http://localhost:8000',
      contactEmail: 'contact@example.com',
      web3formsAccessKey: 'web3forms-test-key',
    },
  })
})

afterEach(() => {
  setRuntimeConfig({ public: { apiBaseUrl: 'http://localhost:8000' } })
})

async function mountFilledForm() {
  const wrapper = mountContact()
  await wrapper.get('#contact-name').setValue('Prénom Nom')
  await wrapper.get('#contact-email').setValue('person@example.com')
  await wrapper.get('#contact-subject').setValue('Signalement')
  await wrapper.get('#contact-message').setValue('Une remarque sur la référence 2:255.')
  await wrapper.get('form').trigger('focusin')
  wrapper.getComponent(CaptchaStub).vm.$emit('verified', 'test-captcha-token')
  return wrapper
}

it('keeps the portfolio layout and email available while disabling an unconfigured form', () => {
  setRuntimeConfig({
    public: { apiBaseUrl: 'http://localhost:8000', contactEmail: 'contact@example.com' },
  })
  const wrapper = mountContact()
  expect(wrapper.get('fieldset').attributes('disabled')).toBeDefined()
  expect(wrapper.get('.contact-link[href^="mailto:"]').attributes('href')).toBe(
    'mailto:contact@example.com',
  )
  expect($fetch).not.toHaveBeenCalled()
  wrapper.unmount()
})

it('submits once and clears the fields only after Web3Forms confirms success', async () => {
  let resolve!: (response: { success: boolean }) => void
  vi.mocked($fetch).mockReturnValueOnce(
    new Promise((done) => {
      resolve = done
    }),
  )
  const wrapper = await mountFilledForm()
  await wrapper.get('form').trigger('submit')
  await wrapper.get('form').trigger('submit')

  expect($fetch).toHaveBeenCalledTimes(1)
  expect($fetch).toHaveBeenCalledWith('https://api.web3forms.com/submit', {
    method: 'POST',
    body: {
      access_key: 'web3forms-test-key',
      name: 'Prénom Nom',
      email: 'person@example.com',
      subject: 'Sawt AI — Signalement',
      message: 'Une remarque sur la référence 2:255.',
      from_name: 'Sawt AI',
      'h-captcha-response': 'test-captcha-token',
    },
    retry: 0,
    timeout: 15_000,
    signal: expect.any(AbortSignal),
  })
  expect(wrapper.get('fieldset').attributes('disabled')).toBeDefined()
  expect((wrapper.get('#contact-message').element as HTMLTextAreaElement).value).not.toBe('')
  expect(wrapper.find('[role="status"]').exists()).toBe(false)

  resolve({ success: true })
  await flushPromises()
  expect(wrapper.get('[role="status"]').text()).toContain('Votre message a été transmis.')
  expect((wrapper.get('#contact-message').element as HTMLTextAreaElement).value).toBe('')
  expect(wrapper.get('fieldset').attributes('disabled')).toBeUndefined()
  expect(resetCaptcha).toHaveBeenCalledOnce()
  wrapper.unmount()
})

it.each([
  'rejected response',
  'network error',
])('preserves the message after a %s', async (failure) => {
  if (failure === 'network error')
    vi.mocked($fetch).mockRejectedValueOnce(new Error('Network error'))
  else vi.mocked($fetch).mockResolvedValueOnce({ success: false })
  const wrapper = await mountFilledForm()
  await wrapper.get('form').trigger('submit')
  await flushPromises()
  expect(wrapper.get('[role="alert"]').text()).toContain('L’envoi n’a pas pu être confirmé.')
  expect(wrapper.find('[role="status"]').exists()).toBe(false)
  expect((wrapper.get('#contact-message').element as HTMLTextAreaElement).value).toBe(
    'Une remarque sur la référence 2:255.',
  )
  expect(wrapper.get('fieldset').attributes('disabled')).toBeUndefined()
  await wrapper.get('form').trigger('submit')
  expect($fetch).toHaveBeenCalledTimes(1)
  expect(wrapper.get('[role="alert"]').text()).toContain('vérification antispam')
  wrapper.unmount()
})

it('does not send an incomplete form', async () => {
  const wrapper = mountContact()
  await wrapper.get('form').trigger('submit')
  expect($fetch).not.toHaveBeenCalled()
  wrapper.unmount()
})

it('blocks a missing or expired captcha before submitting', async () => {
  const wrapper = await mountFilledForm()
  wrapper.getComponent(CaptchaStub).vm.$emit('verified', '')
  await wrapper.get('form').trigger('submit')
  expect($fetch).not.toHaveBeenCalled()
  expect(wrapper.get('[role="alert"]').text()).toContain('vérification antispam')
  wrapper.unmount()
})
