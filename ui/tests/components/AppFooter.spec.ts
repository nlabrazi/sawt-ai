import { mount } from '@vue/test-utils'

import AppFooter from '~/components/AppFooter.vue'
import { setRuntimeConfig } from '../mocks/nuxt-app'

describe('AppFooter', () => {
  afterEach(() => {
    setRuntimeConfig({
      public: {
        apiBaseUrl: 'http://localhost:8000',
        contactEmail: '',
      },
    })
  })

  it('uses the configured branded contact email', () => {
    setRuntimeConfig({
      public: {
        apiBaseUrl: 'http://localhost:8000',
        contactEmail: 'contact@sawt-ai.example',
      },
    })

    const wrapper = mount(AppFooter)
    const contactLink = wrapper.get('a[href^="mailto:"]')

    expect(contactLink.attributes('href')).toBe('mailto:contact@sawt-ai.example')
    expect(wrapper.html()).not.toContain('gmail.com')
  })

  it('falls back to the project issue form without exposing a personal address', () => {
    const wrapper = mount(AppFooter)
    const contactLink = wrapper.get('a[href*="issues/new"]')

    expect(contactLink.attributes('target')).toBe('_blank')
    expect(wrapper.html()).not.toContain('gmail.com')
  })
})
