import { mount } from '@vue/test-utils'

import AppFooter from '~/components/AppFooter.vue'

describe('AppFooter', () => {
  it('links to the two legal documents and the contact page', () => {
    const wrapper = mount(AppFooter)
    expect(wrapper.findAll('nav a').map((link) => link.attributes('href'))).toEqual([
      '/legal-notice',
      '/terms-of-service',
      '/contact',
    ])
  })
})
