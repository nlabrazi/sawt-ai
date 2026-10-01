import { mount } from '@vue/test-utils'
import App from '~/app.vue'

describe('App', () => {
  it('renders the Quran experience and shared footer', () => {
    const wrapper = mount(App, {
      global: { stubs: { QuranRecognitionScreen: true, AppFooter: true } },
    })
    expect(wrapper.findComponent({ name: 'QuranRecognitionScreen' }).exists()).toBe(true)
    expect(wrapper.findComponent({ name: 'AppFooter' }).exists()).toBe(true)
    wrapper.unmount()
  })
})
