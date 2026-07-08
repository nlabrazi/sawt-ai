import { mount } from '@vue/test-utils'

import RecognitionLoadingScreen from '~/components/RecognitionLoadingScreen.vue'

describe('RecognitionLoadingScreen', () => {
  it('marks the matching step as active', () => {
    const wrapper = mount(RecognitionLoadingScreen, {
      props: {
        loading: true,
        step: 'matching',
      },
    })

    expect(wrapper.text()).toContain('Détection')
    expect(wrapper.get('.loading-step.is-active').text()).toContain('Reconnaissance du passage')
  })

  it('marks previous steps as done when finalizing', () => {
    const wrapper = mount(RecognitionLoadingScreen, {
      props: {
        loading: true,
        step: 'done',
      },
    })

    expect(wrapper.text()).toContain('Finalisation')
    expect(wrapper.findAll('.loading-step.is-done')).toHaveLength(2)
    expect(wrapper.get('.loading-step.is-active').text()).toContain('Préparation du résultat')
  })
})
