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

    expect(wrapper.find('.eyebrow').exists()).toBe(false)
    expect(wrapper.find('.button-label').exists()).toBe(false)
    expect(wrapper.text()).toContain('Nous comparons le passage aux versets du Coran.')
    expect(wrapper.get('.loading-step.is-active').text()).toContain('Recherche')
    expect(wrapper.get('.loading-step.is-active').attributes('aria-current')).toBe('step')
  })

  it('marks previous steps as done when finalizing', () => {
    const wrapper = mount(RecognitionLoadingScreen, {
      props: {
        loading: true,
        step: 'done',
      },
    })

    expect(wrapper.text()).toContain('Votre résultat est presque prêt.')
    expect(wrapper.findAll('.loading-step.is-done')).toHaveLength(2)
    expect(wrapper.get('.loading-step.is-active').text()).toContain('Résultat')
  })
})
