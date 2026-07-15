import { mount } from '@vue/test-utils'

import RecognitionActionButton from '~/components/RecognitionActionButton.vue'

describe('RecognitionActionButton', () => {
  it('emits a click event when pressed', async () => {
    const wrapper = mount(RecognitionActionButton)

    await wrapper.get('button').trigger('click')

    expect(wrapper.emitted('click')).toHaveLength(1)
  })

  it('forwards the disabled state to the native button', () => {
    const wrapper = mount(RecognitionActionButton, {
      props: {
        disabled: true,
      },
    })

    expect(wrapper.get('button').attributes('disabled')).toBeDefined()
  })

  it('exposes an accessible label and recording state', () => {
    const wrapper = mount(RecognitionActionButton, {
      props: {
        isRecording: true,
      },
    })

    expect(wrapper.get('button').attributes('aria-label')).toBe('Arrêter et analyser')
    expect(wrapper.get('button').attributes('aria-pressed')).toBe('true')
    expect(wrapper.get('button').attributes('aria-busy')).toBe('false')
    expect(wrapper.get('.button-label').text()).toBe('Arrêter et analyser')
  })

  it('makes the non-interactive analysis state explicit', () => {
    const wrapper = mount(RecognitionActionButton, {
      props: {
        loading: true,
      },
    })

    expect(wrapper.get('button').attributes('aria-label')).toBe('Analyse en cours')
    expect(wrapper.get('button').attributes('aria-busy')).toBe('true')
    expect(wrapper.get('button').attributes('disabled')).toBeDefined()
    expect(wrapper.get('.button-label').text()).toBe('Analyse en cours')
    expect(wrapper.find('.lucide-loader-circle').exists()).toBe(true)
  })

  it('uses Lucide icons for microphone and stop states', async () => {
    const wrapper = mount(RecognitionActionButton)

    expect(wrapper.find('.lucide-mic').exists()).toBe(true)
    expect(wrapper.find('.lucide-square').exists()).toBe(false)

    await wrapper.setProps({ isRecording: true })

    expect(wrapper.find('.lucide-mic').exists()).toBe(false)
    expect(wrapper.find('.lucide-square').exists()).toBe(true)
  })
})
