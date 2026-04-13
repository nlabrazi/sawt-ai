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
        loading: true,
      },
    })

    expect(wrapper.get('button').attributes('aria-label')).toBe('Arrêter l’enregistrement')
    expect(wrapper.get('button').attributes('aria-pressed')).toBe('true')
    expect(wrapper.get('button').attributes('aria-busy')).toBe('true')
  })
})
