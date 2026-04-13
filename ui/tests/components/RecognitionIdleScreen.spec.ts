import { mount } from '@vue/test-utils'

import RecognitionIdleScreen from '~/components/RecognitionIdleScreen.vue'

describe('RecognitionIdleScreen', () => {
  it('allows webm uploads and documents the supported format', () => {
    const wrapper = mount(RecognitionIdleScreen, {
      props: {
        detectImam: true,
      },
    })

    const fileInput = wrapper.get('input[type="file"]')

    expect(fileInput.attributes('accept')).toContain('.webm')
    expect(wrapper.text()).toContain('Formats : wav, mp3, m4a, ogg, webm')
  })
})
