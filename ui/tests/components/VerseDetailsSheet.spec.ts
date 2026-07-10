import { mount } from '@vue/test-utils'

import VerseDetailsSheet from '~/components/VerseDetailsSheet.vue'

const result = {
  transcription_text: 'قل هو الله احد',
  verse: {
    sourate_id: 112,
    sourate_name: 'الإخلاص',
    transliteration: 'Al-Ikhlas',
    start_verse: 1,
    end_verse: 4,
    text: 'قل هو الله احد',
    similarity: 0.92,
  },
  imam_predictions: [],
  imam_status: 'unknown' as const,
  imam_detection_enabled: true,
}

describe('VerseDetailsSheet', () => {
  it('uses icon actions for close and copy', async () => {
    const wrapper = mount(VerseDetailsSheet, {
      props: {
        open: true,
        result,
      },
      global: {
        stubs: {
          teleport: true,
        },
      },
    })

    expect(wrapper.get('.close-btn').attributes('aria-label')).toBe('Fermer')
    expect(wrapper.get('.sheet-btn').text()).toBe('Copier')
    expect(wrapper.text()).not.toContain('Fermer')

    await wrapper.get('.close-btn').trigger('click')

    expect(wrapper.emitted('close')).toHaveLength(1)
    wrapper.unmount()
  })
})
