import { flushPromises, mount } from '@vue/test-utils'

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
    expect(wrapper.find('.lucide-x').exists()).toBe(true)
    expect(wrapper.find('.lucide-copy').exists()).toBe(true)
    expect(wrapper.find('.lucide-book-open').exists()).toBe(true)
    expect(wrapper.text()).not.toContain('Fermer')

    await wrapper.get('.close-btn').trigger('click')

    expect(wrapper.emitted('close')).toHaveLength(1)
    wrapper.unmount()
  })

  it('manages focus and closes with Escape', async () => {
    const trigger = document.createElement('button')
    document.body.append(trigger)
    trigger.focus()

    const wrapper = mount(VerseDetailsSheet, {
      attachTo: document.body,
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

    await flushPromises()

    expect(document.activeElement).toBe(wrapper.get('.close-btn').element)

    await wrapper.get('.sheet').trigger('keydown', { key: 'Escape' })

    expect(wrapper.emitted('close')).toHaveLength(1)

    wrapper.unmount()
    expect(document.activeElement).toBe(trigger)
    trigger.remove()
  })

  it('keeps keyboard focus inside the dialog', async () => {
    const wrapper = mount(VerseDetailsSheet, {
      attachTo: document.body,
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

    await flushPromises()

    const closeButton = wrapper.get('.close-btn')
    const actionButtons = wrapper.findAll('.sheet-btn')
    const lastAction = actionButtons.at(-1)

    lastAction?.element.focus()
    await wrapper.get('.sheet').trigger('keydown', { key: 'Tab' })
    expect(document.activeElement).toBe(closeButton.element)

    closeButton.element.focus()
    await wrapper.get('.sheet').trigger('keydown', { key: 'Tab', shiftKey: true })
    expect(document.activeElement).toBe(lastAction?.element)

    wrapper.unmount()
  })
})
