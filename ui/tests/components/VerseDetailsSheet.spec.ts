import { flushPromises, mount } from '@vue/test-utils'
import { $fetch } from 'ofetch'

import VerseDetailsSheet from '~/components/VerseDetailsSheet.vue'
import { clearTajwidCache } from '~/composables/useTajwid'
import { TAJWID_READING_SURFACE_COLOR } from '~/utils/tajwidRules'

vi.mock('ofetch', () => ({
  $fetch: vi.fn(),
}))

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
  beforeEach(() => {
    clearTajwidCache()
  })

  it('uses icon actions for close and copy', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: { writeText },
    })

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
    expect(wrapper.text()).not.toContain('Texte coranique')
    expect(wrapper.text()).toContain('Transcription brute')
    expect(wrapper.get('.sheet-subtitle').text()).toBe('Sourate Al-Ikhlas · Versets 1 à 4')

    const transcriptionCard = wrapper.get('.transcription-text').element.closest('.content-card')
    const actionCard = wrapper.get('.action-card').element
    expect(
      transcriptionCard?.compareDocumentPosition(actionCard) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy()

    await wrapper.get('.sheet-btn').trigger('click')
    await flushPromises()

    expect(writeText).toHaveBeenCalledWith('الإخلاص — Al-Ikhlas\nVersets 1 à 4\nقل هو الله احد')
    expect(wrapper.get('.mini-toast').text()).toBe('Verset copié')

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

  it('renders fetched tajwid with semantic tokens on the light reading surface', async () => {
    vi.mocked($fetch).mockResolvedValueOnce({
      surah_id: 112,
      start_verse: 1,
      end_verse: 4,
      text: 'وَلَقَ[q:341[دْ] عَهِ[q:8627[دْ]ن[o[َآ]',
      ayahs: [
        { number: 1, tajwid_text: 'وَلَقَ[q:341[دْ]' },
        { number: 2, tajwid_text: 'عَهِ[q:8627[دْ]ن[o[َآ]' },
      ],
    })
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

    const tajwidToggle = wrapper.findAll('.sheet-btn')[1]
    await tajwidToggle?.trigger('click')
    await flushPromises()

    expect($fetch).toHaveBeenCalledWith('http://localhost:8000/tajwid', {
      method: 'GET',
      query: {
        surah_id: 112,
        start_verse: 1,
        end_verse: 4,
      },
    })
    expect(wrapper.get('.tajwid-reading-card').text()).toContain('Affichage tajwid')
    expect(wrapper.get('.tajwid-text').element.textContent).toBe('وَلَقَدْ ۝١ عَهِدْنَآ ۝٢')
    expect(wrapper.findAll('.tajwid-ayah')).toHaveLength(2)
    expect(wrapper.findAll('.tajwid-rule--qalaqah')).toHaveLength(2)
    expect(wrapper.find('.tajwid-rule--madda-obligatory').exists()).toBe(true)
    expect(wrapper.get('.tajwid-legend-summary').text()).toContain('2 règles')
    expect(wrapper.findAll('.tajwid-legend-item')).toHaveLength(2)
    expect(wrapper.get('.tajwid-legend').attributes('open')).toBeUndefined()
    expect(
      (wrapper.get('.sheet').element as HTMLElement).style.getPropertyValue(
        '--tajwid-reading-surface',
      ),
    ).toBe(TAJWID_READING_SURFACE_COLOR)

    await tajwidToggle?.trigger('click')

    expect(wrapper.find('.tajwid-reading-card').exists()).toBe(false)
    wrapper.unmount()
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
