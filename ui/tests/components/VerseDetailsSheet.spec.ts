import { flushPromises, mount } from '@vue/test-utils'
import { $fetch } from 'ofetch'

import VerseDetailsSheet from '~/components/VerseDetailsSheet.vue'
import { clearTajwidCache } from '~/composables/useTajwid'
import { TAJWID_READING_SURFACE_COLOR } from '~/utils/tajwidRules'
import { emptyQuranContent, quranContentFixture } from '../fixtures/quran-content'

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

const ambiguousResult = {
  ...result,
  detection: {
    status: 'ambiguous' as const,
    score: 0.92,
    score_margin: 0.03,
    matched_word_count: 4,
    rejection_reason: 'ambiguous_match' as const,
    analyzed_duration_seconds: 8,
    analysis_attempts: 1,
  },
}

const probableResult = {
  ...result,
  detection: {
    status: 'probable' as const,
    score: 0.74,
    score_margin: 0.1,
    matched_word_count: 4,
    rejection_reason: 'score_too_low' as const,
    analyzed_duration_seconds: 8,
    analysis_attempts: 1,
  },
}

describe('VerseDetailsSheet', () => {
  beforeEach(() => {
    clearTajwidCache()
    vi.mocked($fetch)
      .mockReset()
      .mockResolvedValue(emptyQuranContent(112, 1, 4))
  })

  it('offers a visible return action and icon actions', async () => {
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

    expect(wrapper.get('.close-btn').attributes('aria-label')).toBe('Retour au résultat')
    expect(wrapper.get('.close-btn').text()).toBe('Retour au résultat')
    expect(wrapper.get('.sheet-btn').text()).toBe('Copier')
    expect(wrapper.find('.lucide-arrow-left').exists()).toBe(true)
    expect(wrapper.find('.lucide-copy').exists()).toBe(true)
    expect(wrapper.find('.lucide-book-open').exists()).toBe(true)
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

  it('labels an ambiguous match as a proposal to verify', () => {
    const wrapper = mount(VerseDetailsSheet, {
      props: {
        open: true,
        result: ambiguousResult,
      },
      global: {
        stubs: {
          teleport: true,
        },
      },
    })

    expect(wrapper.get('.sheet').attributes('aria-label')).toBe('Passage proposé')
    expect(wrapper.get('.sheet-kicker').text()).toBe('Passage proposé')

    wrapper.unmount()
  })

  it('labels a probable match as a proposal to verify', () => {
    const wrapper = mount(VerseDetailsSheet, {
      props: {
        open: true,
        result: probableResult,
      },
      global: {
        stubs: {
          teleport: true,
        },
      },
    })

    expect(wrapper.get('.sheet').attributes('aria-label')).toBe('Passage proposé')
    expect(wrapper.get('.sheet-kicker').text()).toBe('Passage proposé')

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
    const tajwidResponse = {
      surah_id: 112,
      start_verse: 1,
      end_verse: 4,
      text: 'وَلَقَ[q:341[دْ] عَهِ[q:8627[دْ]ن[o[َآ]',
      ayahs: [
        { number: 1, tajwid_text: 'وَلَقَ[q:341[دْ]' },
        { number: 2, tajwid_text: 'عَهِ[q:8627[دْ]ن[o[َآ]' },
      ],
    }
    vi.mocked($fetch).mockImplementation(async (url) =>
      String(url).endsWith('/tajwid') ? tajwidResponse : emptyQuranContent(112, 1, 4),
    )
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

  it('displays verse-specific French content and switches between separate tafsir sources', async () => {
    const response = quranContentFixture(112, 1, 4)
    response.ayahs[2] = { ayah: 3, translation: null, tafsirs: [] }
    const draft = {
      ...response.ayahs[3]?.tafsirs[0],
      status: 'need_review',
      text_fr: 'Brouillon fictif privé',
    }
    vi.mocked($fetch).mockResolvedValue({
      ...response,
      ayahs: response.ayahs.map((entry) =>
        entry.ayah === 4 ? { ...entry, tafsirs: [draft] } : entry,
      ),
    })
    const wrapper = mount(VerseDetailsSheet, {
      props: { open: true, result },
      global: { stubs: { teleport: true } },
    })
    await flushPromises()
    expect(wrapper.get('.arabic-verse-text').text()).toBe(result.verse.text)
    const first = wrapper.get('[data-ayah="1"]')
    expect(first.get('.translation-text').text()).toBe('Traduction fictive 112:1.')
    expect(first.get('.source-metadata').text()).toContain('Traducteur de test')
    expect(first.get('a').attributes('href')).toBe('https://example.com/translation/112/1')
    expect(first.get('.translation-notes').text()).toContain('<script>test</script>')
    expect(wrapper.find('script').exists()).toBe(false)
    expect(first.get('.tafsir-text').text()).toBe('Commentaire fictif ibn_kathir 112:1.')
    await first.findAll('.source-button')[1]?.trigger('click')
    expect(first.get('.tafsir-text').text()).toBe('Commentaire fictif as_saadi 112:1.')
    expect(first.text()).not.toContain('Commentaire fictif ibn_kathir')
    expect(wrapper.get('[data-ayah="2"] .tafsir-text').text()).toBe(
      'Commentaire fictif as_saadi 112:2.',
    )
    expect(wrapper.get('[data-ayah="2"] .source-button').attributes('disabled')).toBeDefined()
    expect(wrapper.get('[data-ayah="3"]').text()).toContain(
      'Traduction indisponible pour ce verset.',
    )
    expect(wrapper.find('[data-ayah="3"] .tafsir-section').exists()).toBe(false)
    expect(wrapper.find('[data-ayah="4"] .tafsir-section').exists()).toBe(false)
    expect(wrapper.text()).not.toContain('Brouillon fictif privé')
    wrapper.unmount()
  })

  it('retains the Arabic result on failure and allows retrying the French content', async () => {
    vi.mocked($fetch)
      .mockRejectedValueOnce(new Error('Provider unavailable'))
      .mockResolvedValueOnce(quranContentFixture(112, 1, 4))
    const wrapper = mount(VerseDetailsSheet, {
      props: { open: true, result },
      global: { stubs: { teleport: true } },
    })
    await flushPromises()
    expect(wrapper.get('.arabic-verse-text').text()).toBe(result.verse.text)
    expect(wrapper.text()).toContain('Impossible de charger le contenu français.')
    expect(wrapper.text()).not.toContain('Provider unavailable')
    expect(wrapper.findAll('.sheet-btn').map((button) => button.text())).toContain(
      'Afficher le tajwid',
    )
    await wrapper.get('.content-retry').trigger('click')
    await flushPromises()
    expect(wrapper.get('[data-ayah="1"] .translation-text').text()).toContain('112:1')
    expect(wrapper.find('.content-retry').exists()).toBe(false)
    wrapper.unmount()
  })

  it('loads only while open and rechecks the verified content on reopening or changing passage', async () => {
    vi.mocked($fetch)
      .mockResolvedValueOnce(quranContentFixture(112, 1, 4))
      .mockResolvedValueOnce(emptyQuranContent(112, 1, 4))
      .mockResolvedValueOnce(quranContentFixture(2, 255, 255))
    const wrapper = mount(VerseDetailsSheet, {
      props: { open: false, result },
      global: { stubs: { teleport: true } },
    })
    expect($fetch).not.toHaveBeenCalled()
    await wrapper.setProps({ open: true })
    await flushPromises()
    expect(wrapper.find('.tafsir-text').exists()).toBe(true)
    await wrapper.setProps({ open: false })
    await wrapper.setProps({ open: true })
    await flushPromises()
    expect(wrapper.find('.tafsir-section').exists()).toBe(false)
    await wrapper.setProps({
      result: {
        ...result,
        verse: { ...result.verse, sourate_id: 2, start_verse: 255, end_verse: 255 },
      },
    })
    await flushPromises()
    expect(wrapper.find('[data-ayah="1"]').exists()).toBe(false)
    expect(wrapper.get('[data-ayah="255"] .translation-text').text()).toContain('2:255')
    expect($fetch).toHaveBeenCalledTimes(3)
    wrapper.unmount()
  })
})
