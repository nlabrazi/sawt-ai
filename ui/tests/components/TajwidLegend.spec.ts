import { mount } from '@vue/test-utils'

import TajwidLegend from '~/components/TajwidLegend.vue'
import { parseTajwidToTokens } from '~/utils/parseTajwid'
import { TAJWID_RULES } from '~/utils/tajwidRules'

describe('TajwidLegend', () => {
  it('lists each supported rule once in its order of appearance', () => {
    const tokens = parseTajwidToTokens('[q:341[دْ][q:8627[دْ][o[َآ][x:42[ظ]')
    const wrapper = mount(TajwidLegend, {
      props: { tokens },
    })

    expect(wrapper.get('.tajwid-legend-summary').text()).toContain('2 règles')
    expect(wrapper.get('.tajwid-legend').attributes('open')).toBeUndefined()

    const items = wrapper.findAll('.tajwid-legend-item')
    expect(items).toHaveLength(2)
    expect(items[0]?.text()).toContain('Qalqalah')
    expect(items[0]?.text()).toContain('قلقلة')
    expect(items[1]?.text()).toContain('Madd obligatoire')
    expect(wrapper.text()).not.toContain('Règle inconnue')

    expect(
      (items[0]?.get('.tajwid-legend-swatch').element as HTMLElement).style.getPropertyValue(
        '--tajwid-legend-color',
      ),
    ).toBe(TAJWID_RULES.q.displayColor)
    expect(items[0]?.get('.tajwid-legend-name-arabic').attributes('lang')).toBe('ar')
    expect(items[0]?.get('.tajwid-legend-name-arabic').attributes('dir')).toBe('rtl')
  })

  it('uses the native details interaction', async () => {
    const wrapper = mount(TajwidLegend, {
      props: {
        tokens: parseTajwidToTokens('[g[نّ]'),
      },
    })
    const details = wrapper.get('.tajwid-legend').element as HTMLDetailsElement

    expect(details.open).toBe(false)

    await wrapper.get('.tajwid-legend-summary').trigger('click')

    expect(details.open).toBe(true)
  })

  it('stays hidden when the passage contains no supported rule', () => {
    const wrapper = mount(TajwidLegend, {
      props: {
        tokens: parseTajwidToTokens('Texte [x:42[inconnu]'),
      },
    })

    expect(wrapper.find('.tajwid-legend').exists()).toBe(false)
  })
})
