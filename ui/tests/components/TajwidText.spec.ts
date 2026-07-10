import { mount } from '@vue/test-utils'

import TajwidText from '~/components/TajwidText.vue'
import { parseTajwidToTokens } from '~/utils/parseTajwid'
import type { TajwidToken } from '~/utils/parseTajwid'
import { TAJWID_RULES, TAJWID_SOURCE_CODES } from '~/utils/tajwidRules'

describe('TajwidText', () => {
  it('renders every supported rule with its semantic class and display color', () => {
    const tokens: TajwidToken[] = TAJWID_SOURCE_CODES.map((sourceCode, index) => ({
      text: String(index + 1),
      rule: TAJWID_RULES[sourceCode].id,
      sourceCode,
      annotationId: index + 1,
    }))
    const wrapper = mount(TajwidText, {
      props: { tokens },
    })

    const renderedRuleTokens = wrapper.findAll('.tajwid-token--rule')

    expect(renderedRuleTokens).toHaveLength(17)

    for (const [index, sourceCode] of TAJWID_SOURCE_CODES.entries()) {
      const renderedToken = renderedRuleTokens[index]
      const rule = TAJWID_RULES[sourceCode]

      expect(renderedToken?.classes()).toContain(`tajwid-rule--${rule.id}`)
      expect(renderedToken?.attributes('data-rule')).toBe(rule.id)
      expect(renderedToken?.attributes('data-source-code')).toBe(sourceCode)
      expect(renderedToken?.attributes('data-annotation-id')).toBe(String(index + 1))
      expect(
        (renderedToken?.element as HTMLElement | undefined)?.style.getPropertyValue(
          '--tajwid-token-color',
        ),
      ).toBe(rule.displayColor)
    }
  })

  it('preserves the exact Arabic text and reading direction', () => {
    const tokens = parseTajwidToTokens('وَلَقَ[q:341[دْ] عَهِ[q:8627[دْ]ن[o[َآ]')
    const wrapper = mount(TajwidText, {
      props: { tokens },
    })
    const text = wrapper.get('.tajwid-text')

    expect(text.element.textContent).toBe('وَلَقَدْ عَهِدْنَآ')
    expect(text.attributes('dir')).toBe('rtl')
    expect(text.attributes('lang')).toBe('ar')
  })

  it('renders raw HTML as text instead of interpreting it', () => {
    const wrapper = mount(TajwidText, {
      props: {
        tokens: [
          {
            text: '<img src=x onerror=alert(1)>',
            rule: null,
            sourceCode: null,
            annotationId: null,
          },
        ],
      },
    })

    expect(wrapper.get('.tajwid-text').element.textContent).toBe('<img src=x onerror=alert(1)>')
    expect(wrapper.find('img').exists()).toBe(false)
  })

  it('keeps unsupported rules visible with a neutral fallback', () => {
    const wrapper = mount(TajwidText, {
      props: {
        tokens: [
          {
            text: 'ظ',
            rule: null,
            sourceCode: 'x',
            annotationId: 42,
          },
        ],
      },
    })
    const unknownToken = wrapper.get('.tajwid-token--unknown')

    expect(unknownToken.text()).toBe('ظ')
    expect(unknownToken.attributes('data-rule')).toBe('unknown')
    expect(unknownToken.attributes('data-source-code')).toBe('x')
    expect(unknownToken.attributes('data-annotation-id')).toBe('42')
    expect(
      (unknownToken.element as HTMLElement).style.getPropertyValue('--tajwid-token-color'),
    ).toBe('')
  })
})
