import { parseTajwidToHtml, parseTajwidToTokens } from '~/utils/parseTajwid'
import { TAJWID_RULES, TAJWID_SOURCE_CODES } from '~/utils/tajwidRules'

describe('parseTajwidToTokens', () => {
  it('parses every supported source code into a semantic token', () => {
    const fragments = [
      'ٱ',
      'و',
      'ل',
      'ا',
      'ي',
      'آ',
      'دْ',
      'ىٰٓ',
      'مْ',
      'نْ',
      'م',
      'ن',
      'ي',
      'ل',
      'ت',
      'ر',
      'نّ',
    ]
    const rawText = TAJWID_SOURCE_CODES.map(
      (sourceCode, index) => `[${sourceCode}:${index + 1}[${fragments[index]}]`,
    ).join(' ')

    const tokens = parseTajwidToTokens(rawText)
    const ruleTokens = tokens.filter((token) => token.sourceCode !== null)

    expect(ruleTokens).toHaveLength(17)
    expect(
      ruleTokens.map(({ sourceCode, rule, annotationId }) => ({
        sourceCode,
        rule,
        annotationId,
      })),
    ).toEqual(
      TAJWID_SOURCE_CODES.map((sourceCode, index) => ({
        sourceCode,
        rule: TAJWID_RULES[sourceCode].id,
        annotationId: index + 1,
      })),
    )
    expect(tokens.map((token) => token.text).join('')).toBe(fragments.join(' '))
  })

  it('separates annotation identifiers from their source rule', () => {
    const tokens = parseTajwidToTokens('وَلَقَ[q:341[دْ] عَهِ[q:8627[دْ]ن[o[َآ]')

    expect(tokens).toEqual([
      { text: 'وَلَقَ', rule: null, sourceCode: null, annotationId: null },
      { text: 'دْ', rule: 'qalaqah', sourceCode: 'q', annotationId: 341 },
      { text: ' عَهِ', rule: null, sourceCode: null, annotationId: null },
      { text: 'دْ', rule: 'qalaqah', sourceCode: 'q', annotationId: 8627 },
      { text: 'ن', rule: null, sourceCode: null, annotationId: null },
      { text: 'َآ', rule: 'madda-obligatory', sourceCode: 'o', annotationId: null },
    ])
  })

  it('preserves unknown annotations with a neutral semantic rule', () => {
    expect(parseTajwidToTokens('ق[x:42[ل]')).toEqual([
      { text: 'ق', rule: null, sourceCode: null, annotationId: null },
      { text: 'ل', rule: null, sourceCode: 'x', annotationId: 42 },
    ])
  })

  it('preserves malformed annotations as plain text', () => {
    const rawText = 'قبل [q:abc[د] وبعد [q:12[ناقص'
    const tokens = parseTajwidToTokens(rawText)

    expect(tokens).toEqual([{ text: rawText, rule: null, sourceCode: null, annotationId: null }])
  })

  it('keeps raw HTML as inert token text', () => {
    expect(parseTajwidToTokens('<script>alert(1)</script>')).toEqual([
      {
        text: '<script>alert(1)</script>',
        rule: null,
        sourceCode: null,
        annotationId: null,
      },
    ])
  })

  it('merges adjacent tokens only when their metadata is identical', () => {
    expect(parseTajwidToTokens('[n[ا][n[ب][n:2[ج]')).toEqual([
      { text: 'اب', rule: 'madda-normal', sourceCode: 'n', annotationId: null },
      { text: 'ج', rule: 'madda-normal', sourceCode: 'n', annotationId: 2 },
    ])
  })
})

describe('parseTajwidToHtml', () => {
  it('wraps tagged segments with a sanitized CSS class', () => {
    const html = parseTajwidToHtml('[ghunnah[الرَّحْمَٰنِ]]')

    expect(html).toContain('tajwid-rule-ghunnah')
    expect(html).toContain('data-rule="ghunnah"')
    expect(html).toContain('الرَّحْمَٰنِ')
  })

  it('escapes raw HTML outside tajwid tags', () => {
    const html = parseTajwidToHtml('<script>alert(1)</script>')

    expect(html).toBe('&lt;script&gt;alert(1)&lt;/script&gt;')
  })

  it('supports nested tajwid tags', () => {
    const html = parseTajwidToHtml('[outer[abc [inner[def]] ghi]]')

    expect(html).toContain('tajwid-rule-outer')
    expect(html).toContain('tajwid-rule-inner')
  })
})
