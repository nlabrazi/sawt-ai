import {
  getTajwidRuleBySourceCode,
  TAJWID_READING_SURFACE_COLOR,
  TAJWID_RULES,
  TAJWID_SOURCE_CODES,
} from '~/utils/tajwidRules'

const expectedRuleIds = {
  h: 'hamza-wasl',
  s: 'silent',
  l: 'lam-shamsiyyah',
  n: 'madda-normal',
  p: 'madda-permissible',
  m: 'madda-necessary',
  q: 'qalaqah',
  o: 'madda-obligatory',
  c: 'ikhafa-shafawi',
  f: 'ikhafa',
  w: 'idgham-shafawi',
  i: 'iqlab',
  a: 'idgham-with-ghunnah',
  u: 'idgham-without-ghunnah',
  d: 'idgham-mutajanisayn',
  b: 'idgham-mutaqaribayn',
  g: 'ghunnah',
}

function relativeLuminance(hexColor: string) {
  const channels = hexColor
    .slice(1)
    .match(/.{2}/g)
    ?.map((value) => Number.parseInt(value, 16) / 255)
    .map((value) => (value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4))

  if (!channels || channels.length !== 3) throw new Error(`Invalid color: ${hexColor}`)

  return 0.2126 * (channels[0] ?? 0) + 0.7152 * (channels[1] ?? 0) + 0.0722 * (channels[2] ?? 0)
}

function contrastRatio(firstColor: string, secondColor: string) {
  const firstLuminance = relativeLuminance(firstColor)
  const secondLuminance = relativeLuminance(secondColor)
  const lightest = Math.max(firstLuminance, secondLuminance)
  const darkest = Math.min(firstLuminance, secondLuminance)

  return (lightest + 0.05) / (darkest + 0.05)
}

describe('tajwidRules', () => {
  it('maps every AlQuran Cloud source code to a unique semantic rule', () => {
    expect(TAJWID_SOURCE_CODES).toHaveLength(17)
    expect(
      Object.fromEntries(TAJWID_SOURCE_CODES.map((code) => [code, TAJWID_RULES[code].id])),
    ).toEqual(expectedRuleIds)
    expect(new Set(TAJWID_SOURCE_CODES.map((code) => TAJWID_RULES[code].id))).toHaveLength(17)
  })

  it('provides complete metadata for every rule', () => {
    for (const sourceCode of TAJWID_SOURCE_CODES) {
      const rule = TAJWID_RULES[sourceCode]

      expect(rule.sourceCode).toBe(sourceCode)
      expect(rule.label).not.toBe('')
      expect(rule.labelArabic).not.toBe('')
      expect(rule.description).not.toBe('')
      expect(rule.sourceColor).toMatch(/^#[0-9A-F]{6}$/)
      expect(rule.displayColor).toMatch(/^#[0-9A-F]{6}$/)
    }
  })

  it('uses display colors with readable contrast on the planned reading surface', () => {
    for (const sourceCode of TAJWID_SOURCE_CODES) {
      expect(
        contrastRatio(TAJWID_RULES[sourceCode].displayColor, TAJWID_READING_SURFACE_COLOR),
      ).toBeGreaterThanOrEqual(4.5)
    }
  })

  it('returns no rule for unsupported source codes', () => {
    expect(getTajwidRuleBySourceCode('q')).toBe(TAJWID_RULES.q)
    expect(getTajwidRuleBySourceCode('q:341')).toBeNull()
    expect(getTajwidRuleBySourceCode('unknown')).toBeNull()
  })
})
