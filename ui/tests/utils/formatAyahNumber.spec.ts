import { formatAyahNumber } from '~/utils/formatAyahNumber'

describe('formatAyahNumber', () => {
  it.each([
    [1, '١'],
    [10, '١٠'],
    [255, '٢٥٥'],
  ])('formats ayah number %i with Arabic-Indic digits', (ayahNumber, expected) => {
    expect(formatAyahNumber(ayahNumber)).toBe(expected)
  })
})
