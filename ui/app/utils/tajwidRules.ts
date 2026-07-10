export const TAJWID_READING_SURFACE_COLOR = '#DCE3EA'

export const TAJWID_SOURCE_CODES = [
  'h',
  's',
  'l',
  'n',
  'p',
  'm',
  'q',
  'o',
  'c',
  'f',
  'w',
  'i',
  'a',
  'u',
  'd',
  'b',
  'g',
] as const

export type TajwidSourceCode = (typeof TAJWID_SOURCE_CODES)[number]

export type TajwidRuleCategory =
  | 'silent'
  | 'elongation'
  | 'echo'
  | 'concealment'
  | 'assimilation'
  | 'conversion'
  | 'nasalization'

type TajwidRuleDefinition = {
  sourceCode: TajwidSourceCode
  id: string
  category: TajwidRuleCategory
  label: string
  labelArabic: string
  description: string
  sourceColor: `#${string}`
  displayColor: `#${string}`
}

export const TAJWID_RULES = {
  h: {
    sourceCode: 'h',
    id: 'hamza-wasl',
    category: 'silent',
    label: 'Hamzat al-Wasl',
    labelArabic: 'همزة الوصل',
    description: 'Hamza non prononcée lorsqu’elle est précédée par un autre mot.',
    sourceColor: '#AAAAAA',
    displayColor: '#475569',
  },
  s: {
    sourceCode: 's',
    id: 'silent',
    category: 'silent',
    label: 'Lettre muette',
    labelArabic: 'حرف ساكن',
    description: 'Lettre ou voyelle écrite mais non prononcée.',
    sourceColor: '#AAAAAA',
    displayColor: '#475569',
  },
  l: {
    sourceCode: 'l',
    id: 'lam-shamsiyyah',
    category: 'silent',
    label: 'Lam Shamsiyyah',
    labelArabic: 'لام شمسية',
    description: 'Lam assimilée par la lettre solaire qui la suit.',
    sourceColor: '#AAAAAA',
    displayColor: '#475569',
  },
  n: {
    sourceCode: 'n',
    id: 'madda-normal',
    category: 'elongation',
    label: 'Madd normal',
    labelArabic: 'مد عادي',
    description: 'Prolongation normale de deux temps.',
    sourceColor: '#537FFF',
    displayColor: '#274B9F',
  },
  p: {
    sourceCode: 'p',
    id: 'madda-permissible',
    category: 'elongation',
    label: 'Madd permis',
    labelArabic: 'مد جائز',
    description: 'Prolongation permise de deux, quatre ou six temps.',
    sourceColor: '#4050FF',
    displayColor: '#2D3EB0',
  },
  m: {
    sourceCode: 'm',
    id: 'madda-necessary',
    category: 'elongation',
    label: 'Madd nécessaire',
    labelArabic: 'مد واجب',
    description: 'Prolongation nécessaire de six temps.',
    sourceColor: '#000EBC',
    displayColor: '#172278',
  },
  q: {
    sourceCode: 'q',
    id: 'qalaqah',
    category: 'echo',
    label: 'Qalqalah',
    labelArabic: 'قلقلة',
    description: 'Rebond sonore sur une lettre de qalqalah portant un soukoun.',
    sourceColor: '#DD0008',
    displayColor: '#96212A',
  },
  o: {
    sourceCode: 'o',
    id: 'madda-obligatory',
    category: 'elongation',
    label: 'Madd obligatoire',
    labelArabic: 'مد لازم',
    description: 'Prolongation obligatoire de quatre à cinq temps.',
    sourceColor: '#2144C1',
    displayColor: '#1F398F',
  },
  c: {
    sourceCode: 'c',
    id: 'ikhafa-shafawi',
    category: 'concealment',
    label: 'Ikhfa Shafawi',
    labelArabic: 'إخفاء شفوي',
    description: 'Dissimulation du mīm sākin devant le bāʾ.',
    sourceColor: '#D500B7',
    displayColor: '#861273',
  },
  f: {
    sourceCode: 'f',
    id: 'ikhafa',
    category: 'concealment',
    label: 'Ikhfa',
    labelArabic: 'إخفاء',
    description: 'Dissimulation du nūn sākin ou du tanwīn devant certaines lettres.',
    sourceColor: '#9400A8',
    displayColor: '#681972',
  },
  w: {
    sourceCode: 'w',
    id: 'idgham-shafawi',
    category: 'assimilation',
    label: 'Idgham Shafawi',
    labelArabic: 'إدغام شفوي',
    description: 'Assimilation d’un mīm sākin dans le mīm qui le suit.',
    sourceColor: '#58B800',
    displayColor: '#2B6714',
  },
  i: {
    sourceCode: 'i',
    id: 'iqlab',
    category: 'conversion',
    label: 'Iqlab',
    labelArabic: 'إقلاب',
    description: 'Conversion du nūn sākin ou du tanwīn en mīm devant le bāʾ.',
    sourceColor: '#26BFFD',
    displayColor: '#096A84',
  },
  a: {
    sourceCode: 'a',
    id: 'idgham-with-ghunnah',
    category: 'assimilation',
    label: 'Idgham avec ghunnah',
    labelArabic: 'إدغام بغنة',
    description: 'Assimilation accompagnée d’une résonance nasale.',
    sourceColor: '#169777',
    displayColor: '#0F5F4E',
  },
  u: {
    sourceCode: 'u',
    id: 'idgham-without-ghunnah',
    category: 'assimilation',
    label: 'Idgham sans ghunnah',
    labelArabic: 'إدغام بلا غنة',
    description: 'Assimilation sans résonance nasale.',
    sourceColor: '#169200',
    displayColor: '#1B5F18',
  },
  d: {
    sourceCode: 'd',
    id: 'idgham-mutajanisayn',
    category: 'assimilation',
    label: 'Idgham Mutajanisayn',
    labelArabic: 'إدغام متجانسين',
    description: 'Assimilation de lettres partageant le même point d’articulation.',
    sourceColor: '#A1A1A1',
    displayColor: '#475569',
  },
  b: {
    sourceCode: 'b',
    id: 'idgham-mutaqaribayn',
    category: 'assimilation',
    label: 'Idgham Mutaqaribayn',
    labelArabic: 'إدغام متقاربين',
    description: 'Assimilation de lettres dont les points d’articulation sont proches.',
    sourceColor: '#A1A1A1',
    displayColor: '#475569',
  },
  g: {
    sourceCode: 'g',
    id: 'ghunnah',
    category: 'nasalization',
    label: 'Ghunnah',
    labelArabic: 'غنة',
    description: 'Résonance nasale tenue pendant deux temps.',
    sourceColor: '#FF7E1E',
    displayColor: '#94400A',
  },
} as const satisfies Record<TajwidSourceCode, TajwidRuleDefinition>

export type TajwidRule = (typeof TAJWID_RULES)[TajwidSourceCode]
export type TajwidRuleId = TajwidRule['id']

const tajwidSourceCodeSet = new Set<string>(TAJWID_SOURCE_CODES)

export function isTajwidSourceCode(value: string): value is TajwidSourceCode {
  return tajwidSourceCodeSet.has(value)
}

export function getTajwidRuleBySourceCode(sourceCode: string): TajwidRule | null {
  return isTajwidSourceCode(sourceCode) ? TAJWID_RULES[sourceCode] : null
}
