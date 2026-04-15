// ROLE
// ----
// Transforme le texte brut tajwid en HTML affichable.
// V1 : nettoie les marqueurs techniques et garde un arabe propre.
// Le parseur supporte les motifs du type : [rule[text]]

function escapeHtml(value: string) {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;')
}

function sanitizeRuleName(rule: string) {
  return rule
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_-]/g, '-')
}

function readTaggedSegment(
  input: string,
  startIndex: number,
): {
  html: string
  nextIndex: number
} | null {
  if (input[startIndex] !== '[') return null

  const innerOpenIndex = input.indexOf('[', startIndex + 1)
  if (innerOpenIndex === -1) return null

  const rule = input.slice(startIndex + 1, innerOpenIndex).trim()
  if (!rule) return null

  let cursor = innerOpenIndex + 1
  let depth = 1
  let content = ''

  while (cursor < input.length) {
    const char = input[cursor]

    if (char === '[') {
      depth += 1
      content += char
      cursor += 1
      continue
    }

    if (char === ']') {
      depth -= 1

      if (depth === 0) {
        const parsedInner = parseTajwidToHtml(content)
        const safeRule = sanitizeRuleName(rule)

        return {
          html: `<span class="tajwid-fragment tajwid-rule-${safeRule}" data-rule="${escapeHtml(rule)}">${parsedInner}</span>`,
          nextIndex: cursor + 1,
        }
      }

      content += char
      cursor += 1
      continue
    }

    content += char
    cursor += 1
  }

  return null
}

export function parseTajwidToHtml(rawText: string) {
  if (!rawText) return ''

  let html = ''
  let index = 0

  while (index < rawText.length) {
    const char = rawText[index] ?? ''

    if (char === '[') {
      const parsed = readTaggedSegment(rawText, index)

      if (parsed) {
        html += parsed.html
        index = parsed.nextIndex
        continue
      }
    }

    html += escapeHtml(char)
    index += 1
  }

  // nettoyage léger des reliquats éventuels si l'API renvoie
  // encore des marqueurs bruts non parsés
  html = html
    .replace(/\[[^[\]]+:\d+\]/g, '')
    .replace(/\[[^[\]]+\]/g, '')
    .replace(/\s{2,}/g, ' ')
    .trim()

  return html
}
