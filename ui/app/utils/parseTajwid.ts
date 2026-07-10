import type { TajwidRuleId } from '~/utils/tajwidRules'
import { getTajwidRuleBySourceCode } from '~/utils/tajwidRules'

export type TajwidToken = {
  text: string
  rule: TajwidRuleId | null
  sourceCode: string | null
  annotationId: number | null
}

type ParsedTajwidAnnotation = {
  token: TajwidToken
  nextIndex: number
}

const TAJWID_MARKER_PATTERN = /^([a-z]+)(?::([0-9]+))?$/i
const MAX_TAJWID_MARKER_LENGTH = 32

function readTajwidAnnotation(input: string, startIndex: number): ParsedTajwidAnnotation | null {
  if (input[startIndex] !== '[') return null

  const markerEndIndex = input.indexOf('[', startIndex + 1)
  if (markerEndIndex === -1 || markerEndIndex - startIndex > MAX_TAJWID_MARKER_LENGTH) {
    return null
  }

  const closingBeforeMarker = input.indexOf(']', startIndex + 1)
  if (closingBeforeMarker !== -1 && closingBeforeMarker < markerEndIndex) return null

  const marker = input.slice(startIndex + 1, markerEndIndex)
  const markerMatch = TAJWID_MARKER_PATTERN.exec(marker)
  if (!markerMatch) return null

  const contentEndIndex = input.indexOf(']', markerEndIndex + 1)
  if (contentEndIndex === -1) return null

  const sourceCode = (markerMatch[1] ?? '').toLowerCase()
  const rawAnnotationId = markerMatch[2]
  const annotationId = rawAnnotationId ? Number.parseInt(rawAnnotationId, 10) : null

  if (annotationId !== null && !Number.isSafeInteger(annotationId)) return null

  return {
    token: {
      text: input.slice(markerEndIndex + 1, contentEndIndex),
      rule: getTajwidRuleBySourceCode(sourceCode)?.id ?? null,
      sourceCode,
      annotationId,
    },
    nextIndex: contentEndIndex + 1,
  }
}

function appendTajwidToken(tokens: TajwidToken[], token: TajwidToken) {
  if (!token.text) return

  const previousToken = tokens.at(-1)
  const canMergeWithPrevious =
    previousToken?.rule === token.rule &&
    previousToken.sourceCode === token.sourceCode &&
    previousToken.annotationId === token.annotationId

  if (previousToken && canMergeWithPrevious) {
    previousToken.text += token.text
    return
  }

  tokens.push(token)
}

export function parseTajwidToTokens(rawText: string): TajwidToken[] {
  if (!rawText) return []

  const tokens: TajwidToken[] = []
  let cursor = 0
  let plainTextStart = 0

  while (cursor < rawText.length) {
    if (rawText[cursor] !== '[') {
      cursor += 1
      continue
    }

    const parsedAnnotation = readTajwidAnnotation(rawText, cursor)
    if (!parsedAnnotation) {
      cursor += 1
      continue
    }

    appendTajwidToken(tokens, {
      text: rawText.slice(plainTextStart, cursor),
      rule: null,
      sourceCode: null,
      annotationId: null,
    })
    appendTajwidToken(tokens, parsedAnnotation.token)

    cursor = parsedAnnotation.nextIndex
    plainTextStart = cursor
  }

  appendTajwidToken(tokens, {
    text: rawText.slice(plainTextStart),
    rule: null,
    sourceCode: null,
    annotationId: null,
  })

  return tokens
}

// Adaptateur HTML historique, conservé jusqu'à la migration du rendu Vue.

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
