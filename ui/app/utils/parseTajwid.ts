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
