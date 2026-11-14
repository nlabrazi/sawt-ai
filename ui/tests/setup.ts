import { afterEach, vi } from 'vitest'

// Happy DOM does not render animations; use Motion's JavaScript fallback in unit tests.
Object.defineProperty(Element.prototype, 'animate', { configurable: true, value: undefined })

afterEach(() => {
  vi.clearAllMocks()
  vi.useRealTimers()
})
