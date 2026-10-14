import { Capacitor } from '@capacitor/core'
import { onScopeDispose } from 'vue'

type BackHandler = () => boolean | Promise<boolean>
const handlers: BackHandler[] = []

export function useAndroidBackHandler(handler: BackHandler) {
  if (typeof window === 'undefined' || Capacitor.getPlatform() !== 'android') return
  handlers.push(handler)
  onScopeDispose(() => {
    const index = handlers.indexOf(handler)
    if (index >= 0) handlers.splice(index, 1)
  })
}

export async function dispatchAndroidBack() {
  for (const handler of [...handlers].reverse()) {
    if (await handler()) return true
  }
  return false
}
